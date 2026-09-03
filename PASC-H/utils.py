import logging
import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

import torch
from torch import Tensor

def _ensure_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)

def get_signed_laplacian_features(adj, logger, k=128, which='SA'):
    """
    计算符号拉普拉斯矩阵的特征向量作为节点嵌入。
    使用 Torch.svd_lowrank 加速运算 (SSE近似)。
    原理: Normalized Signed Laplacian L_sym = I - D_bar^-1/2 A D_bar^-1/2
    A_sym = D_bar^-1/2 A D_bar^-1/2 的 Top SVD 分量包含主要的聚类结构信息。
    """
    logger.info(f"🚀 启动 Torch GPU 加速符号谱嵌入 (Rank={k})...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_csr = _ensure_csr(adj)
    num_nodes = adj_csr.shape[0]

    try:
        adj_t = torch.from_numpy(adj_csr.toarray()).float().to(device)
        d_bar = torch.sum(torch.abs(adj_t), dim=1)
        d_bar[d_bar == 0] = 1.0
        inv_sqrt_d = torch.pow(d_bar, -0.5)
        a_sym = inv_sqrt_d.unsqueeze(1) * adj_t * inv_sqrt_d.unsqueeze(0)
        u, s, v = torch.svd_lowrank(a_sym, q=k)
        features = u * torch.sqrt(s)
        norms = torch.norm(features, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        features = features / norms
        logger.info(f"Torch SVD SSE 计算完成: {features.shape}")
        return features.cpu().numpy()

    except RuntimeError as re:
        logger.warning(f"GPU OOM or Runtime Error ({re}), trying CPU fallback...")
        try:
            return get_sparse_svd_features(adj_csr, logger, rank=k, normalize=True)
        except Exception as e2:
            logger.error(f"CPU Fallback failed: {e2}")
            return get_sparse_svd_features(adj_csr, logger, rank=k, normalize=True)

    except Exception as e:
        logger.error(f"Torch SVD Method failed: {e}. Fallback to Random.")
        return get_sparse_svd_features(adj_csr, logger, rank=k, normalize=True)

def get_sparse_svd_features(adj_matrix, logger, rank=128, normalize=True, random_state=42, which='LM'):
    """使用稀疏SVD生成节点嵌入。
    Args:
        adj_matrix (sp.spmatrix | np.ndarray): 原始邻接矩阵，可以是稀疏或稠密矩阵。
        logger: 日志记录器。
        rank (int): 保留的奇异值数量，也对应嵌入维度。
        normalize (bool): 是否对生成的嵌入按行归一化。
        random_state (int | None): SVD 随机种子。
        which (str): 奇异值类型，同 `svds` 的 `which` 参数。
    Returns:
        np.ndarray: 形状为 (num_nodes, rank) 的节点嵌入矩阵。
    """
    adj_csr = _ensure_csr(adj_matrix).astype(np.float32)
    num_nodes, num_cols = adj_csr.shape
    if adj_csr.nnz == 0:
        logger.warning("稀疏SVD: 邻接矩阵为空，返回零向量嵌入。")
        return np.zeros((num_nodes, max(1, min(rank, num_cols))), dtype=np.float32)

    max_rank = max(1, min(num_nodes, num_cols) - 1)
    effective_rank = max(1, min(rank, max_rank))
    if effective_rank < rank:
        logger.info(f"稀疏SVD: 将rank从{rank}裁剪至{effective_rank}以满足矩阵维度限制。")
    logger.info(f"开始执行稀疏SVD: 节点数={num_nodes}, 有效rank={effective_rank}, 非零元素={adj_csr.nnz}")

    try:
        u, s, _ = svds(
            adj_csr,
            k=effective_rank,
            which=which,
            return_singular_vectors=True,
            random_state=random_state,
        )
    except Exception as err:
        logger.warning(f"稀疏SVD失败({err})，尝试使用稠密SVD作为回退方案。")
        try:
            if num_nodes * num_cols > 5000000:
                raise MemoryError("矩阵规模过大，不适合稠密SVD回退。")
            dense_matrix = adj_csr.toarray().astype(np.float32)
            u_dense, s_dense, _ = np.linalg.svd(dense_matrix, full_matrices=False)
            u = u_dense[:, :effective_rank]
            s = s_dense[:effective_rank]
        except Exception as fallback_err:
            logger.error(
                f"稠密SVD回退同样失败({fallback_err})，返回零嵌入以保持流程可继续。"
            )
            return np.zeros((num_nodes, effective_rank), dtype=np.float32)

    # svds 返回的奇异值是升序，需要反转为降序
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    u = u[:, sort_idx]

    # 对应谱嵌入：U * sqrt(S)
    embeddings = u * np.sqrt(s)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    logger.info(f"稀疏SVD完成,嵌入维度: {embeddings.shape}")
    return embeddings

def get_node_features(args, adj, train_data, num_nodes, device, logger):
    feature_type = getattr(args, 'feature_type', 'svd')
    # Switch to Laplacian features cache
    features_cache_file = os.path.join(
        args.datapath,
        'cache',
        f'{args.file_name}_{args.dataset}_features_{feature_type}.pkl',
    )

    if os.path.exists(features_cache_file):
        try:
            features = pickle.load(open(features_cache_file, "rb"))
            # Check dimension robustly
            if features.shape[1] == args.svd_rank:
                logger.info(f"加载缓存的特征 ({feature_type}), 维度: {features.shape}")
                return torch.FloatTensor(features).to(device)
            else:
                 logger.info("缓存维度不匹配，重新计算...")
        except:
            pass

    logger.info(f'开始生成符号谱特征 (Method: {feature_type})...')
    svd_rank = args.svd_rank

    if feature_type == 'laplacian':
        features = get_signed_laplacian_features(adj, logger, k=svd_rank)
    else:
        features = get_sparse_svd_features(adj, logger, rank=svd_rank)

    os.makedirs(os.path.dirname(features_cache_file), exist_ok=True)
    pickle.dump(features, open(features_cache_file, "wb"))
    logger.info(f"特征生成完成，维度: {features.shape}")
    return torch.FloatTensor(features).to(device)


# 获取日志记录器
def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

def structured_negative_sampling(edge_index, num_nodes=None):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j

    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))
