import numpy as np
import networkx as nx
import os
import random
import scipy.sparse as sp

def _sort_edge_array(edge_array):
    edge_array = np.asarray(edge_array, dtype=int)
    sort_idx = np.lexsort((edge_array[:, 1], edge_array[:, 0]))
    return edge_array[sort_idx]

def _read_and_remap_lcc(file_path):
    """读取原始边并在最大连通分量内进行节点重映射。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    print(f"正在读取原始数据: {file_path}")
    edge_list = []
    node_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            arr = line.strip().split()
            u, v, s = int(arr[0]), int(arr[1]), int(arr[2])
            edge_list.append((u, v, s))
            node_set.add(u)
            node_set.add(v)

    all_triples = np.asarray(edge_list, dtype=int)
    all_edges = all_triples[:, :2]

    G = nx.Graph()
    G.add_nodes_from(list(node_set))
    G.add_edges_from([tuple(e) for e in all_edges])
    connected_G = list(nx.connected_components(G))
    largest_G = max(connected_G, key=len)
    largest_node_set = set(largest_G)
    print(f"全边图有 {len(connected_G)} 个连通分量，最大分量包含 {len(largest_G)} 个节点")

    lcc_triples = [
        triple for triple in all_triples
        if triple[0] in largest_node_set and triple[1] in largest_node_set
    ]
    lcc_triples = np.asarray(lcc_triples, dtype=int)

    lcc_nodes = set(lcc_triples[:, 0]) | set(lcc_triples[:, 1])
    sorted_nodes = sorted(list(lcc_nodes))
    node2id = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    print(f"节点重映射完成: {num_nodes} 个节点，ID范围 0-{num_nodes - 1}")

    remapped_triples = np.empty_like(lcc_triples)
    remapped_triples[:, 0] = np.vectorize(node2id.get)(lcc_triples[:, 0])
    remapped_triples[:, 1] = np.vectorize(node2id.get)(lcc_triples[:, 1])
    remapped_triples[:, 2] = lcc_triples[:, 2]
    return remapped_triples, num_nodes

def _build_unsigned_adjacency(remapped_triples, num_nodes):
    """构建无向无权邻接矩阵，用于计算度与共同邻居。"""
    edges = remapped_triples[:, :2].astype(int)
    row = edges[:, 0]
    col = edges[:, 1]
    # 使用更宽整型，避免CN计算时出现int8溢出导致负值。
    data = np.ones(len(edges), dtype=np.int32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.int32)
    A = A + A.T
    A.data[:] = 1
    # setdiag会改稀疏结构，先转LIL可避免效率告警。
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def _compute_degree_product_scores(remapped_triples, num_nodes):
    A = _build_unsigned_adjacency(remapped_triples, num_nodes)
    degrees = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.int64, copy=False)
    edges = remapped_triples[:, :2].astype(int)
    return degrees[edges[:, 0]] * degrees[edges[:, 1]]

def _compute_cn_scores(remapped_triples, num_nodes):
    A = _build_unsigned_adjacency(remapped_triples, num_nodes)
    cn_matrix = A.dot(A).tocsr()
    edges = remapped_triples[:, :2].astype(int)
    return np.asarray(cn_matrix[edges[:, 0], edges[:, 1]]).reshape(-1).astype(np.int64, copy=False)

def _get_mst_mandatory_mask(remapped_triples, num_nodes):
    """
    按现有IID切分逻辑先构造MST，并将对应无向边全部强制放入训练集。
    返回：mandatory_mask, mst_edge_count
    """
    edges_for_mst = remapped_triples[:, :2].astype(np.int64)

    mst_G = nx.Graph()
    mst_G.add_nodes_from(range(num_nodes))
    mst_G.add_edges_from([tuple(e) for e in edges_for_mst], weight=1.0)
    mst = nx.minimum_spanning_tree(mst_G)

    mst_edge_pairs = np.array(list(mst.edges()), dtype=np.int64)
    if mst_edge_pairs.size == 0:
        return np.zeros(len(remapped_triples), dtype=bool), 0

    mst_low = np.minimum(mst_edge_pairs[:, 0], mst_edge_pairs[:, 1])
    mst_high = np.maximum(mst_edge_pairs[:, 0], mst_edge_pairs[:, 1])
    mst_keys = mst_low * np.int64(num_nodes) + mst_high

    low = np.minimum(edges_for_mst[:, 0], edges_for_mst[:, 1])
    high = np.maximum(edges_for_mst[:, 0], edges_for_mst[:, 1])
    edge_keys = low * np.int64(num_nodes) + high

    mandatory_mask = np.isin(edge_keys, mst_keys)
    return mandatory_mask, len(mst_edge_pairs)

def _shuffle_and_concat(index_groups, seed):
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g, dtype=int) for g in index_groups if len(g) > 0]
    if not groups:
        return np.array([], dtype=int)
    merged = np.concatenate(groups)
    rng.shuffle(merged)
    return merged

def _split_degree_shift_indices(remapped_triples, scores, mandatory_mask, train_ratio=0.4, val_ratio=0.1, seed=1):
    signs = remapped_triples[:, 2]
    mandatory_idx = np.where(mandatory_mask)[0]
    train_groups, val_groups, test_groups = [mandatory_idx], [], []

    for sign_mask in [signs > 0, signs <= 0]:
        idx = np.where(sign_mask & (~mandatory_mask))[0]
        idx_sorted = idx[np.argsort(-scores[idx])]
        n = len(idx_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_groups.append(idx_sorted[:train_end])
        val_groups.append(idx_sorted[train_end:val_end])
        test_groups.append(idx_sorted[val_end:])

    train_idx = _shuffle_and_concat(train_groups, seed)
    val_idx = _shuffle_and_concat(val_groups, seed + 1)
    test_idx = _shuffle_and_concat(test_groups, seed + 2)
    return train_idx, val_idx, test_idx

def _split_shortcut_shift_indices(
    remapped_triples,
    cn_scores,
    mandatory_mask,
    cn_train_min=3,
    cn_test_max=1,
    fallback_val_frac=0.1,
    seed=1,
):
    signs = remapped_triples[:, 2]
    mandatory_idx = np.where(mandatory_mask)[0]
    train_groups, val_groups, test_groups = [mandatory_idx], [], []
    rng = np.random.default_rng(seed)

    for sign_mask in [signs > 0, signs <= 0]:
        idx = np.where(sign_mask & (~mandatory_mask))[0]
        idx_train_pool = idx[cn_scores[idx] >= cn_train_min]
        idx_test = idx[cn_scores[idx] <= cn_test_max]
        idx_val = idx[(cn_scores[idx] > cn_test_max) & (cn_scores[idx] < cn_train_min)]

        mandatory_sign_cnt = int(np.sum(sign_mask & mandatory_mask))

        # 如果中间区间不足，按比例从训练池中切出验证集
        if len(idx_val) == 0 and len(idx_train_pool) > 1:
            idx_train_pool = idx_train_pool.copy()
            rng.shuffle(idx_train_pool)
            val_size = max(1, int(len(idx_train_pool) * fallback_val_frac))
            val_size = min(val_size, len(idx_train_pool) - 1)
            idx_val = idx_train_pool[:val_size]
            idx_train_pool = idx_train_pool[val_size:]

        if len(idx_train_pool) == 0 and mandatory_sign_cnt == 0:
            raise ValueError("Shortcut-Shift 切分失败：训练集为空，请降低 cn_train_min")
        if len(idx_test) == 0:
            raise ValueError("Shortcut-Shift 切分失败：测试集为空，请增大 cn_test_max")

        train_groups.append(idx_train_pool)
        val_groups.append(idx_val)
        test_groups.append(idx_test)

    train_idx = _shuffle_and_concat(train_groups, seed)
    val_idx = _shuffle_and_concat(val_groups, seed + 1)
    test_idx = _shuffle_and_concat(test_groups, seed + 2)
    return train_idx, val_idx, test_idx

def _ensure_train_has_all_nodes(train_idx, val_idx, test_idx, remapped_triples, num_nodes, seed=1):
    """兜底保障：若仍有节点未进入训练集，则从val/test移动相关边到train。"""
    split_tag = np.full(len(remapped_triples), -1, dtype=np.int8)
    split_tag[train_idx] = 0
    split_tag[val_idx] = 1
    split_tag[test_idx] = 2

    covered = np.zeros(num_nodes, dtype=bool)
    train_edges = remapped_triples[split_tag == 0, :2].astype(int)
    if len(train_edges) > 0:
        covered[train_edges[:, 0]] = True
        covered[train_edges[:, 1]] = True

    missing_nodes = np.where(~covered)[0]
    if len(missing_nodes) == 0:
        return train_idx, val_idx, test_idx, 0

    moved = 0
    rng = np.random.default_rng(seed + 2026)
    for node_id in missing_nodes:
        candidates = np.where(
            ((remapped_triples[:, 0] == node_id) | (remapped_triples[:, 1] == node_id))
            & (split_tag != 0)
        )[0]
        if len(candidates) == 0:
            continue
        chosen = int(candidates[rng.integers(len(candidates))])
        if split_tag[chosen] != 0:
            split_tag[chosen] = 0
            moved += 1

    covered_after = np.zeros(num_nodes, dtype=bool)
    train_edges_after = remapped_triples[split_tag == 0, :2].astype(int)
    if len(train_edges_after) > 0:
        covered_after[train_edges_after[:, 0]] = True
        covered_after[train_edges_after[:, 1]] = True
    still_missing = np.where(~covered_after)[0]
    if len(still_missing) > 0:
        raise ValueError(
            f"无法保证训练集覆盖所有节点，仍缺失 {len(still_missing)} 个节点: {still_missing[:10]}"
        )

    train_idx_new = _shuffle_and_concat([np.where(split_tag == 0)[0]], seed + 10)
    val_idx_new = _shuffle_and_concat([np.where(split_tag == 1)[0]], seed + 11)
    test_idx_new = _shuffle_and_concat([np.where(split_tag == 2)[0]], seed + 12)
    return train_idx_new, val_idx_new, test_idx_new, moved

def _print_split_stats(split_name, split_data, score_values=None, score_name='score'):
    pos_cnt = int(np.sum(split_data[:, 2] > 0))
    neg_cnt = int(np.sum(split_data[:, 2] <= 0))
    total = len(split_data)
    pos_ratio = (pos_cnt / total) if total > 0 else 0.0
    print(f"[{split_name}] 边数={total}, 正边={pos_cnt}, 负边={neg_cnt}, 正边占比={pos_ratio:.2%}")
    if score_values is not None and total > 0:
        print(
            f"[{split_name}] {score_name}: "
            f"mean={float(np.mean(score_values)):.4f}, "
            f"p50={float(np.percentile(score_values, 50)):.4f}, "
            f"p90={float(np.percentile(score_values, 90)):.4f}"
        )

def split_data_ood(dataset, file_path, args):
    """
    OOD切分入口：
    - degree_shift: 按度乘积 Top/Mid/Bottom 切分（默认 40/10/50）。
    - shortcut_shift: 按共同邻居CN阈值切分（默认 Train: CN>=3, Test: CN<=1）。
    """
    remapped_triples, num_nodes = _read_and_remap_lcc(file_path)
    mandatory_mask, mst_edge_count = _get_mst_mandatory_mask(remapped_triples, num_nodes)
    mandatory_cnt = int(np.sum(mandatory_mask))
    print(f"MST包含 {mst_edge_count} 条无向边，对应 {mandatory_cnt} 条样本边将强制进入训练集")

    strategy = args.ood_strategy
    output_dataset = args.output_dataset.strip() if args.output_dataset else ''
    if not output_dataset:
        output_dataset = f"{dataset}-{strategy}"

    if strategy == 'degree_shift':
        score_values_all = _compute_degree_product_scores(remapped_triples, num_nodes)
        train_idx, val_idx, test_idx = _split_degree_shift_indices(
            remapped_triples,
            score_values_all,
            mandatory_mask=mandatory_mask,
            train_ratio=args.ood_train_ratio,
            val_ratio=args.ood_val_ratio,
            seed=args.seed,
        )
        score_name = 'degree_product'
    elif strategy == 'shortcut_shift':
        score_values_all = _compute_cn_scores(remapped_triples, num_nodes)
        train_idx, val_idx, test_idx = _split_shortcut_shift_indices(
            remapped_triples,
            score_values_all,
            mandatory_mask=mandatory_mask,
            cn_train_min=args.cn_train_min,
            cn_test_max=args.cn_test_max,
            fallback_val_frac=args.fallback_val_frac,
            seed=args.seed,
        )
        score_name = 'common_neighbors'
    else:
        raise ValueError(f"Unsupported ood_strategy: {strategy}")

    train_idx, val_idx, test_idx, moved_cnt = _ensure_train_has_all_nodes(
        train_idx,
        val_idx,
        test_idx,
        remapped_triples,
        num_nodes,
        seed=args.seed,
    )
    if moved_cnt > 0:
        print(f"节点覆盖兜底: 已从验证/测试集移动 {moved_cnt} 条边到训练集")
    else:
        print("节点覆盖检查通过: 训练集已覆盖全部节点")

    train_data = _sort_edge_array(remapped_triples[train_idx])
    val_data = _sort_edge_array(remapped_triples[val_idx])
    test_data = _sort_edge_array(remapped_triples[test_idx])

    train_scores = score_values_all[train_idx]
    val_scores = score_values_all[val_idx]
    test_scores = score_values_all[test_idx]

    print("OOD数据集分割完成:")
    _print_split_stats('Train', train_data, train_scores, score_name)
    _print_split_stats('Val', val_data, val_scores, score_name)
    _print_split_stats('Test', test_data, test_scores, score_name)

    save_dir = f'Data/{output_dataset}'
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f'{save_dir}/{output_dataset}_train.txt', train_data, fmt='%d')
    np.savetxt(f'{save_dir}/{output_dataset}_val.txt', val_data, fmt='%d')
    np.savetxt(f'{save_dir}/{output_dataset}_test.txt', test_data, fmt='%d')
    print(f"OOD数据已保存到 {save_dir}")
    return output_dataset

def split_data(dataset, file_path, args):
    """
    读取原始数据，首先提取最大连通分量，然后进行节点重映射，最后切分数据集并保存。
    """
    test_frac = args.test_frac
    val_frac = args.val_frac
    save_dir = f'Data/{dataset}'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    # 1. 读取数据并提取最大连通分量 (LCC)
    print(f"正在读取原始数据: {file_path}")
    edge_list = []
    node_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            arr = line.strip().split()
            u, v, s = int(arr[0]), int(arr[1]), int(arr[2])
            edge_list.append((u, v, s))
            node_set.add(u)
            node_set.add(v)

    all_triples = np.array(edge_list)
    all_edges = all_triples[:, :2]

    G = nx.Graph()
    G.add_nodes_from(list(node_set))
    G.add_edges_from([tuple(e) for e in all_edges])
    connected_G = list(nx.connected_components(G))
    largest_G = max(connected_G, key=len)
    largest_node_set = set(largest_G)
    print(f"全边图有 {len(connected_G)} 个连通分量，最大分量包含 {len(largest_G)} 个节点")

    # 过滤出LCC中的三元组
    lcc_triples = [triple for triple in all_triples
                   if triple[0] in largest_node_set and triple[1] in largest_node_set]
    lcc_triples = np.array(lcc_triples)

    # 2. 节点重映射 (在LCC基础上)
    lcc_nodes = set(lcc_triples[:, 0]) | set(lcc_triples[:, 1])
    sorted_nodes = sorted(list(lcc_nodes))
    node2id = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    print(f"节点重映射完成: {num_nodes} 个节点，ID范围 0-{num_nodes-1}")

    # 将三元组转换为新ID
    remapped_triples = []
    for u, v, s in lcc_triples:
        remapped_triples.append([node2id[u], node2id[v], s])
    remapped_triples = np.array(remapped_triples)

    # 3. 基于MST切分数据集
    # 使用重映射后的边构建MST
    edges_for_mst = remapped_triples[:, :2]
    mst_G = nx.Graph()
    mst_G.add_nodes_from(range(num_nodes))
    mst_G.add_edges_from([tuple(e) for e in edges_for_mst], weight=1.0)
    mst = nx.minimum_spanning_tree(mst_G)
    mst_edge_set = set((min(u, v), max(u, v)) for u, v in mst.edges())
    print(f"MST包含 {len(mst_edge_set)} 条边，将全部分配到训练集")

    train_triples, val_triples, test_triples = [], [], []

    mst_pos_count = 0
    mst_neg_count = 0

    # 分离MST边和非MST边
    non_mst_pos = []
    non_mst_neg = []

    for triple in remapped_triples:
        u, v, s = triple
        edge_key = (min(u, v), max(u, v))
        if edge_key in mst_edge_set:
            train_triples.append(triple)
            if s == 1:
                mst_pos_count += 1
            else:
                mst_neg_count += 1
        else:
            if s == 1:
                non_mst_pos.append(triple)
            else:
                non_mst_neg.append(triple)

    # 计算分配数量
    total_pos = mst_pos_count + len(non_mst_pos)
    total_neg = mst_neg_count + len(non_mst_neg)

    target_train_pos = int(total_pos * (1 - test_frac - val_frac))
    target_val_pos = int(total_pos * val_frac)

    target_train_neg = int(total_neg * (1 - test_frac - val_frac))
    target_val_neg = int(total_neg * val_frac)

    needed_train_pos = max(0, target_train_pos - mst_pos_count)
    needed_train_neg = max(0, target_train_neg - mst_neg_count)

    # 随机打乱并分配
    random.shuffle(non_mst_pos)
    random.shuffle(non_mst_neg)

    train_pos = non_mst_pos[:needed_train_pos]
    val_pos = non_mst_pos[needed_train_pos : needed_train_pos + target_val_pos]
    test_pos = non_mst_pos[needed_train_pos + target_val_pos:]

    train_neg = non_mst_neg[:needed_train_neg]
    val_neg = non_mst_neg[needed_train_neg : needed_train_neg + target_val_neg]
    test_neg = non_mst_neg[needed_train_neg + target_val_neg:]

    train_triples.extend(train_pos + train_neg)
    val_triples.extend(val_pos + val_neg)
    test_triples.extend(test_pos + test_neg)

    # 4. 保存数据
    train_data = _sort_edge_array(train_triples)
    val_data = _sort_edge_array(val_triples)
    test_data = _sort_edge_array(test_triples)

    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_data)} 条边")
    print(f"  验证集: {len(val_data)} 条边")
    print(f"  测试集: {len(test_data)} 条边")

    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f'{save_dir}/{dataset}_train.txt', train_data, fmt='%d')
    np.savetxt(f'{save_dir}/{dataset}_val.txt', val_data, fmt='%d')
    np.savetxt(f'{save_dir}/{dataset}_test.txt', test_data, fmt='%d')
    print(f"数据已保存到 {save_dir}")

def load_data(dataset,file_path,args,logger=None):
    # 直接读取当前目录下 Data/数据集名/数据集名_{split}.txt
    data_dir = f'Data/{dataset}'

    train_path = os.path.join(data_dir, f'{dataset}_train.txt')
    val_path = os.path.join(data_dir, f'{dataset}_val.txt')
    test_path = os.path.join(data_dir, f'{dataset}_test.txt')

    # 3. 读取数据
    train_data = np.loadtxt(train_path, dtype=int, ndmin=2)
    val_data = np.loadtxt(val_path, dtype=int, ndmin=2)
    test_data = np.loadtxt(test_path, dtype=int, ndmin=2)

    # 获取最大节点ID用于构建矩阵
    max_node = max(train_data[:, 0].max(), train_data[:, 1].max())
    max_node = max(max_node, val_data[:, 0].max(), val_data[:, 1].max())
    max_node = max(max_node, test_data[:, 0].max(), test_data[:, 1].max())

    num_nodes = int(max_node + 1)

    # 4. 构建稀疏矩阵
    rows, cols, data = [], [], []
    pos_rows, pos_cols, pos_data = [], [], []
    neg_rows, neg_cols, neg_data = [], [], []

    for source, target, sign in train_data:
        source, target = int(source), int(target)
        rows.extend([source, target])
        cols.extend([target, source])
        data.extend([sign, sign])
        if sign > 0:
            pos_rows.extend([source, target])
            pos_cols.extend([target, source])
            pos_data.extend([1, 1])
        else:
            neg_rows.extend([source, target])
            neg_cols.extend([target, source])
            neg_data.extend([1, 1])

    # 创建稀疏矩阵
    signed_adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.int8)
    return train_data, val_data, test_data, signed_adj

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Bitcoin-Alpha')
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--split_mode', type=str, default='iid', choices=['iid', 'ood'])
    parser.add_argument('--ood_strategy', type=str, default='degree_shift', choices=['degree_shift', 'shortcut_shift'])
    parser.add_argument('--ood_train_ratio', type=float, default=0.4)
    parser.add_argument('--ood_val_ratio', type=float, default=0.1)
    parser.add_argument('--cn_train_min', type=int, default=3)
    parser.add_argument('--cn_test_max', type=int, default=1)
    parser.add_argument('--fallback_val_frac', type=float, default=0.1)
    parser.add_argument('--output_dataset', type=str, default='')
    args = parser.parse_args()
    dataset = args.dataset
    Data_file = f"Data/data_file/{dataset}.txt"

    if args.split_mode == 'ood':
        final_dataset = split_data_ood(dataset, Data_file, args)
    else:
        split_data(dataset, Data_file, args)
        final_dataset = dataset

    train_data, val_data, test_data, signed_adj = load_data(final_dataset, Data_file, args)
    print(f'训练集: 正边={np.sum(train_data[:, 2] == 1)}, 负边={np.sum(train_data[:, 2] == -1)}')
    print(f'验证集: 正边={np.sum(val_data[:, 2] == 1)}, 负边={np.sum(val_data[:, 2] == -1)}')
    print(f'测试集: 正边={np.sum(test_data[:, 2] == 1)}, 负边={np.sum(test_data[:, 2] == -1)}')
    neg_cnt = np.sum(train_data[:, 2] == -1)
    if neg_cnt > 0:
        print(f'正负边比例: {np.sum(train_data[:, 2] == 1)/neg_cnt:.3f}:1')
    else:
        print('正负边比例: inf:1 (训练集无负边)')
