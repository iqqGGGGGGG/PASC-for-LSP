import numpy as np
import networkx as nx
import os
import random
import scipy.sparse as sp

def _sort_edge_array(edge_array):
    edge_array = np.asarray(edge_array, dtype=int)
    sort_idx = np.lexsort((edge_array[:, 1], edge_array[:, 0]))
    return edge_array[sort_idx]

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
    save_dir = f'Data/{dataset}'
    train_path = os.path.join(save_dir, f'{dataset}_train.txt')
    val_path = os.path.join(save_dir, f'{dataset}_val.txt')
    test_path = os.path.join(save_dir, f'{dataset}_test.txt')

    train_data = np.loadtxt(train_path, dtype=int, ndmin=2)
    val_data = np.loadtxt(val_path, dtype=int, ndmin=2)
    test_data = np.loadtxt(test_path, dtype=int, ndmin=2)
    num_nodes = int(max(train_data[:, 0].max(), train_data[:, 1].max()) + 1)

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
    pos_adj = sp.csr_matrix((pos_data, (pos_rows, pos_cols)), shape=(num_nodes, num_nodes), dtype=np.int8)
    neg_adj = sp.csr_matrix((neg_data, (neg_rows, neg_cols)), shape=(num_nodes, num_nodes), dtype=np.int8)

    return train_data, val_data, test_data, signed_adj, pos_adj, neg_adj

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Bitcoin-Alpha')
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--val_frac', type=float, default=0.1)
    args = parser.parse_args()
    dataset = args.dataset
    Data_file = f"Data/data_file/{dataset}.txt"
    split_data(dataset, Data_file, args)
    train_data,val_data,test_data,signed_adj,pos_adj,neg_adj = load_data(dataset, Data_file, args)
    print(f'训练集: 正边={np.sum(train_data[:, 2] == 1)}, 负边={np.sum(train_data[:, 2] == -1)}')
    print(f'验证集: 正边={np.sum(val_data[:, 2] == 1)}, 负边={np.sum(val_data[:, 2] == -1)}')
    print(f'测试集: 正边={np.sum(test_data[:, 2] == 1)}, 负边={np.sum(test_data[:, 2] == -1)}')
    print(f'正负边比例: {np.sum(train_data[:, 2] == 1)/np.sum(train_data[:, 2] == -1):.3f}:1')
