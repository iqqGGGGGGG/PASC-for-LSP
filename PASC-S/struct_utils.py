import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from torch_geometric_signed_directed import SSSNET_node_clustering
from models import PriorScoreNet

def pretrain_psn(Z_grad, community_labels, train_data, device, logger,epochs=50):
    dim_z = Z_grad.shape[1]
    prior_score_net = PriorScoreNet(dim_z).to(device)
    optimizer = optim.Adam(prior_score_net.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    edges = torch.from_numpy(train_data[:, :2]).long().to(device)

    u = edges[:, 0]
    v = edges[:, 1]

    if not isinstance(community_labels, torch.Tensor):
        community_labels = torch.from_numpy(community_labels).long().to(device)
    else:
        community_labels = community_labels.to(device)

    T_fact_labels = (community_labels[u] == community_labels[v]).float().view(-1, 1)

    Z_grad = Z_grad.to(device)

    print(f"Start PriorScoreWarmUp ({epochs} epochs)...")
    prior_score_net.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        z_u = Z_grad[u]
        z_v = Z_grad[v]

        logits = prior_score_net(z_u, z_v)
        loss = criterion(logits, T_fact_labels)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0:
            acc = ((torch.sigmoid(logits) > 0.5) == (T_fact_labels > 0.5)).float().mean()
            if acc > 0.7 and ep > 30:
                logger.info(f"  [PriorScoreWarmUp] Early Stopping at Ep {ep+1} (Acc > 0.7)")
                break
            logger.info(f"  [PriorScoreWarmUp] Ep {ep+1}: Loss={loss.item():.4f} Acc={acc.item():.4f}")

    prior_score_net.eval()
    return prior_score_net

def load_comm_labels(args, comm_cache, adj, node, logger, train_data, features):
    device = torch.device(f'cuda:{args.gpu}')

    if os.path.exists(comm_cache):
        logger.info('Community label cache exists, skipping detection')
        with open(comm_cache, 'rb') as f:
            T_data = pickle.load(f)
            if 'community_labels' in T_data:
                community_labels = T_data['community_labels']
            elif 'cluster_labels' in T_data:
                community_labels = T_data['cluster_labels']
            else:
                community_labels = T_data
            if isinstance(community_labels, np.ndarray):
                community_labels = torch.from_numpy(community_labels).long().to(device)
            else:
                community_labels = community_labels.to(device)


    else:
        logger.info('Community label cache not found, computing')
        k_clusters = 2
        community_labels_np = community_detect(args, adj, logger, k=k_clusters)
        community_labels = torch.from_numpy(community_labels_np).long().to(device)

        T_data = {'community_labels': community_labels}
        with open(comm_cache, 'wb') as f:
            pickle.dump(T_data, f)

    return community_labels




def community_detect(args, edge_signs, logger, k, hidden_dim=128, dropout=0.2, hop=2, epochs=400, lr=0.005, patience=10):
    logger.info("Starting signed community detection via SSSNET")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    signed_coo = edge_signs.tocoo()
    pos_rows, pos_cols, pos_vals = [], [], []
    neg_rows, neg_cols, neg_vals = [], [], []

    for i, j, v in zip(signed_coo.row, signed_coo.col, signed_coo.data):
        if i == j: continue
        if v > 0:
            pos_rows.append(i); pos_cols.append(j); pos_vals.append(1.0)
        elif v < 0:
            neg_rows.append(i); neg_cols.append(j); neg_vals.append(1.0)

    edge_index_p = torch.tensor([pos_rows, pos_cols], dtype=torch.long, device=device) if pos_rows else torch.empty((2,0), dtype=torch.long, device=device)
    edge_index_n = torch.tensor([neg_rows, neg_cols], dtype=torch.long, device=device) if neg_rows else torch.empty((2,0), dtype=torch.long, device=device)
    edge_weight_p = torch.tensor(pos_vals, dtype=torch.float32, device=device) if pos_vals else torch.empty((0,), dtype=torch.float32, device=device)
    edge_weight_n = torch.tensor(neg_vals, dtype=torch.float32, device=device) if neg_vals else torch.empty((0,), dtype=torch.float32, device=device)

    num_nodes = edge_signs.shape[0]
    features_cache_file = os.path.join(
        args.datapath,
        'cache',
        f'{args.file_name}_{args.dataset}_features_{args.feature_type}.pkl',
    )
    node_embs_raw = pickle.load(open(features_cache_file, 'rb'))
    node_features = torch.tensor(np.array(node_embs_raw), dtype=torch.float32, device=device)

    model = SSSNET_node_clustering(nfeat=node_features.shape[1], hidden=hidden_dim, nclass=k, dropout=dropout, hop=hop, fill_value=1.0, directed=False, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    best_modularity = -float('inf')
    best_labels = None
    patience_counter = 0
    min_clusters = max(2, int(k * 0.3))

    for epoch in range(epochs):
        optimizer.zero_grad()
        z, output, predictions_cluster, prob = model(edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, node_features)

        reg_loss = torch.norm(z, p=2) * 0.001

        cluster_soft_counts = torch.sum(prob, dim=0)
        ideal_size = num_nodes / k
        balance_loss = torch.mean((cluster_soft_counts - ideal_size) ** 2) / (ideal_size ** 2) * 2.0

        eps = 1e-8
        pos_src, pos_dst = edge_index_p[0], edge_index_p[1]
        prob_pos_src = prob[pos_src]
        prob_pos_dst = prob[pos_dst]
        same_prob_pos = torch.sum(prob_pos_src * prob_pos_dst, dim=1)
        same_prob_pos = torch.clamp(same_prob_pos, eps, 1 - eps)
        loss_pos = -torch.mean(torch.log(same_prob_pos))

        neg_src, neg_dst = edge_index_n[0], edge_index_n[1]
        prob_neg_src = prob[neg_src]
        prob_neg_dst = prob[neg_dst]
        same_prob_neg = torch.sum(prob_neg_src * prob_neg_dst, dim=1)
        same_prob_neg = torch.clamp(same_prob_neg, eps, 1 - eps)
        loss_neg = -torch.mean(torch.log(1 - same_prob_neg))

        structure_loss = 0.5 * (loss_pos + loss_neg)
        ortho_loss = torch.norm(torch.mm(z.T, z) - torch.eye(z.shape[1], device=device), p='fro') * 0.001

        total_loss = reg_loss + balance_loss + structure_loss + ortho_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            current_labels = torch.argmax(prob, dim=1).cpu().numpy()
            unique_labels = np.unique(current_labels)
            n_clusters = len(unique_labels)

            cluster_sizes = [np.sum(current_labels == l) for l in unique_labels]
            balance_score = min(cluster_sizes) / (max(cluster_sizes) + 1e-5)
            score = n_clusters * balance_score

            logger.info(f"SSSNET Epoch {epoch}, Loss: {total_loss.item():.4f}, "
                       f"Clusters: {n_clusters}, BalanceLoss: {balance_loss.item():.4f}")

            if score > best_modularity and n_clusters >= min_clusters:
                best_modularity = score
                best_labels = current_labels.copy()
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"SSSNET early stopping at epoch {epoch}")
            break

    if best_labels is not None:
        labels = best_labels
        logger.info("Using best clustering result from training")
    else:
        model.eval()
        with torch.no_grad():
            _, _, _, prob = model(edge_index_p, edge_weight_p, edge_index_n, edge_weight_n, node_features)
            labels = torch.argmax(prob, dim=1).cpu().numpy()
        logger.info("Using final training result")

    unique_labels = np.unique(labels)
    logger.info(f"Signed community detection complete: {len(unique_labels)} communities")

    total_nodes_counted = 0
    for cluster_id in unique_labels:
        cluster_size = np.sum(labels == cluster_id)
        total_nodes_counted += cluster_size
        logger.info(f"Community {cluster_id}: {cluster_size} nodes")

    logger.info(f"Total nodes counted: {total_nodes_counted} / {num_nodes}")

    return labels


def compute_struct_grad(adj, features, logger):
    logger.info("Computing structural gradient Z_G = H_2 - H_1 ...")
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    else:
        features_np = features
    adj_abs = np.abs(adj)
    degrees = np.array(adj_abs.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    D_inv = sp.diags(1.0 / degrees, format='csr')
    A_norm = D_inv @ adj_abs
    H_1 = A_norm @ features_np
    H_2 = A_norm @ H_1
    Z_G = H_2 - H_1
    Z_tensor = torch.from_numpy(Z_G.astype(np.float32))
    logger.info(f"Structural gradient computed: Z_G.shape={Z_tensor.shape}")
    return Z_tensor
