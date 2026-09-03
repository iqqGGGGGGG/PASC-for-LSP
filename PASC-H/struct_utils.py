import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from torch_geometric_signed_directed import SSSNET_node_clustering
from models import PriorScoreNet
import torch.nn.functional as F

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
            if acc > 0.7 and ep >= 30:
                logger.info(f"  [PriorScoreWarmUp] Early Stopping at Ep {ep+1} (Acc > 0.7)")
                break
            logger.info(f"  [PriorScoreWarmUp] Ep {ep+1}: Loss={loss.item():.4f} Acc={acc.item():.4f}")

    prior_score_net.eval()
    return prior_score_net

def load_comm_labels(args, comm_cache, adj, node, logger, train_data, features, Z_grad):
    device = torch.device(f'cuda:{args.gpu}')
    community_labels = None
    twins_data = None

    if os.path.exists(comm_cache):
        logger.info(f'Community label cache exists, loading: {comm_cache}')
        try:
            with open(comm_cache, 'rb') as f:
                T_data = pickle.load(f)
            community_labels = T_data.get('community_labels', T_data.get('cluster_labels'))
            if not isinstance(community_labels, torch.Tensor):
                community_labels = torch.from_numpy(community_labels).long().to(device)
            else:
                community_labels = community_labels.to(device)
            twins_data = T_data.get('twins_data', T_data.get('train_data_cf'))

            prior_score_net = pretrain_psn(Z_grad, community_labels, train_data, device, logger, epochs=400)
            logger.info('Successfully loaded all data from cache')
            return community_labels, twins_data, prior_score_net
        except Exception as e:
            logger.warning(f'Failed to load cache: {e}, will recompute')
    k_clusters = 2
    community_labels_np = community_detect(args, adj, logger, k=k_clusters)
    community_labels = torch.from_numpy(community_labels_np).long().to(device)
    logger.info(f"Nodes: {node}, Community labels: {len(community_labels)}")
    prior_score_net = pretrain_psn(Z_grad, community_labels, train_data, device, logger, epochs=400)
    def psn_predictor(edges_np):
        u_np = edges_np[:, 0]
        v_np = edges_np[:, 1]
        u_t = torch.from_numpy(u_np).long().to(device)
        v_t = torch.from_numpy(v_np).long().to(device)
        with torch.no_grad():
            z_u = Z_grad[u_t]
            z_v = Z_grad[v_t]
            logits = prior_score_net(z_u, z_v)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy(), logits.cpu().numpy()
    twins_data = gen_twins(args, adj, features, community_labels, psn_predictor, Z_grad, logger, train_data)

    T_data_save = {
        'community_labels': community_labels_np,
        'twins_data': twins_data,
    }
    with open(comm_cache, 'wb') as f:
        pickle.dump(T_data_save, f)
    logger.info(f'Community labels and structural twins cached to {comm_cache}')
    save_dir = f'Data/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(f'{save_dir}/{args.dataset}_train_twins.txt', twins_data, fmt='%d')
    logger.info(f'Structural twin training data saved: {save_dir}/{args.dataset}_train_twins.txt')
    return community_labels, twins_data, prior_score_net


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

def gen_twins(args, adj, feature, community_labels, psn_predictor, Z_grad, logger, train_data,
                                        caliper_score=0.0001, threshold_struct=20.0,
                                        batch_size=8192,
                                        struct_quantiles=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    logger.info(f"Starting two-step structural twin matching: Caliper={caliper_score}")
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    quantile_tensor = torch.tensor(struct_quantiles, dtype=torch.float32)

    target_u = torch.from_numpy(train_data[:, 0].astype(np.int64)).long().to(device)
    target_v = torch.from_numpy(train_data[:, 1].astype(np.int64)).long().to(device)

    train_signs_raw = train_data[:, 2].astype(np.float32)
    train_signs_normalized = np.where(train_signs_raw > 0, 1.0, -1.0)
    edge_signs = torch.from_numpy(train_signs_normalized).to(device)

    edges_np = train_data[:, :2].astype(np.int64)
    logger.info(f"train_data: sign range=[{train_signs_raw.min()}, {train_signs_raw.max()}], normalized to +/-1")

    num_nodes = adj.shape[0]
    num_edges = target_u.size(0)
    node_feats = torch.from_numpy(feature).float().to(device)

    adj_coo = adj.tocoo()
    pool_u = torch.from_numpy(adj_coo.row).long().to(device)
    pool_v = torch.from_numpy(adj_coo.col).long().to(device)

    adj_signs_raw = adj_coo.data
    pool_signs = torch.where(
        torch.from_numpy(adj_signs_raw).float().to(device) > 0,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device)
    )

    logger.info(f"adj candidate pool: sign range=[{adj_signs_raw.min()}, {adj_signs_raw.max()}], normalized to +/-1, pool size={pool_u.size(0)}")

    sample_size = min(10000, num_nodes)
    idx1 = torch.randint(0, num_nodes, (sample_size,), device=device)
    idx2 = torch.randint(0, num_nodes, (sample_size,), device=device)
    sample_sim = F.cosine_similarity(node_feats[idx1], node_feats[idx2], dim=1)
    sample_dist = 1.0 - sample_sim
    dist_sample_size = 100000
    idx_a = torch.randint(0, num_nodes, (dist_sample_size,), device=device)
    idx_b = torch.randint(0, num_nodes, (dist_sample_size,), device=device)
    sim_samples = F.cosine_similarity(node_feats[idx_a], node_feats[idx_b], dim=1)
    dist_samples = 1.0 - sim_samples
    percentile_val = np.percentile(dist_samples.cpu().numpy(), threshold_struct)
    struct_threshold_current = float(percentile_val)

    logger.info(f"Threshold from sampling (Top {threshold_struct}%): {struct_threshold_current:.4f}")
    logger.info("Enhanced matching: fusing node features with structural gradient features")
    Z_feats = Z_grad.float().to(device)
    pred_output = psn_predictor(edges_np)
    scores = torch.from_numpy(pred_output[0]).float()
    scores = scores.view(-1).to(device)
    labels = community_labels.to(device)
    target_s = edge_signs.clone()

    pool_edges_np = np.stack([adj_coo.row, adj_coo.col], axis=1)
    pred_output = psn_predictor(pool_edges_np)
    pool_scores = torch.from_numpy(pred_output[0]).float().view(-1).to(device)

    target_pred_output = psn_predictor(edges_np)
    target_scores = torch.from_numpy(target_pred_output[0]).float().view(-1).to(device)

    pool_row = pool_u
    pool_col = pool_v
    is_same_community_pool = (labels[pool_row] == labels[pool_col])
    all_pool_indices = torch.arange(pool_u.size(0), device=device)

    intra_indices = all_pool_indices[is_same_community_pool]
    inter_indices = all_pool_indices[~is_same_community_pool]

    def sort_pool(indices):
        if indices.numel() == 0:
            return indices, torch.tensor([], device=device)
        pool_scores_subset = pool_scores[indices]
        sorted_scores, sort_idx = torch.sort(pool_scores_subset)
        sorted_indices = indices[sort_idx]
        return sorted_indices, sorted_scores

    sorted_intra_indices, sorted_intra_scores = sort_pool(intra_indices)
    sorted_inter_indices, sorted_inter_scores = sort_pool(inter_indices)

    logger.info(f"Candidate pool ready: Intra={intra_indices.numel()}, Inter={inter_indices.numel()}")

    is_intra_target = (labels[target_u] == labels[target_v])

    final_cf_u = target_u.clone()
    final_cf_v = target_v.clone()
    final_cf_s = target_s.clone()
    final_cf_t = is_intra_target.float()

    total_queries = target_u.size(0)
    matched_count = 0
    caliper_fail = 0
    struct_fail = 0

    struct_min_samples = []
    for start in range(0, total_queries, batch_size):
        end = min(start + batch_size, total_queries)
        batch_indices = torch.arange(start, end, device=device)

        batch_u = target_u[batch_indices]
        batch_v = target_v[batch_indices]
        batch_scores = target_scores[batch_indices]
        batch_is_intra = is_intra_target[batch_indices]
        batch_s = target_s[batch_indices]

        matched_before_batch = matched_count

        for is_intra_group in [True, False]:
            mask = (batch_is_intra == is_intra_group)
            if not mask.any():
                continue

            sub_indices = batch_indices[mask]
            sub_scores = batch_scores[mask]
            sub_u = batch_u[mask]
            sub_v = batch_v[mask]
            sub_s = batch_s[mask]

            if is_intra_group:
                pool_indices = sorted_inter_indices
                pool_scores_local = sorted_inter_scores
                target_cf_t = 0.0
            else:
                pool_indices = sorted_intra_indices
                pool_scores_local = sorted_intra_scores
                target_cf_t = 1.0

            if pool_indices.numel() == 0:
                caliper_fail += mask.sum().item()
                continue

            left_bound = torch.searchsorted(pool_scores_local, sub_scores - caliper_score)
            right_bound = torch.searchsorted(pool_scores_local, sub_scores + caliper_score)

            for i, (l, r) in enumerate(zip(left_bound.tolist(), right_bound.tolist())):
                if l >= r:
                    caliper_fail += 1
                    continue

                candidates = pool_indices[l:r]
                if candidates.numel() == 0:
                    caliper_fail += 1
                    continue

                u_q = sub_u[i]
                v_q = sub_v[i]
                emb_u = node_feats[u_q].unsqueeze(0)
                emb_v = node_feats[v_q].unsqueeze(0)

                u_cands = pool_row[candidates]
                v_cands = pool_col[candidates]
                emb_u_cands = node_feats[u_cands]
                emb_v_cands = node_feats[v_cands]

                sim_feat_u = F.cosine_similarity(emb_u, emb_u_cands)
                sim_feat_v = F.cosine_similarity(emb_v, emb_v_cands)
                sim_feat_direct = sim_feat_u + sim_feat_v
                sim_feat_cross = F.cosine_similarity(emb_u, emb_v_cands) + F.cosine_similarity(emb_v, emb_u_cands)

                if Z_feats is not None:
                    z_u = Z_feats[u_q].unsqueeze(0)
                    z_v = Z_feats[v_q].unsqueeze(0)
                    z_u_cands = Z_feats[u_cands]
                    z_v_cands = Z_feats[v_cands]

                    sim_z_u = F.cosine_similarity(z_u, z_u_cands)
                    sim_z_v = F.cosine_similarity(z_v, z_v_cands)
                    sim_z_direct = sim_z_u + sim_z_v
                    sim_z_cross = F.cosine_similarity(z_u, z_v_cands) + F.cosine_similarity(z_v, z_u_cands)

                    sim_direct = (sim_feat_direct + sim_z_direct) / 2.0
                    sim_cross = (sim_feat_cross + sim_z_cross) / 2.0
                    sim_direct =  sim_z_direct
                    sim_cross = sim_z_cross
                else:
                    sim_direct = sim_feat_direct
                    sim_cross = sim_feat_cross

                dist_direct = 1.0 - (sim_direct / 2.0)
                dist_cross = 1.0 - (sim_cross / 2.0)

                struct_dists = torch.min(dist_direct, dist_cross)
                min_dist, min_idx = torch.min(struct_dists, dim=0)
                struct_min_samples.append(min_dist.item())

                if min_dist <= struct_threshold_current:
                    best_candidate_idx = candidates[min_idx]

                    q_idx = sub_indices[i]
                    final_cf_s[q_idx] = pool_signs[best_candidate_idx]
                    final_cf_t[q_idx] = target_cf_t

                    matched_count += 1
                else:
                    struct_fail += 1

        if end % (batch_size * 4) == 0 or end == total_queries:
            logger.info(f"Processed {end}/{total_queries} edges, match rate {matched_count/end:.2%}")

    match_rate = matched_count / max(total_queries, 1)
    logger.info(
        f"Matching complete: Queries={total_queries}, Matched={matched_count}, CaliperFail={caliper_fail}, StructFail={struct_fail}, "
        f"MatchRate={match_rate:.2%}, StructThrFinal={struct_threshold_current:.3f}"
    )

    if struct_min_samples:
        sample_tensor = torch.tensor(struct_min_samples)
        mean_val = sample_tensor.mean().item()
        stats_msg = f"Struct distance samples (Cosine): n={sample_tensor.numel()}, mean={mean_val:.4f}"
        if quantile_tensor is not None:
            quantile_tensor_cpu = quantile_tensor.clamp(0.0, 1.0)
            quantiles = torch.quantile(sample_tensor, quantile_tensor_cpu)
            quantile_pairs = [f"q{int(q*100):02d}={val:.4f}" for q, val in zip(quantile_tensor_cpu.tolist(), quantiles.tolist())]
            stats_msg += ", " + ", ".join(quantile_pairs)
        logger.info(stats_msg)

    cf_u_np = final_cf_u.cpu().numpy()
    cf_v_np = final_cf_v.cpu().numpy()
    cf_s_np = final_cf_s.cpu().numpy()
    cf_t_np = final_cf_t.cpu().numpy()

    twins_data = np.stack([cf_u_np, cf_v_np, cf_s_np, cf_t_np], axis=1)

    sort_indices = np.lexsort((twins_data[:, 1], twins_data[:, 0]))
    twins_data = twins_data[sort_indices]

    unique_signs = np.unique(cf_s_np)
    logger.info(f"Structural twins generated: shape={twins_data.shape}, signs={unique_signs}, sample={twins_data[0]}")

    return twins_data
