import os
import sys
import time
import argparse
import yaml
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, balanced_accuracy_score,
    average_precision_score
)
from utils import get_logger, get_node_features
from models import PASC_S
from struct_utils import load_comm_labels, compute_struct_grad, pretrain_psn
from load_data import load_data

def get_args():
    parser = argparse.ArgumentParser(description='PASC-S: Soft Residual-Aware Reweighting')
    parser.add_argument('--config', type=str, default='_config-OTC.yaml', help='Path to configuration file')
    args_temp, _ = parser.parse_known_args()

    default_config = {
        'dataset': 'Bitcoin-OTC',
        'datapath': 'Data/',
        'log_dir': 'logs/',
        'seed': 1,
        'gpu': 3,
        'dim_h': 128,
        'dim_z': 128,
        'dropout': 0.2,
        'l2reg': 0.03,
        'epochs': 500,
        'batch_size': 32768,
        'patience': 40,
        'lr': 0.0001,
        'drop_edge_rate': 0.2,
        'name': 'PASC-S',
        'svd_rank': 128
    }

    if os.path.exists(args_temp.config):
        try:
            with open(args_temp.config, 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    default_config.update(config)
            print(f"Loaded configuration from {args_temp.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    parser.add_argument('--dataset', type=str, default=default_config['dataset'])
    parser.add_argument('--datapath', type=str, default=default_config['datapath'])
    parser.add_argument('--log_dir', type=str, default=default_config['log_dir'])
    parser.add_argument('--seed', type=int, default=default_config['seed'])
    parser.add_argument('--gpu', type=int, default=default_config['gpu'])
    parser.add_argument('--dim_h', type=int, default=default_config['dim_h'])
    parser.add_argument('--dim_z', type=int, default=default_config['dim_z'])
    parser.add_argument('--dropout', type=float, default=default_config['dropout'])
    parser.add_argument('--l2reg', type=float, default=default_config['l2reg'])
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--patience', type=int, default=default_config['patience'])
    parser.add_argument('--lr', type=float, default=default_config['lr'])
    parser.add_argument('--drop_edge_rate', type=float, default=default_config['drop_edge_rate'])
    parser.add_argument('--name', type=str, default=default_config['name'])
    parser.add_argument('--svd_rank', type=int, default=default_config['svd_rank'])
    parser.add_argument('--use_cf', action='store_true', default=True, help='Use Control Function residuals')
    parser.add_argument('--feature_type', type=str, default='laplacian', choices=['svd', 'laplacian'], help='Initial node features type: svd or laplacian')

    file_name = str(os.path.basename(__file__).split('.')[0])
    parser.add_argument('--file_name', type=str, default=file_name)

    args = parser.parse_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
    return args

def train(args, logger):
    device = args.device
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Data_file = os.path.join(args.datapath, 'data_file', f'{args.dataset}.txt')
    train_data_full, val_data, test_data, adj, _, _ = load_data(args.dataset, Data_file, args)
    nodes = int(max(train_data_full[:, 0].max(), train_data_full[:, 1].max()) + 1)

    logger.info(f'=== Dataset Stats ===')
    logger.info(f'Nodes: {nodes}, Train: {len(train_data_full)}')

    X_prior = get_node_features(args, adj, train_data_full, nodes, device, logger)
    X_prior = X_prior.to(device)

    Z_grad = compute_struct_grad(adj, X_prior, logger).to(device)

    comm_cache = os.path.join(
        args.datapath, 'cache', f'{args.file_name}_{args.dataset}_clusters.pkl'
    )
    comm_labels = load_comm_labels(
        args, comm_cache, adj, nodes, logger, train_data_full, X_prior.cpu().numpy()
    )

    prior_score_net = pretrain_psn(Z_grad, comm_labels, train_data_full, device, logger, epochs=400)

    def calc_pscore(u, v):
        z_u = Z_grad[u]
        z_v = Z_grad[v]
        return prior_score_net(z_u, z_v)

    edges_f_t1 = train_data_full[train_data_full[:, 2] == 1]
    edges_f_t0 = train_data_full[train_data_full[:, 2] == -1]
    edge_index_pos = torch.LongTensor(edges_f_t1[:, :2]).t().contiguous().to(device)
    edge_index_neg = torch.LongTensor(edges_f_t0[:, :2]).t().contiguous().to(device)

    train_edges_tensor = torch.from_numpy(train_data_full[:, :2].astype(np.int64)).to(device)
    train_labels = torch.from_numpy((train_data_full[:, 2] == 1).astype(np.float32)).to(device)

    val_edges_device = torch.from_numpy(val_data[:, :2].astype(np.int64)).to(device)
    val_labels = torch.from_numpy((val_data[:, 2] == 1).astype(np.float32))

    test_edges_device = torch.from_numpy(test_data[:, :2].astype(np.int64)).to(device)
    test_labels = torch.from_numpy((test_data[:, 2] == 1).astype(np.float32))

    model = PASC_S(X_prior.shape[1], args.dim_h, args.dim_z, args.dropout, num_nodes=nodes).to(device)
    logger.info(f"Model Initialized.")
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2reg}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    num_pos = float((train_labels == 1).sum())
    num_neg = float((train_labels == 0).sum())
    total = num_pos + num_neg

    logger.info(f"Loss Configuration: Mode = [BCE] with Soft Reweighting")
    logger.info(f"Stats: Pos={int(num_pos)} ({num_pos/total:.1%}), Neg={int(num_neg)}")
    logger.info(f"Start Training: Epochs={args.epochs} Batch={args.batch_size}")
    best_model_state = None
    cnt_wait = 0
    best_thresh = 0.5

    if not isinstance(comm_labels, torch.Tensor):
        comm_labels = torch.from_numpy(comm_labels).long().to(device)
    else:
        comm_labels = comm_labels.to(device)

    try:
        for epoch in range(args.epochs):
            model.train()
            if args.drop_edge_rate > 0:
                mask_pos = torch.rand(edge_index_pos.size(1), device=device) > args.drop_edge_rate
                ei_pos_aug = edge_index_pos[:, mask_pos]
                mask_neg = torch.rand(edge_index_neg.size(1), device=device) > args.drop_edge_rate
                ei_neg_aug = edge_index_neg[:, mask_neg]
            else:
                ei_pos_aug, ei_neg_aug = edge_index_pos, edge_index_neg
            total_loss = 0

            for perm in DataLoader(range(len(train_edges_tensor)), args.batch_size, shuffle=True):
                batch_edges = train_edges_tensor[perm]
                batch_lbl = train_labels[perm]

                u = batch_edges[:, 0]
                v = batch_edges[:, 1]
                T_fact = (comm_labels[u] == comm_labels[v]).float().view(-1)

                with torch.no_grad():
                    z_u = Z_grad[u]
                    z_v = Z_grad[v]
                    e_logit = prior_score_net(z_u, z_v).view(-1)
                    e_score = torch.sigmoid(e_logit)
                    e_score = torch.clamp(e_score, 0.05, 0.95)

                w_reweight = (T_fact / e_score) + ((1 - T_fact) / (1 - e_score))
                w_reweight = w_reweight / w_reweight.mean()

                cf_residuals = T_fact.view(-1, 1) - e_score.view(-1, 1)

                with torch.no_grad():
                    u_p, v_p = ei_pos_aug[0], ei_pos_aug[1]
                    T_pos = (comm_labels[u_p] == comm_labels[v_p]).float().view(-1, 1)
                    e_pos = torch.sigmoid(calc_pscore(u_p, v_p)).view(-1, 1)
                    R_pos = T_pos - e_pos

                    if ei_neg_aug is not None and ei_neg_aug.size(1) > 0:
                        u_n, v_n = ei_neg_aug[0], ei_neg_aug[1]
                        T_neg = (comm_labels[u_n] == comm_labels[v_n]).float().view(-1, 1)
                        e_neg = torch.sigmoid(calc_pscore(u_n, v_n)).view(-1, 1)
                        R_neg = T_neg - e_neg

                optimizer.zero_grad()
                logits_fused, logits_gate = model(
                    X_prior, batch_edges,
                    ei_pos_aug, ei_neg_aug,
                    T=T_fact,
                    R_pos=R_pos,
                    R_neg=R_neg
                )

                loss_fused = F.binary_cross_entropy_with_logits(logits_fused.squeeze(), batch_lbl)

                w_gate = torch.sigmoid(logits_gate)
                loss_gate = w_gate.mean()
                loss = loss_fused

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(perm)

            scheduler.step()
            avg_loss = total_loss / len(train_edges_tensor)

            model.eval()
            with torch.no_grad():
                u_val = val_edges_device[:, 0]
                v_val = val_edges_device[:, 1]
                t_val = (comm_labels[u_val] == comm_labels[v_val]).float().view(-1)

                u_p, v_p = edge_index_pos[0], edge_index_pos[1]
                T_pos = (comm_labels[u_p] == comm_labels[v_p]).float().view(-1, 1)
                e_pos = torch.sigmoid(calc_pscore(u_p, v_p)).view(-1, 1)
                R_pos_val = T_pos - e_pos

                if edge_index_neg is not None and edge_index_neg.size(1) > 0:
                    u_n, v_n = edge_index_neg[0], edge_index_neg[1]
                    T_neg = (comm_labels[u_n] == comm_labels[v_n]).float().view(-1, 1)
                    e_neg = torch.sigmoid(calc_pscore(u_n, v_n)).view(-1, 1)
                    R_neg_val = T_neg - e_neg

                logits_val, _ = model(
                    X_prior, val_edges_device,
                    edge_index_pos, edge_index_neg,
                    T=t_val,
                    R_pos=R_pos_val,
                    R_neg=R_neg_val
                )


                val_probs = torch.sigmoid(logits_val.squeeze()).cpu().numpy()
                best_f1_in_epoch = 0
                current_best_thresh = 0.5
                thresholds = np.arange(0.00, 1, 0.02)
                for th in thresholds:
                    temp_preds = (val_probs > th).astype(int)
                    temp_f1 = f1_score(val_labels.cpu().numpy(), temp_preds, average='macro')
                    if temp_f1 > best_f1_in_epoch:
                        best_f1_in_epoch = temp_f1
                        current_best_thresh = th

                val_preds = (val_probs > current_best_thresh).astype(int)
                val_auc = roc_auc_score(val_labels, val_probs)
                val_f1_bin = f1_score(val_labels, val_preds, average='binary')
                val_f1_macro = f1_score(val_labels, val_preds, average='macro')
                val_f1_micro = f1_score(val_labels, val_preds, average='micro')

            val_f1_bin = float(val_f1_bin)
            val_f1_macro = float(val_f1_macro)
            val_f1_micro = float(val_f1_micro)
            val_auc = float(val_auc)
            avg_metric = (float(val_auc) ) / 1.0
            if 'best_avg_metric' not in locals():
                best_avg_metric = -1.0
            if avg_metric > best_avg_metric:
                best_avg_metric = avg_metric
                best_thresh = current_best_thresh
                best_model_state = copy.deepcopy(model.state_dict())
                cnt_wait = 0
                status = f"✓ (AvgMetric={avg_metric:.4f})"
            else:
                cnt_wait += 1
                status = f"Wait {cnt_wait} (AvgMetric={avg_metric:.4f})"
            logger.info(f"Ep {epoch+1}: Loss {avg_loss:.4f} w_gate:{loss_gate.item():.4f} | Val AUC {val_auc:.4f} F1(b/m/u) {val_f1_bin:.4f}/{val_f1_macro:.4f}/{val_f1_micro:.4f} | Thresh {current_best_thresh:.4f} | {status}")

            if cnt_wait >= args.patience:
                logger.info("Early Stopping Triggered.")
                break

    except Exception as e:
        logger.error(f"Training Loop Failed: {e}", exc_info=True)
        raise e

    if best_model_state: model.load_state_dict(best_model_state)
    inf_pos = train_data_full[train_data_full[:, 2] == 1]
    inf_neg = train_data_full[train_data_full[:, 2] == -1]
    full_ei_pos = torch.LongTensor(inf_pos[:, :2]).t().contiguous().to(device)
    full_ei_neg = torch.LongTensor(inf_neg[:, :2]).t().contiguous().to(device)

    model.eval()
    with torch.no_grad():
        t_test = None
        R_pos_test, R_neg_test = None, None

        if args.use_cf:
            u_test = test_edges_device[:, 0]
            v_test = test_edges_device[:, 1]
            t_test = (comm_labels[u_test] == comm_labels[v_test]).float().view(-1)

            u_p, v_p = full_ei_pos[0], full_ei_pos[1]
            T_pos = (comm_labels[u_p] == comm_labels[v_p]).float().view(-1, 1)
            e_pos = torch.sigmoid(calc_pscore(u_p, v_p)).view(-1, 1)
            R_pos_test = T_pos - e_pos

            if full_ei_neg is not None and full_ei_neg.size(1) > 0:
                u_n, v_n = full_ei_neg[0], full_ei_neg[1]
                T_neg = (comm_labels[u_n] == comm_labels[v_n]).float().view(-1, 1)
                e_neg = torch.sigmoid(calc_pscore(u_n, v_n)).view(-1, 1)
                R_neg_test = T_neg - e_neg

        logits_test, logits_gate = model(
            X_prior, test_edges_device,
             full_ei_pos, full_ei_neg,
             T=t_test,
             R_pos=R_pos_test,
             R_neg=R_neg_test
        )
        w_mean = torch.sigmoid(logits_gate).mean()
        test_probs = torch.sigmoid(logits_test.squeeze()).cpu().numpy()
        test_pred = (test_probs > best_thresh).astype(int)


        test_auc = roc_auc_score(test_labels, test_probs)
        test_ap = average_precision_score(test_labels, test_probs)
        test_f1 = f1_score(test_labels, test_pred)
        test_f1_macro = f1_score(test_labels, test_pred, average='macro')
        test_acc = accuracy_score(test_labels, test_pred)
        test_bal_acc = balanced_accuracy_score(test_labels, test_pred)
        test_precision = precision_score(test_labels, test_pred)
        test_recall = recall_score(test_labels, test_pred)

        test_pos_ratio = test_labels.sum() / len(test_labels)

        logger.info("="*60)
        logger.info(f"FINAL TEST RESULTS ({args.dataset}) using Best Threshold: {best_thresh:.4f}")
        logger.info(f"AUC        : {test_auc:.4f}")
        logger.info(f"AP         : {test_ap:.4f}  (Baseline: {test_pos_ratio:.4f})")
        logger.info(f"F1 (Bin)   : {test_f1:.4f}")
        logger.info(f"F1 (Macro) : {test_f1_macro:.4f}")
        logger.info(f"Accuracy   : {test_acc:.4f}")
        logger.info(f"Bal. Acc   : {test_bal_acc:.4f}")
        logger.info(f"Precision  : {test_precision:.4f}")
        logger.info(f"Recall     : {test_recall:.4f}")
        logger.info(f"Avg w_gate : {w_mean.item():.4f}")
        logger.info("-" * 60)
        logger.info(f"Test: AUC:{test_auc:.4f}|F1(bin):{test_f1:.4f}|F1(macro):{test_f1_macro:.4f}|Acc:{test_acc:.4f}|Bal_Acc:{test_bal_acc:.4f}|Precision:{test_precision:.4f}|Recall:{test_recall:.4f}")
        logger.info("="*60)

    return {
        'auc': float(test_auc),
        'f1_bin': float(test_f1),
        'f1_micro': float(test_acc),
        'f1_macro': float(test_f1_macro),
    }


def main(args):
    num_runs = 5
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    summary_log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{time.strftime("%m-%d_%H-%M")}_summary'
    summary_logger = get_logger(summary_log_name)
    summary_logger.info(f"Base Args: {args}")

    auc_list, f1_bin_list, f1_micro_list, f1_macro_list, time_list = [], [], [], [], []

    try:
        for run_idx in range(num_runs):
            run_args = copy.deepcopy(args)
            run_args.seed = int(args.seed) + run_idx

            run_log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{time.strftime("%m-%d_%H-%M")}_run{run_idx+1}'
            run_logger = get_logger(run_log_name)
            run_logger.info(f"Run {run_idx+1}/{num_runs} Args: {run_args}")

            start_time = time.time()
            metrics = train(run_args, run_logger)
            elapsed = time.time() - start_time
            run_logger.info(f"Total elapsed time: {elapsed:.2f}s")

            auc_list.append(metrics['auc'])
            f1_bin_list.append(metrics['f1_bin'])
            f1_micro_list.append(metrics['f1_micro'])
            f1_macro_list.append(metrics['f1_macro'])
            time_list.append(elapsed)

            summary_logger.info(
                f"Run {run_idx+1}: AUC={metrics['auc']:.4f}, "
                f"F1(bin)={metrics['f1_bin']:.4f}, "
                f"F1(micro)={metrics['f1_micro']:.4f}, "
                f"F1(macro)={metrics['f1_macro']:.4f}, Time={elapsed:.2f}s"
            )

        ddof = 1 if num_runs > 1 else 0
        auc_mean, auc_std = float(np.mean(auc_list)), float(np.std(auc_list, ddof=ddof))
        f1_bin_mean, f1_bin_std = float(np.mean(f1_bin_list)), float(np.std(f1_bin_list, ddof=ddof))
        f1_micro_mean, f1_micro_std = float(np.mean(f1_micro_list)), float(np.std(f1_micro_list, ddof=ddof))
        f1_macro_mean, f1_macro_std = float(np.mean(f1_macro_list)), float(np.std(f1_macro_list, ddof=ddof))
        time_mean, time_std = float(np.mean(time_list)), float(np.std(time_list, ddof=ddof))

        summary_logger.info("=" * 60)
        summary_logger.info(f"5-run summary ({args.dataset})")
        summary_logger.info(f"AUC        : {auc_mean:.4f} ± {auc_std:.4f}")
        summary_logger.info(f"F1 (Bin)   : {f1_bin_mean:.4f} ± {f1_bin_std:.4f}")
        summary_logger.info(f"F1 (Micro) : {f1_micro_mean:.4f} ± {f1_micro_std:.4f}")
        summary_logger.info(f"F1 (Macro) : {f1_macro_mean:.4f} ± {f1_macro_std:.4f}")
        summary_logger.info(f"Time (s)   : {time_mean:.2f} ± {time_std:.2f}")
        summary_logger.info(f"Time range : min={min(time_list):.2f}, max={max(time_list):.2f}")
        summary_logger.info("=" * 60)

    except Exception as e:
        summary_logger.error(f"Critical Failure in Main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    args = get_args()
    main(args)
