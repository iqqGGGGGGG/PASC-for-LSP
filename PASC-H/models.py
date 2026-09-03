import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax
class PriorScoreNet(nn.Module):
    def __init__(self, dim_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_z * 2, dim_z),
            nn.BatchNorm1d(dim_z),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_z, 1)
        )
    def forward(self, z_u, z_v):
        x = torch.cat([z_u, z_v], dim=1)
        return self.net(x)

class BaseRepEncoder(nn.Module):
    def __init__(self, num_nodes, dim_feat, dim_h, dim_z, dropout):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.dim_feat = int(dim_feat)
        self.scale = float(0.3)
        self.delta = nn.Embedding(self.num_nodes, self.dim_feat)
        nn.init.zeros_(self.delta.weight)

        self.net = nn.Sequential(
            nn.Linear(dim_feat, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim_h, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_h, dim_z),
        )

    def forward(self, x):
        x_aug = x + self.scale * self.delta.weight
        z = self.net(x_aug)
        return z

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta.weight)
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

class RGALayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        self.input_proj = nn.Linear(dim, dim, bias=True)
        # Residual Bias Encoder for Residual-Guided Attention (phi_bias)
        self.res_bias_enc = nn.Sequential(
            nn.Linear(1, heads),
            nn.Tanh()
        )

        self.att_heads_pos = nn.ModuleList([
            nn.Sequential(nn.Linear(self.head_dim * 2, self.head_dim), nn.LeakyReLU(0.2), nn.Linear(self.head_dim, 1))
            for _ in range(heads)
        ])
        self.att_heads_neg = nn.ModuleList([
            nn.Sequential(nn.Linear(self.head_dim * 2, self.head_dim), nn.LeakyReLU(0.2), nn.Linear(self.head_dim, 1))
            for _ in range(heads)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def aggregate(self, z, edge_index, att_nets, cf_residuals):
        row, col = edge_index
        z_reshaped = z.view(-1, self.heads, self.head_dim)
        z_src = z_reshaped[row]
        z_dst = z_reshaped[col]

        if cf_residuals.dim() == 1: cf_residuals = cf_residuals.unsqueeze(1)
        phi_bias = self.res_bias_enc(cf_residuals)

        head_outs = []
        for h in range(self.heads):
            cat_feat = torch.cat([z_src[:, h], z_dst[:, h]], dim=-1)
            scores = att_nets[h](cat_feat).squeeze(-1)
            scores = scores + phi_bias[:, h]
            alpha = softmax(scores, row)
            msg = z_dst[:, h] * alpha.view(-1, 1)
            agg_h = scatter(msg, row, dim=0, dim_size=z.size(0), reduce='sum')
            head_outs.append(agg_h)

        return torch.cat(head_outs, dim=-1)

    def forward(self, z, edge_index_pos, edge_index_neg, R_pos, R_neg):
        z_proj = self.input_proj(z)
        pos_agg = self.aggregate(z_proj, edge_index_pos, self.att_heads_pos, R_pos)
        if edge_index_neg is not None and edge_index_neg.size(1) > 0:
            neg_agg = self.aggregate(z_proj, edge_index_neg, self.att_heads_neg, R_neg)
        else:
            neg_agg = torch.zeros_like(pos_agg)
        combined = torch.cat([pos_agg, neg_agg], dim=-1)
        return self.output_proj(combined)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        for m in self.res_bias_enc:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        for net in self.att_heads_pos:
            for m in net:
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        for net in self.att_heads_neg:
            for m in net:
                if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
        for m in self.output_proj:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

class RGA(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([RGALayer(dim, heads=4) for _ in range(num_layers)])
    def forward(self, z, edge_index_pos, edge_index_neg, R_pos, R_neg):
        x = z
        for layer in self.layers:
            x = layer(x, edge_index_pos, edge_index_neg, R_pos, R_neg)
        return x

    def reset_parameters(self):
        for l in self.layers: l.reset_parameters()

class CondDecoder(nn.Module):
    def __init__(self, dim_z, dim_h=64):
        super().__init__()

        input_dim = dim_z * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim_h),
            nn.LayerNorm(dim_h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim_h, dim_h // 2),
            nn.ReLU(),
            nn.Linear(dim_h // 2, 1)
        )

    def forward(self, z_u, z_v, T):
        if T.dim() == 1:
            T = T.unsqueeze(1)
        x = torch.cat([z_u, z_v, T], dim=1)
        return self.net(x)

    def reset_parameters(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

class PASC_H(nn.Module):
    def __init__(
        self, dim_feat, dim_h, dim_z, dropout, num_nodes, Z_batch=None,
    ):
        super().__init__()
        self.base_encoder = BaseRepEncoder(num_nodes, dim_feat, dim_h, dim_z, dropout)
        self.ctx_agg = RGA(dim=dim_z, num_layers=2)
        self.decoder = CondDecoder(dim_z, dim_h=dim_h)
        self.gate_net = nn.Sequential(
            nn.Linear(dim_z * 4, dim_z),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_z, 1)
        )

    def reset_parameters(self):
        self.base_encoder.reset_parameters()
        self.ctx_agg.reset_parameters()
        self.decoder.reset_parameters()
        for m in self.gate_net:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, X_prior, edges, T_factual, T_switched, edge_index_pos, edge_index_neg, R_pos, R_neg):
        z_base = self.base_encoder(X_prior)
        z_context = self.ctx_agg(z_base, edge_index_pos, edge_index_neg, R_pos, R_neg)
        u, v = edges.T[0], edges.T[1]
        z_base_u, z_base_v = z_base[u], z_base[v]
        z_ctx_u, z_ctx_v = z_context[u], z_context[v]
        logits_original = self.decoder(z_base_u, z_base_v, T_factual)
        logits_switched = self.decoder(z_base_u, z_base_v, T_switched)
        logits_gate = self.gate_net(torch.cat([z_base_u, z_base_v, z_ctx_u, z_ctx_v], dim=1))
        w = torch.sigmoid(logits_gate)
        z_fused_u = z_base_u + w * z_ctx_u
        z_fused_v = z_base_v + w * z_ctx_v
        logits_fused = self.decoder(z_fused_u, z_fused_v, T_factual)
        return logits_fused, logits_original, logits_switched, logits_gate
