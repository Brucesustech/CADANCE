import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter


def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0
    )
    return kl.sum(dim=-1).mean()


class HeteroTemporalEncoding(nn.Module):
    def __init__(self, n_hid, temp_dim, max_len=128):
        super(HeteroTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, temp_dim, 2) * -(math.log(10000.0) / temp_dim))

        emb = nn.Embedding(max_len, temp_dim)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(temp_dim)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(temp_dim)
        emb.weight.requires_grad = False
        self.emb = emb

        self.mlp = nn.Sequential(
            nn.Linear(n_hid + temp_dim, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid)
        )

    def forward(self, x, t):
        t_emb = self.emb(t)
        t_emb = t_emb.expand(x.size(0), -1)
        return x + self.mlp(torch.cat([x, t_emb], dim=-1))


class CausalStructurePropagation(nn.Module):
    def __init__(self, hid_dim):
        super(CausalStructurePropagation, self).__init__()
        self.gate_z = nn.Linear(2 * hid_dim, hid_dim)
        self.gate_r = nn.Linear(2 * hid_dim, hid_dim)
        self.gate_h = nn.Linear(2 * hid_dim, hid_dim)
        self.attn_pool = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )
        self.belief_to_node = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x, prev_belief):
        attn_weights = F.softmax(self.attn_pool(x), dim=0)
        graph_summary = (attn_weights * x).sum(dim=0, keepdim=True)

        combined = torch.cat([graph_summary, prev_belief], dim=-1)
        z = torch.sigmoid(self.gate_z(combined))
        r = torch.sigmoid(self.gate_r(combined))
        h_tilde = torch.tanh(self.gate_h(torch.cat([graph_summary, r * prev_belief], dim=-1)))
        new_belief = (1 - z) * prev_belief + z * h_tilde

        node_bias = self.belief_to_node(new_belief).expand(x.size(0), -1)
        return node_bias, new_belief


class SoftCausalSubgraphNet(nn.Module):
    """
    Soft causal/confounding decomposition.
    """
    def __init__(self, hid_dim, dropout=0.1):
        super(SoftCausalSubgraphNet, self).__init__()
        self.conv1 = GCNConv(hid_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hid_dim * 4, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, 1)
        )
        self.deg_embed = nn.Embedding(256, hid_dim)

    def forward(self, x, edge_index, t, time_encoding):
        if edge_index.numel() == 0:
            zero = torch.tensor(0.0, device=x.device)
            return edge_index, torch.zeros(0, device=x.device), torch.zeros(0, device=x.device), zero

        num_nodes = x.size(0)
        deg = degree(edge_index[0], num_nodes=num_nodes).long().clamp(max=255)
        x = x + self.deg_embed(deg)

        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        h = time_encoding(h, t)

        row, col = edge_index
        edge_rep = torch.cat([
            h[row], h[col], h[row] - h[col], h[row] * h[col]
        ], dim=-1)

        score = torch.sigmoid(self.edge_mlp(edge_rep)).view(-1)
        score = score.clamp(1e-4, 1 - 1e-4)

        causal_w = score
        conf_w = 1.0 - score

        mask_loss = (score * (1.0 - score)).mean()
        return edge_index, causal_w, conf_w, mask_loss


class StructuralAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, dropout,
                 residual=True, use_fmask=True, norm=True, skip=True):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.in_dim, self.hid_dim = input_dim, output_dim
        self.update_norm = nn.LayerNorm(output_dim)
        self.cs_mlp = nn.Sequential(
            nn.Linear(output_dim, 2 * output_dim),
            nn.GELU(),
            nn.Linear(2 * output_dim, output_dim)
        )
        self.fmask = nn.Parameter(torch.ones(output_dim))
        self.residual = residual
        if residual and input_dim != output_dim:
            self.lin_residual = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.lin_residual = None

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.d_k = output_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.dropout = nn.Dropout(dropout)
        self.use_fmask = use_fmask
        self.norm = norm
        self.skip = skip
        self.update_skip = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index, edge_weight=None):
        if edge_index.numel() == 0:
            return x

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)

        q_mat = self.q_linear(x[edge_index[1]]).view(-1, self.n_heads, self.d_k)
        k_mat = self.k_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k)
        v_mat = self.v_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k)

        att = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        att = edge_weight.view(-1, 1) * att
        att = softmax(att, edge_index[1])

        msg = v_mat * att.view(-1, self.n_heads, 1)
        msg = msg.view(-1, self.hid_dim)
        out = scatter(msg, edge_index[1], dim=0, dim_size=x.shape[0], reduce='add')

        if self.use_fmask:
            out = out * F.softmax(self.fmask, dim=0)

        if self.residual:
            if self.lin_residual is not None:
                out = out + self.lin_residual(x)
            else:
                out = out + x

        if self.norm:
            upd = self.cs_mlp(self.update_norm(out))
        else:
            upd = self.cs_mlp(out)

        upd = self.dropout(upd)
        if self.skip:
            alpha = torch.sigmoid(self.update_skip)
            out = (1 - alpha) * out + alpha * upd
        else:
            out = out + upd
        return out


class GCNExpert(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(hid_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_index.numel() == 0:
            return x
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return self.norm(h + x)


class MixHopExpert(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(hid_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.proj = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_index.numel() == 0:
            return x
        h1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight=edge_weight))
        h = self.proj(torch.cat([h1, h2], dim=-1))
        return self.norm(h + x)


class AttnExpert(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        heads = 4 if hid_dim % 4 == 0 else 1
        self.layer = StructuralAttentionLayer(
            input_dim=hid_dim,
            output_dim=hid_dim,
            n_heads=heads,
            dropout=dropout,
            residual=True,
            use_fmask=True,
            norm=True,
            skip=True
        )

    def forward(self, x, edge_index, edge_weight=None):
        return self.layer(x, edge_index, edge_weight=edge_weight)


class MLPExpert(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.norm(self.mlp(x) + x)


class DistributionDetector(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2 * hid_dim, hid_dim)
        self.conv = GCNConv(hid_dim, hid_dim)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, memory, edge_index, t, time_encoding):
        h = self.input_proj(torch.cat([x, memory], dim=-1))
        h = time_encoding(h, t)
        if edge_index.numel() > 0:
            h = self.conv(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h)


class PrototypeRouter(nn.Module):
    """
    Fixed number of experts = 4.
    Temperature is fixed internally, not exposed as hyperparameter.
    """
    def __init__(self, hid_dim, temperature=0.7):
        super().__init__()
        self.temperature = temperature
        self.num_experts = 4
        self.prototypes = nn.Parameter(torch.randn(self.num_experts, hid_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, routing_embed):
        r = F.normalize(routing_embed, p=2, dim=-1)
        p = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.matmul(r, p.t()) / self.temperature
        weights = F.softmax(logits, dim=-1)

        sim = torch.matmul(p, p.t())
        eye = torch.eye(self.num_experts, device=sim.device)
        proto_loss = ((sim - eye) ** 2 * (1 - eye)).sum() / (self.num_experts * (self.num_experts - 1) + 1e-8)

        balance_loss = ((weights.mean(dim=0) - 1.0 / self.num_experts) ** 2).mean()
        return weights, proto_loss + 0.1 * balance_loss


class VariationalEnvironmentGenerator(nn.Module):
    def __init__(self, hid_dim, dropout=0.1):
        super().__init__()
        self.posterior = nn.Sequential(
            nn.Linear(2 * hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hid_dim, 2 * hid_dim)
        )
        self.prior = nn.Sequential(
            nn.Linear(hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hid_dim, 2 * hid_dim)
        )
        self.env_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, variant_base, memory):
        q_stats = self.posterior(torch.cat([variant_base, memory], dim=-1))
        mu_q, logvar_q = q_stats.chunk(2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -8.0, 8.0)

        p_stats = self.prior(memory)
        mu_p, logvar_p = p_stats.chunk(2, dim=-1)
        logvar_p = torch.clamp(logvar_p, -8.0, 8.0)

        if self.training:
            eps = torch.randn_like(mu_q)
            env = mu_q + eps * torch.exp(0.5 * logvar_q)
        else:
            env = mu_q

        enhanced_variant = self.norm(variant_base + self.env_proj(env))
        kl = kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p)
        return enhanced_variant, kl


class MemoryUpdater(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.gate = nn.Linear(3 * hid_dim, hid_dim)
        self.candidate = nn.Linear(3 * hid_dim, hid_dim)

    def forward(self, causal_z, variant_z, memory):
        cat = torch.cat([causal_z, variant_z, memory], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        cand = torch.tanh(self.candidate(cat))
        return g * cand + (1 - g) * memory


class MultiplyPredictor(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(MultiplyPredictor, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, z, edge_index):
        z = F.normalize(z, p=2, dim=1)
        pred = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(pred / (self.temperature.abs() + 1e-6))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(LinkPredictor, self).__init__()
        self.lins = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, z, e):
        x = torch.cat([z[e[0]], z[e[1]]], dim=1)
        x = self.lins(x)
        return torch.sigmoid(x).squeeze(-1)


class NodeClf(nn.Module):
    def __init__(self, layers, num_class, hid_dim):
        super().__init__()
        self.clf = nn.Linear(hid_dim, num_class)

    def forward(self, x):
        return self.clf(x)


class MutiNodeClf(nn.Module):
    def __init__(self, layers, num_class, hid_dim):
        super().__init__()
        modules = []
        for _ in range(max(1, layers)):
            modules.append(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU()))
        modules.append(nn.Linear(hid_dim, num_class))
        self.clf = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x


class CADENCE(nn.Module):
    def __init__(self, args):
        super(CADENCE, self).__init__()
        self.args = args
        self.device = args.device

        self.feat = Parameter(
            torch.ones(args.num_nodes, args.input_dim, device=args.device),
            requires_grad=True
        )

        self.input_proj = nn.Linear(args.input_dim, args.hid_dim, bias=bool(args.lin_bias))
        self.time_emb = HeteroTemporalEncoding(
            n_hid=args.hid_dim,
            temp_dim=args.hid_dim,
            max_len=args.max_time_len
        )

        self.cspn = CausalStructurePropagation(args.hid_dim)

        self.causal_subgraph_net = SoftCausalSubgraphNet(
            hid_dim=args.hid_dim,
            dropout=args.dropout
        )

        self.distribution_detector = DistributionDetector(
            hid_dim=args.hid_dim,
            dropout=args.dropout
        )

        self.router = PrototypeRouter(hid_dim=args.hid_dim)

        self.experts = nn.ModuleList([
            GCNExpert(args.hid_dim, args.dropout),
            AttnExpert(args.hid_dim, args.dropout),
            MixHopExpert(args.hid_dim, args.dropout),
            MLPExpert(args.hid_dim, args.dropout)
        ])

        self.variant_encoder = GCNExpert(args.hid_dim, args.dropout)
        self.env_generator = VariationalEnvironmentGenerator(args.hid_dim, args.dropout)
        self.memory_updater = MemoryUpdater(args.hid_dim)

        if args.experiment_name == 'lp':
            self.cs_decoder = MultiplyPredictor()
            self.ss_decoder = LinkPredictor(2 * args.hid_dim, args.hid_dim, 1, args.dropout)
        else:
            self.cs_decoder = NodeClf(args.nc_layers, args.num_classes, args.hid_dim)
            self.ss_decoder = MutiNodeClf(args.nc_layers, args.num_classes, args.hid_dim)

    def forward(self, edge_index_list, x_list=None, seq_len=None):
        if seq_len is None:
            seq_len = len(edge_index_list)

        if x_list is None:
            x_seq = [self.input_proj(self.feat) for _ in range(seq_len)]
        else:
            x_seq = [self.input_proj(x_list[t].to(self.device)) for t in range(seq_len)]

        num_nodes = x_seq[0].size(0)
        memory = torch.zeros(num_nodes, self.args.hid_dim, device=self.device)
        causal_belief = torch.zeros(1, self.args.hid_dim, device=self.device)

        causal_outputs, variant_outputs, route_outputs = [], [], []
        proto_losses, kl_losses, mask_losses, orth_losses = [], [], [], []

        for t in range(seq_len):
            x_t = x_seq[t]
            t_tensor = torch.LongTensor([t]).to(self.device)
            x_t = self.time_emb(x_t, t_tensor)

            causal_bias, causal_belief = self.cspn(x_t, causal_belief)
            x_struct = x_t + causal_bias

            edge_index = edge_index_list[t].to(self.device)
            _, causal_w, conf_w, mask_loss = self.causal_subgraph_net(
                x_struct, edge_index, t_tensor, self.time_emb
            )

            routing_embed = self.distribution_detector(
                x_t, memory, edge_index, t_tensor, self.time_emb
            )
            route_probs, proto_loss = self.router(routing_embed)

            expert_outs = []
            for expert in self.experts:
                expert_outs.append(expert(x_struct, edge_index, causal_w))
            expert_outs = torch.stack(expert_outs, dim=1)  # [N,4,H]
            causal_z = (route_probs.unsqueeze(-1) * expert_outs).sum(dim=1)

            variant_base = self.variant_encoder(x_struct, edge_index, conf_w)
            variant_z, kl = self.env_generator(variant_base, memory)

            orth = (
                (F.normalize(causal_z, p=2, dim=-1) *
                 F.normalize(variant_z, p=2, dim=-1)).sum(dim=-1).pow(2).mean()
            )

            memory = self.memory_updater(causal_z, variant_z, memory)

            causal_outputs.append(causal_z)
            variant_outputs.append(variant_z)
            route_outputs.append(route_probs)
            proto_losses.append(proto_loss)
            kl_losses.append(kl)
            mask_losses.append(mask_loss)
            orth_losses.append(orth)

        aux = {
            'causal_embeddings': torch.stack(causal_outputs, dim=1),
            'variant_embeddings': torch.stack(variant_outputs, dim=1),
            'route_probs': torch.stack(route_outputs, dim=1),
            'loss_proto': torch.stack(proto_losses).mean(),
            'loss_kl': torch.stack(kl_losses).mean(),
            'loss_mask': torch.stack(mask_losses).mean(),
            'loss_orth': torch.stack(orth_losses).mean(),
        }
        return aux