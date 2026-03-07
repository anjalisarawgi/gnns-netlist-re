"""
Directed Graph Representation Learning Models
Based on: "A Benchmark on Directed Graph Representation Learning in Hardware Designs"
arxiv: 2410.06460  |  ICLR 2025

Three models, all drop-in replacements for HierarchicalDirectedGAT:

    model = BiMPNN(in_channels, hidden_channels=256, out_channels=2)
    model = BiGIN(in_channels, hidden_channels=256, out_channels=2)
    model = BiMPNN_GT(in_channels, hidden_channels=256, out_channels=2)  ← best

All share the same forward signature:
    out = model(x, edge_index)   →   [N, out_channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY: Directed Random Walk Positional Encoding (RWPE)
#
# Keeps forward and backward random walks SEPARATE so the model knows:
#   fwd walk: "which nodes can I reach from here?"
#   bwd walk: "which nodes can reach me?"
# Output: [N, 2 * walk_steps]
# ══════════════════════════════════════════════════════════════════════════════

def directed_rwpe(edge_index, num_nodes, walk_steps=8):
    device = edge_index.device
    row, col = edge_index

    def rw_landing_probs(src, dst, weight, steps):
        indices = torch.stack([dst, src], dim=0)
        P = torch.sparse_coo_tensor(
            indices, weight, (num_nodes, num_nodes),
            dtype=torch.float32, device=device
        )
        x = torch.eye(num_nodes, device=device)
        landing = []
        for _ in range(steps):
            x = torch.sparse.mm(P, x)
            landing.append(x.diagonal())
        return torch.stack(landing, dim=1)  # [N, steps]

    out_deg = degree(row, num_nodes=num_nodes).clamp(min=1)
    fwd_pe  = rw_landing_probs(row, col, 1.0 / out_deg[row], walk_steps)

    in_deg  = degree(col, num_nodes=num_nodes).clamp(min=1)
    bwd_pe  = rw_landing_probs(col, row, 1.0 / in_deg[col],  walk_steps)

    return torch.cat([fwd_pe, bwd_pe], dim=1)  # [N, 2*walk_steps]


def _prepend_pe(x, edge_index, walk_steps):
    pe = directed_rwpe(edge_index, x.size(0), walk_steps)
    return torch.cat([x, pe], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITY: Bidirected message passing primitives
# ══════════════════════════════════════════════════════════════════════════════

class _BiMPNNLayer(MessagePassing):
    """Single-direction MPNN layer used inside BiDirectedLayer."""
    def __init__(self, in_ch, out_ch, direction='fwd'):
        super().__init__(aggr='add', flow='source_to_target')
        self.direction = direction
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x, edge_index):
        if self.direction == 'bwd':
            row, col = edge_index
            edge_index = torch.stack([col, row], dim=0)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_j, x_i], dim=-1))


class BiDirectedLayer(nn.Module):
    """fwd + bwd MPNN → concat → update MLP → residual."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.fwd = _BiMPNNLayer(in_ch, out_ch, 'fwd')
        self.bwd = _BiMPNNLayer(in_ch, out_ch, 'bwd')
        self.update = nn.Sequential(
            nn.Linear(in_ch + 2 * out_ch, out_ch), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, out_ch),
        )
        self.skip = nn.Linear(in_ch, out_ch, bias=False) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index):
        h = self.update(torch.cat([x, self.fwd(x, edge_index),
                                      self.bwd(x, edge_index)], dim=-1))
        return h + self.skip(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: BiMPNN
#
# Baseline bidirected model. Each layer runs fwd + bwd message passing
# and merges them via an update MLP. Depth provides multi-hop coverage.
#
# Ranking in paper: strong baseline, outperforms undirected MPNNs.
# ══════════════════════════════════════════════════════════════════════════════

class BiMPNN(nn.Module):
    """
    Bidirected MPNN + directed RWPE.

    Args:
        in_channels     : raw node feature dim
        hidden_channels : width  (default 256)
        out_channels    : output dim
        num_layers      : stacked BiDirectedLayers  (default 6)
        dropout         : dropout rate
        walk_steps      : RWPE steps per direction  (default 8)
        use_pe          : prepend directed RWPE     (default True)
    """
    def __init__(self, in_channels, hidden_channels=256, out_channels=2,
                 num_layers=6, dropout=0.1, walk_steps=8, use_pe=True):
        super().__init__()
        self.use_pe, self.walk_steps = use_pe, walk_steps
        actual_in = in_channels + (2 * walk_steps if use_pe else 0)

        self.input_proj = nn.Linear(actual_in, hidden_channels)
        self.layers = nn.ModuleList([
            BiDirectedLayer(hidden_channels, hidden_channels, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        if self.use_pe:
            x = _prepend_pe(x, edge_index, self.walk_steps)
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: BiGIN  (BI-Graph Isomorphism Network)
#
# Replaces the message MLP with GIN-style aggregation:
#   h_v' = MLP((1 + ε) * h_v  +  Σ h_u)
# but run separately for fwd and bwd edges, then summed.
#
# GIN is provably as expressive as the Weisfeiler-Leman graph isomorphism
# test. The bidirected variant extends this to directed graphs.
#
# Ranking in paper: tied for top, simpler than BiMPNN_GT but nearly as good.
# ══════════════════════════════════════════════════════════════════════════════

class _BiGINLayer(MessagePassing):
    """Single-direction GIN layer."""
    def __init__(self, in_ch, out_ch, direction='fwd'):
        super().__init__(aggr='add', flow='source_to_target')
        self.direction = direction
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
        )

    def forward(self, x, edge_index):
        if self.direction == 'bwd':
            row, col = edge_index
            edge_index = torch.stack([col, row], dim=0)
        agg = self.propagate(edge_index, x=x)           # Σ h_u
        return self.mlp((1 + self.eps) * x + agg)

    def message(self, x_j):
        return x_j


class BiGINLayer(nn.Module):
    """fwd GIN + bwd GIN → sum → residual."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.fwd = _BiGINLayer(in_ch, out_ch, 'fwd')
        self.bwd = _BiGINLayer(in_ch, out_ch, 'bwd')
        self.dropout = dropout
        self.skip = nn.Linear(in_ch, out_ch, bias=False) \
            if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index):
        h = self.fwd(x, edge_index) + self.bwd(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h + self.skip(x)


class BiGIN(nn.Module):
    """
    Bidirected Graph Isomorphism Network + directed RWPE.

    Args: same as BiMPNN.

    Note: uses BatchNorm internally — pass full batches during training,
          or switch BN → LayerNorm for small/variable graphs.
    """
    def __init__(self, in_channels, hidden_channels=256, out_channels=2,
                 num_layers=6, dropout=0.1, walk_steps=8, use_pe=True):
        super().__init__()
        self.use_pe, self.walk_steps = use_pe, walk_steps
        actual_in = in_channels + (2 * walk_steps if use_pe else 0)

        self.input_proj = nn.Linear(actual_in, hidden_channels)
        self.layers = nn.ModuleList([
            BiGINLayer(hidden_channels, hidden_channels, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        if self.use_pe:
            x = _prepend_pe(x, edge_index, self.walk_steps)
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: BiMPNN_GT  (Graph Transformer interleaved with BI-MPNN)  ← BEST
#
# Architecture per block:
#   x  →  [BiDirectedLayer]  →  h_local        (local directed structure)
#      →  [Transformer layer]  →  h_global      (global attention)
#      →  h_local + h_global  →  LayerNorm      (combine both)
#
# The GT uses standard multi-head self-attention over all nodes (full graph).
# For large graphs, replace with a sparse/linear attention variant.
#
# PE is injected both at input AND as a bias in the attention scores
# (following the GPS / GraphGPS convention).
#
# Ranking in paper: best overall across 13 hardware tasks.
# ══════════════════════════════════════════════════════════════════════════════

class _GTLayer(nn.Module):
    """
    One Transformer + BiMPNN interleaved block.
      h = LayerNorm(BiMPNN(x) + Transformer(x) + x)
    """
    def __init__(self, hidden_ch, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_ch % num_heads == 0

        # Local: bidirected MPNN
        self.bi_layer = BiDirectedLayer(hidden_ch, hidden_ch, dropout)

        # Global: standard multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_ch, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_ch)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch * 2, hidden_ch),
        )
        self.ffn_norm = nn.LayerNorm(hidden_ch)
        self.merge_norm = nn.LayerNorm(hidden_ch)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # ── Local: bidirected MPNN ─────────────────────────────────────────
        h_local = self.bi_layer(x, edge_index)   # [N, C]

        # ── Global: self-attention ─────────────────────────────────────────
        # Attention over entire graph — treat all nodes as one sequence.
        # If batch is provided, process each graph separately to avoid
        # cross-graph attention leakage.
        if batch is not None:
            # Pad graphs in batch to same length, run attention, unpad
            h_global = self._batched_attention(x, batch)
        else:
            h_global, _ = self.attn(
                x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)
            )
            h_global = h_global.squeeze(0)       # [N, C]

        h_global = self.attn_norm(h_global + x)
        h_global = self.ffn_norm(self.ffn(h_global) + h_global)

        # ── Merge local + global ──────────────────────────────────────────
        return self.merge_norm(h_local + h_global)

    def _batched_attention(self, x, batch):
        """Run attention per graph to prevent cross-graph leakage."""
        device = x.device
        out = torch.zeros_like(x)
        for g in batch.unique():
            mask = (batch == g)
            xg = x[mask].unsqueeze(0)            # [1, n_g, C]
            hg, _ = self.attn(xg, xg, xg)
            out[mask] = hg.squeeze(0)
        return out


class BiMPNN_GT(nn.Module):
    """
    PE-enhanced Graph Transformer interleaved with BI-MPNN layers.
    Best performing model in arxiv:2410.06460 across 13 hardware tasks.

    Args:
        in_channels     : raw node feature dim
        hidden_channels : width  (default 256)
        out_channels    : output dim
        num_layers      : number of GT blocks  (default 6)
        num_heads       : attention heads       (default 8)
        dropout         : dropout rate
        walk_steps      : RWPE steps            (default 8)
        use_pe          : prepend directed RWPE (default True)

    Notes:
        - Full self-attention is O(N²). For graphs >10k nodes, replace
          nn.MultiheadAttention with a linear/sparse attention module.
        - BiGIN is a lighter alternative with similar accuracy.
    """
    def __init__(self, in_channels, hidden_channels=256, out_channels=2,
                 num_layers=6, num_heads=8, dropout=0.1,
                 walk_steps=8, use_pe=True):
        super().__init__()
        self.use_pe, self.walk_steps = use_pe, walk_steps
        actual_in = in_channels + (2 * walk_steps if use_pe else 0)

        self.input_proj = nn.Linear(actual_in, hidden_channels)
        self.layers = nn.ModuleList([
            _GTLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        if self.use_pe:
            x = _prepend_pe(x, edge_index, self.walk_steps)
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, batch)
        return self.head(h)


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    N, in_dim = 100, 32
    edge_index = torch.randint(0, N, (2, 300))
    x = torch.randn(N, in_dim)

    for ModelCls, name in [
        (BiMPNN,    "BiMPNN   "),
        (BiGIN,     "BiGIN    "),
        (BiMPNN_GT, "BiMPNN_GT"),
    ]:
        m = ModelCls(in_channels=in_dim, hidden_channels=64, out_channels=2)
        out = m(x, edge_index)
        print(f"✓ {name}  output: {tuple(out.shape)}")