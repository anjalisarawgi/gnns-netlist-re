import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import degree


# ──────────────────────────────────────────────────────────────────────────────
# Directed Random Walk Positional Encoding (RWPE)
#
# Keeps fwd and bwd walks SEPARATE so the model knows:
#   fwd: "which nodes can I reach?"
#   bwd: "which nodes can reach me?"
#
# Output: [N, 2 * walk_steps]
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# DirectedGATBlock  (unchanged from your original)
# ──────────────────────────────────────────────────────────────────────────────

class DirectedGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        # assert out_channels % 8 == 0, "out_channels must be divisible by 8 (2 directions × 4 heads)"
        # per_head = out_channels // 8  # 256//8 = 32

        self.conv_fwd = GATv2Conv(in_channels, per_head, heads=1, concat=True, dropout=dropout)
        self.conv_bwd = GATv2Conv(in_channels, per_head, heads=1, concat=True, dropout=dropout)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)       # [N, 128]
        x_b = self.conv_bwd(x, rev_edge_index)   # [N, 128]
        return torch.cat([x_f, x_b], dim=-1)     # [N, 256]


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalDirectedGAT  +  directed RWPE
#
# Only change from your original:
#   1. __init__ accepts walk_steps and use_pe args
#   2. input projection accounts for the extra PE dimensions
#   3. forward() prepends directed RWPE to x before Scale 1
#
# Everything else — blocks, scales, residuals, fusion — is identical.
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalDirectedGAT(nn.Module):
    """
    3-scale Hierarchical + Directed GAT with directed RWPE.

    Scale 1 (local)  : 2 × DirectedGATBlock on raw x        → 1-2 hop
    Scale 2 (mid)    : 2 × DirectedGATBlock on local output  → 3-4 hop
    Scale 3 (global) : 2 × DirectedGATBlock on mid output    → 5-6 hop

    Fusion: cat(h_local, h_mid, h_global) → MLP → logits

    New args vs original:
        use_pe     : prepend directed RWPE to node features (default True)
        walk_steps : number of random walk steps per direction (default 8)
                     PE dim = 2 * walk_steps
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.1, use_pe=True, walk_steps=8):
        super().__init__()
        C = hidden_channels
        self.dropout   = dropout
        self.use_pe    = use_pe
        self.walk_steps = walk_steps

        # If using PE, the first DirectedGATBlock sees in_channels + 2*walk_steps
        actual_in = in_channels + (2 * walk_steps if use_pe else 0)

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        self.local1 = DirectedGATBlock(actual_in, C, dropout=dropout)
        self.local2 = DirectedGATBlock(C,         C, dropout=dropout)
        self.local_skip = nn.Linear(actual_in, C, bias=False) \
            if actual_in != C else nn.Identity()

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        self.mid1 = DirectedGATBlock(C, C, dropout=dropout)
        self.mid2 = DirectedGATBlock(C, C, dropout=dropout)

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        self.global1 = DirectedGATBlock(C, C, dropout=dropout)
        self.global2 = DirectedGATBlock(C, C, dropout=dropout)

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(3 * C, C),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(C, out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        # ── Directed RWPE ─────────────────────────────────────────────────
        if self.use_pe:
            pe = directed_rwpe(edge_index, x.size(0), self.walk_steps)
            x = torch.cat([x, pe], dim=-1)          # [N, in_channels + 2*k]

        # pre-compute reversed edges once
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        skip = self.local_skip(x)

        h = self.local1(x, edge_index, rev_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.local2(h, edge_index, rev_edge_index)
        h = F.relu(h)
        h_local = h + skip                              # [N, C]

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        m = self.mid1(h_local, edge_index, rev_edge_index)
        m = F.relu(m)
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = self.mid2(m, edge_index, rev_edge_index)
        m = F.relu(m)
        h_mid = m + h_local                             # [N, C]

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        g = self.global1(h_mid, edge_index, rev_edge_index)
        g = F.relu(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = self.global2(g, edge_index, rev_edge_index)
        g = F.relu(g)
        h_global = g + h_mid                            # [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        out = self.fusion(
            torch.cat([h_local, h_mid, h_global], dim=-1)
        )
        return out                                      # [N, out_channels]


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N, in_dim = 100, 32
    edge_index = torch.randint(0, N, (2, 300))
    x = torch.randn(N, in_dim)

    # With PE (new default)
    model = HierarchicalDirectedGAT(
        in_channels=in_dim, hidden_channels=64, out_channels=2,
        use_pe=True, walk_steps=8
    )
    out = model(x, edge_index)
    print(f"✓ with PE     output: {tuple(out.shape)}")

    # Without PE (identical to your original)
    model_no_pe = HierarchicalDirectedGAT(
        in_channels=in_dim, hidden_channels=64, out_channels=2,
        use_pe=False
    )
    out2 = model_no_pe(x, edge_index)
    print(f"✓ without PE  output: {tuple(out2.shape)}")