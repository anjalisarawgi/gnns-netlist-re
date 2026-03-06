import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


# # ──────────────────────────────────────────────────────────────────────────────
# # DirectedGAT
# # Two parallel GATv2 paths per layer: forward edges + reversed edges.
# # Outputs are concatenated so the model sees both in-flow and out-flow.
# # ──────────────────────────────────────────────────────────────────────────────

class DirectedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        half = hidden_channels // 2

        # Layer 1
        self.conv1_fwd = GATv2Conv(in_channels,      half,           heads=1, dropout=0.1)
        self.conv1_bwd = GATv2Conv(in_channels,      half,           heads=1, dropout=0.1)

        # Layer 2
        self.conv2_fwd = GATv2Conv(hidden_channels,  half,           heads=1, dropout=0.1)
        self.conv2_bwd = GATv2Conv(hidden_channels,  half,           heads=1, dropout=0.1)

        # Layer 3
        self.conv3_fwd = GATv2Conv(hidden_channels,  half,           heads=1, dropout=0.1)
        self.conv3_bwd = GATv2Conv(hidden_channels,  half,           heads=1, dropout=0.1)

        # Layer 4 - output
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, dropout=0.1)

    def forward(self, x, edge_index):
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        # Layer 1
        x_f = self.conv1_fwd(x, edge_index)
        x_b = self.conv1_bwd(x, rev_edge_index)
        x = torch.cat([x_f, x_b], dim=-1)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x_f = self.conv2_fwd(x, edge_index)
        x_b = self.conv2_bwd(x, rev_edge_index)
        x = torch.cat([x_f, x_b], dim=-1)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x_f = self.conv3_fwd(x, edge_index)
        x_b = self.conv3_bwd(x, rev_edge_index)
        x = torch.cat([x_f, x_b], dim=-1)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4
        x = self.conv4(x, edge_index)
        return x

##### 2 layers
# # ──────────────────────────────────────────────────────────────────────────────
# # HierarchicalGAT
# #
# # NO TopKPooling -- that removes nodes and breaks node classification.
# #
# # Instead, two "scales" are computed by separate layer stacks and their
# # outputs are concatenated before a final projection:
# #
# #   local_out  = 2 × GATv2 on raw x         (captures 1-2 hop context)
# #   global_out = 2 × GATv2 on local_out     (captures 3-4 hop context)
# #   logits     = Linear(cat[local_out, global_out])
# #
# # Residual connections stabilise training on deep stacks.
# # ──────────────────────────────────────────────────────────────────────────────

# # class HierarchicalGAT(nn.Module):
# #     def __init__(self, in_channels, hidden_channels, out_channels):
# #         super().__init__()
# #         C = hidden_channels

# #         # ── local scale (1-2 hops) ──
# #         self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=0.1, concat=False)
# #         self.local2 = GATv2Conv(C,           C, heads=1, dropout=0.1, concat=False)

# #         # project input to C for the residual add
# #         self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

# #         # ── global scale (3-4 hops, takes local repr as input) ──
# #         self.global1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
# #         self.global2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

# #         # ── fusion: concat(local, global) → logits ──
# #         self.fusion = nn.Sequential(
# #             nn.Linear(C + C, C),
# #             nn.ELU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(C, out_channels),
# #         )

# #         self._init_weights()

# #     def _init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Linear):
# #                 nn.init.xavier_uniform_(m.weight)
# #                 if m.bias is not None:
# #                     nn.init.zeros_(m.bias)

# #     def forward(self, x, edge_index, batch=None):  # batch kept for API compat, unused
# #         # ── local ──
# #         skip = self.local_skip(x)

# #         h = self.local1(x, edge_index)
# #         h = F.elu(h)
# #         h = F.dropout(h, p=0.1, training=self.training)
# #         h = self.local2(h, edge_index)
# #         h = F.elu(h)
# #         h_local = h + skip                          # residual  [N, C]

# #         # ── global ──
# #         g = self.global1(h_local, edge_index)
# #         g = F.elu(g)
# #         g = F.dropout(g, p=0.1, training=self.training)
# #         g = self.global2(g, edge_index)
# #         g = F.elu(g)
# #         h_global = g + h_local                      # residual  [N, C]

# #         # ── fuse ──
# #         out = self.fusion(torch.cat([h_local, h_global], dim=-1))  # [N, out_channels]
# #         return out

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from torch_geometric.nn import GATv2Conv



#### 3 layers
# class HierarchicalGAT(nn.Module):
#     """
#     3-scale Hierarchical GAT for node classification.

#     Scale 1 (local)   : 2 × GATv2 on raw x          → 1-2 hop receptive field
#     Scale 2 (mid)     : 2 × GATv2 on local output    → 3-4 hop receptive field
#     Scale 3 (global)  : 2 × GATv2 on mid output      → 5-6 hop receptive field

#     All three scale outputs are concatenated and fused via a small MLP.
#     Residual connections at each scale stabilise training.
#     """

#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         C = hidden_channels

#         # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
#         self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=0.1, concat=False)
#         self.local2 = GATv2Conv(C,           C, heads=1, dropout=0.1, concat=False)
#         self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

#         # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
#         self.mid1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
#         self.mid2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
#         # residual is just h_local → h_mid, same dim so no projection needed

#         # ── Scale 3: global (5-6 hops) ────────────────────────────────────
#         self.global1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
#         self.global2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

#         # ── Fusion: cat(local, mid, global) → logits ──────────────────────
#         # 3C input because we concatenate all three scales
#         self.fusion = nn.Sequential(
#             nn.Linear(3 * C, C),
#             nn.ELU(),
#             nn.Dropout(0.1),
#             nn.Linear(C, out_channels),
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x, edge_index, batch=None):
#         # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
#         skip = self.local_skip(x)

#         h = self.local1(x, edge_index)
#         h = F.elu(h)
#         h = F.dropout(h, p=0.1, training=self.training)
#         h = self.local2(h, edge_index)
#         h = F.elu(h)
#         h_local = h + skip                              # [N, C]

#         # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
#         m = self.mid1(h_local, edge_index)
#         m = F.elu(m)
#         m = F.dropout(m, p=0.1, training=self.training)
#         m = self.mid2(m, edge_index)
#         m = F.elu(m)
#         h_mid = m + h_local                             # [N, C]  residual from local

#         # ── Scale 3: global (5-6 hops) ────────────────────────────────────
#         g = self.global1(h_mid, edge_index)
#         g = F.elu(g)
#         g = F.dropout(g, p=0.1, training=self.training)
#         g = self.global2(g, edge_index)
#         g = F.elu(g)
#         h_global = g + h_mid                            # [N, C]  residual from mid

#         # ── Fusion ────────────────────────────────────────────────────────
#         # cat all 3 scales so the MLP can see each scale independently
#         out = self.fusion(
#             torch.cat([h_local, h_mid, h_global], dim=-1)  # [N, 3C]
#         )
#         return out                                      # [N, out_channels]





### 4 layers 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class HierarchicalGAT(nn.Module):
    """
    4-scale Hierarchical GAT for node classification.

    Scale 1 (local)    : 2 × GATv2 on raw x             → 1-2 hop receptive field
    Scale 2 (mid)      : 2 × GATv2 on local output       → 3-4 hop receptive field
    Scale 3 (global)   : 2 × GATv2 on mid output         → 5-6 hop receptive field
    Scale 4 (vglobal)  : 2 × GATv2 on global output      → 7-8 hop receptive field

    All four scale outputs are concatenated and fused via a small MLP.
    Residual connections chain through all scales to prevent over-smoothing.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        C = hidden_channels

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=0.1, concat=False)
        self.local2 = GATv2Conv(C,           C, heads=1, dropout=0.1, concat=False)
        self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        self.mid1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
        self.mid2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        self.global1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
        self.global2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

        # ── Scale 4: very global (7-8 hops) ───────────────────────────────
        self.vglobal1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
        self.vglobal2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

        # ── Fusion: cat(local, mid, global, vglobal) → logits ─────────────
        # 4C input because we concatenate all four scales
        self.fusion = nn.Sequential(
            nn.Linear(4 * C, C),
            nn.ELU(),
            nn.Dropout(0.1),
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
        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        skip = self.local_skip(x)

        h = self.local1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.local2(h, edge_index)
        h = F.elu(h)
        h_local = h + skip                                  # [N, C]

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        m = self.mid1(h_local, edge_index)
        m = F.elu(m)
        m = F.dropout(m, p=0.1, training=self.training)
        m = self.mid2(m, edge_index)
        m = F.elu(m)
        h_mid = m + h_local                                 # [N, C]

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        g = self.global1(h_mid, edge_index)
        g = F.elu(g)
        g = F.dropout(g, p=0.1, training=self.training)
        g = self.global2(g, edge_index)
        g = F.elu(g)
        h_global = g + h_mid                                # [N, C]

        # ── Scale 4: very global (7-8 hops) ───────────────────────────────
        v = self.vglobal1(h_global, edge_index)
        v = F.elu(v)
        v = F.dropout(v, p=0.1, training=self.training)
        v = self.vglobal2(v, edge_index)
        v = F.elu(v)
        h_vglobal = v + h_global                            # [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        out = self.fusion(
            torch.cat([h_local, h_mid, h_global, h_vglobal], dim=-1)  # [N, 4C]
        )
        return out                                          # [N, out_channels]


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


# ──────────────────────────────────────────────────────────────────────────────
# DirectedGATBlock
# A single "layer" that runs two parallel GATv2 convolutions:
#   - fwd: aggregates from predecessors (original edge direction)
#   - bwd: aggregates from successors   (reversed edge direction)
# Outputs are concatenated → [N, out_channels] (each branch is out_channels//2)
#
# This is the building block used at every scale in HierarchicalDirectedGAT.
# ──────────────────────────────────────────────────────────────────────────────

# class DirectedGATBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.1):
#         super().__init__()
#         assert out_channels % 2 == 0, "out_channels must be even (split between fwd and bwd)"
#         half = out_channels // 2

#         self.conv_fwd = GATv2Conv(in_channels, half, heads=4, dropout=dropout, concat=True)
#         self.conv_bwd = GATv2Conv(in_channels, half, heads=4, dropout=dropout, concat=True)

#     def forward(self, x, edge_index, rev_edge_index):
#         x_f = self.conv_fwd(x, edge_index)          # predecessors → node
#         x_b = self.conv_bwd(x, rev_edge_index)       # successors   → node
#         return torch.cat([x_f, x_b], dim=-1)         # [N, out_channels]

class DirectedGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        assert out_channels % 8 == 0, "out_channels must be divisible by 8 (2 directions × 4 heads)"
        per_head = out_channels // 8  # 256//8 = 32

        # 4 heads × 32 = 128 = C//2 per branch
        self.conv_fwd = GATv2Conv(in_channels, per_head, heads=4, concat=True, dropout=dropout)
        self.conv_bwd = GATv2Conv(in_channels, per_head, heads=4, concat=True, dropout=dropout)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)       # [N, 4×32] = [N, 128]
        x_b = self.conv_bwd(x, rev_edge_index)   # [N, 128]
        return torch.cat([x_f, x_b], dim=-1)     # [N, 256] = [N, C]  ✓

# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalDirectedGAT
#
# Combines two ideas that each independently improved performance:
#   (1) Hierarchical scales  — local (1-2 hop), mid (3-4 hop), global (5-6 hop)
#   (2) Directed convolution — each layer sees both fwd and bwd signal flow
#
# Each scale uses 2 × DirectedGATBlock instead of plain GATv2Conv.
# Residual connections and a 3-scale fusion MLP are preserved from HierarchicalGAT.
#
# Drop-in replacement:
#   model = HierarchicalDirectedGAT(in_channels=in_dim, hidden_channels=256, out_channels=2)
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalDirectedGAT(nn.Module):
    """
    3-scale Hierarchical + Directed GAT for node classification.

    At every scale, each GATv2Conv is replaced by a DirectedGATBlock that
    runs forward and backward message passing in parallel and concatenates
    the results — so the model explicitly sees both in-flow and out-flow
    at every level of the hierarchy.

    Scale 1 (local)  : 2 × DirectedGATBlock on raw x          → 1-2 hop
    Scale 2 (mid)    : 2 × DirectedGATBlock on local output   → 3-4 hop
    Scale 3 (global) : 2 × DirectedGATBlock on mid output     → 5-6 hop

    Fusion: cat(h_local, h_mid, h_global) → MLP → logits
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        self.local1 = DirectedGATBlock(in_channels, C, dropout=dropout)
        self.local2 = DirectedGATBlock(C,           C, dropout=dropout)
        self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        self.mid1 = DirectedGATBlock(C, C, dropout=dropout)
        self.mid2 = DirectedGATBlock(C, C, dropout=dropout)

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        self.global1 = DirectedGATBlock(C, C, dropout=dropout)
        self.global2 = DirectedGATBlock(C, C, dropout=dropout)

        # ── Fusion: cat(local, mid, global) → logits ──────────────────────
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
        # pre-compute reversed edges once, reuse across all layers and scales
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        skip = self.local_skip(x)

        h = self.local1(x, edge_index, rev_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.local2(h, edge_index, rev_edge_index)
        h = F.relu(h)
        h_local = h + skip                              # residual  [N, C]

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        m = self.mid1(h_local, edge_index, rev_edge_index)
        m = F.relu(m)
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = self.mid2(m, edge_index, rev_edge_index)
        m = F.relu(m)
        h_mid = m + h_local                             # residual  [N, C]

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        g = self.global1(h_mid, edge_index, rev_edge_index)
        g = F.relu(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = self.global2(g, edge_index, rev_edge_index)
        g = F.relu(g)
        h_global = g + h_mid                            # residual  [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        out = self.fusion(
            torch.cat([h_local, h_mid, h_global], dim=-1)  # [N, 3C]
        )
        return out                                      # [N, out_channels]