# import torch.nn.functional as F 
# from torch_geometric.nn import GATConv, GATv2Conv
# import torch.nn as nn

# class DirectedGAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         # We define two paths: one for forward flow, one for backward flow
#         # The hidden channels are halved because we concatenate them
#         half_hidden = hidden_channels // 2
        
#         self.conv1_fwd = GATv2Conv(in_channels, half_hidden)
#         self.conv1_bwd = GATv2Conv(in_channels, half_hidden)
        
#         self.conv2_fwd = GATv2Conv(hidden_channels, half_hidden)
#         self.conv2_bwd = GATv2Conv(hidden_channels, half_hidden)
        
#         self.conv3_fwd = GATv2Conv(hidden_channels, half_hidden)
#         self.conv3_bwd = GATv2Conv(hidden_channels, half_hidden)
        
#         # Final layer merges them to out_channels
#         self.conv4 = GATv2Conv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         # Create reversed edges for the backward path
#         row, col = edge_index
#         rev_edge_index = torch.stack([col, row], dim=0)

#         # Layer 1
#         x_f = self.conv1_fwd(x, edge_index)
#         x_b = self.conv1_bwd(x, rev_edge_index)
#         x = torch.cat([x_f, x_b], dim=-1)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         # Layer 2
#         x_f = self.conv2_fwd(x, edge_index)
#         x_b = self.conv2_bwd(x, rev_edge_index)
#         x = torch.cat([x_f, x_b], dim=-1)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         # Layer 3
#         x_f = self.conv3_fwd(x, edge_index)
#         x_b = self.conv3_bwd(x, rev_edge_index)
#         x = torch.cat([x_f, x_b], dim=-1)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         # Layer 4 (Output)
#         x = self.conv4(x, edge_index)
#         return x


# from torch_geometric.nn import GATv2Conv, TopKPooling
# from torch_geometric.utils import dropout_adj

# class HierarchicalGAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATv2Conv(in_channels, hidden_channels)
#         self.pool1 = TopKPooling(hidden_channels, ratio=0.5) # Pool 50% of nodes
        
#         self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
#         self.pool2 = TopKPooling(hidden_channels, ratio=0.5) # Pool 50% again
        
#         self.conv3 = GATv2Conv(hidden_channels, hidden_channels)
        
#         # Output layer
#         self.conv4 = GATv2Conv(hidden_channels, out_channels)

#     def forward(self, x, edge_index, batch=None):
#         # If batch is None (single large graph), create a zero batch index
#         if batch is None:
#             batch = edge_index.new_zeros(x.size(0))

#         # Layer 1 + Pooling
#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
#         # Layer 2 + Pooling (The graph is now 25% of original size)
#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

#         # Layer 3 (Processing the "Global" context)
#         x = F.relu(self.conv3(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)

#         # Note: Hierarchical models usually need an Upsampling or Global Readout.
#         # For node classification, we skip the final pooling to return all nodes, 
#         # or we use the pooled features to classify the original nodes.
#         x = self.conv4(x, edge_index)
#         return x


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


# ──────────────────────────────────────────────────────────────────────────────
# DirectedGAT
# Two parallel GATv2 paths per layer: forward edges + reversed edges.
# Outputs are concatenated so the model sees both in-flow and out-flow.
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalGAT
#
# NO TopKPooling -- that removes nodes and breaks node classification.
#
# Instead, two "scales" are computed by separate layer stacks and their
# outputs are concatenated before a final projection:
#
#   local_out  = 2 × GATv2 on raw x         (captures 1-2 hop context)
#   global_out = 2 × GATv2 on local_out     (captures 3-4 hop context)
#   logits     = Linear(cat[local_out, global_out])
#
# Residual connections stabilise training on deep stacks.
# ──────────────────────────────────────────────────────────────────────────────

# class HierarchicalGAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         C = hidden_channels

#         # ── local scale (1-2 hops) ──
#         self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=0.1, concat=False)
#         self.local2 = GATv2Conv(C,           C, heads=1, dropout=0.1, concat=False)

#         # project input to C for the residual add
#         self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

#         # ── global scale (3-4 hops, takes local repr as input) ──
#         self.global1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
#         self.global2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

#         # ── fusion: concat(local, global) → logits ──
#         self.fusion = nn.Sequential(
#             nn.Linear(C + C, C),
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

#     def forward(self, x, edge_index, batch=None):  # batch kept for API compat, unused
#         # ── local ──
#         skip = self.local_skip(x)

#         h = self.local1(x, edge_index)
#         h = F.elu(h)
#         h = F.dropout(h, p=0.1, training=self.training)
#         h = self.local2(h, edge_index)
#         h = F.elu(h)
#         h_local = h + skip                          # residual  [N, C]

#         # ── global ──
#         g = self.global1(h_local, edge_index)
#         g = F.elu(g)
#         g = F.dropout(g, p=0.1, training=self.training)
#         g = self.global2(g, edge_index)
#         g = F.elu(g)
#         h_global = g + h_local                      # residual  [N, C]

#         # ── fuse ──
#         out = self.fusion(torch.cat([h_local, h_global], dim=-1))  # [N, out_channels]
#         return out

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class HierarchicalGAT(nn.Module):
    """
    3-scale Hierarchical GAT for node classification.

    Scale 1 (local)   : 2 × GATv2 on raw x          → 1-2 hop receptive field
    Scale 2 (mid)     : 2 × GATv2 on local output    → 3-4 hop receptive field
    Scale 3 (global)  : 2 × GATv2 on mid output      → 5-6 hop receptive field

    All three scale outputs are concatenated and fused via a small MLP.
    Residual connections at each scale stabilise training.
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
        # residual is just h_local → h_mid, same dim so no projection needed

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        self.global1 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)
        self.global2 = GATv2Conv(C, C, heads=1, dropout=0.1, concat=False)

        # ── Fusion: cat(local, mid, global) → logits ──────────────────────
        # 3C input because we concatenate all three scales
        self.fusion = nn.Sequential(
            nn.Linear(3 * C, C),
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
        h_local = h + skip                              # [N, C]

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        m = self.mid1(h_local, edge_index)
        m = F.elu(m)
        m = F.dropout(m, p=0.1, training=self.training)
        m = self.mid2(m, edge_index)
        m = F.elu(m)
        h_mid = m + h_local                             # [N, C]  residual from local

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        g = self.global1(h_mid, edge_index)
        g = F.elu(g)
        g = F.dropout(g, p=0.1, training=self.training)
        g = self.global2(g, edge_index)
        g = F.elu(g)
        h_global = g + h_mid                            # [N, C]  residual from mid

        # ── Fusion ────────────────────────────────────────────────────────
        # cat all 3 scales so the MLP can see each scale independently
        out = self.fusion(
            torch.cat([h_local, h_mid, h_global], dim=-1)  # [N, 3C]
        )
        return out                                      # [N, out_channels]
