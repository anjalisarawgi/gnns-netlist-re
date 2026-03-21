import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, RGATConv

# ──────────────────────────────────────────────────────────────────────────────
# DirectedGATBlock — preserved exactly from original
# ──────────────────────────────────────────────────────────────────────────────

class DirectedGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        per_head = out_channels // 1

        self.conv_fwd = GATv2Conv(in_channels, per_head, heads=1, concat=True, dropout=dropout)
        self.conv_bwd = GATv2Conv(in_channels, per_head, heads=1, concat=True, dropout=dropout)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)
        x_b = self.conv_bwd(x, rev_edge_index)
        return x_f + x_b   # DAG paper method


# ──────────────────────────────────────────────────────────────────────────────
# Ablation 1: Directed-only (no hierarchy), 4 DirectedGATBlocks flat
# ──────────────────────────────────────────────────────────────────────────────

class DirectedOnlyGAT(nn.Module):
    """
    Ablation: Directed-only (no hierarchy).
    4 DirectedGATBlocks stacked flat, with a single input→output residual skip.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        self.skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        self.layers = nn.ModuleList([
            DirectedGATBlock(in_channels if i == 0 else C, C, dropout=dropout)
            for i in range(4)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(C, C),
            nn.ELU(),           # ← matches original HierarchicalGAT fusion
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
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        skip = self.skip(x)

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, rev_edge_index)
            h = F.elu(h)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = h + skip
        return self.classifier(h)



class DirectedOnlyGAT_wEdges(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        # Use the edge_dim in the layers
        self.layers = nn.ModuleList([
            DirectedGATBlock(in_channels if i == 0 else C, C, edge_dim=edge_dim, dropout=dropout)
            for i in range(4)
        ])
        
        # ... (rest of your init)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # 1. Create the reverse index for the bi-directional flow
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        h = x
        for i, layer in enumerate(self.layers):
            # 2. Pass edge_attr into each block
            h = layer(h, edge_index, rev_edge_index, edge_attr)
            h = F.elu(h)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

# ──────────────────────────────────────────────────────────────────────────────
# Ablation 2a: Hierarchical-only (no directed), 2 scales = 4 layers
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalOnlyGAT4(nn.Module):
    """
    Ablation: Hierarchical-only (no directed convolution), 4 layers.
    2-scale hierarchy using standard (undirected) GATv2Conv.

    Scale 1 (local) : 2 × GATv2Conv on raw x        → 1-2 hop
    Scale 2 (mid)   : 2 × GATv2Conv on local output → 3-4 hop

    Fusion: cat(h_local, h_mid) → MLP → logits
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=dropout, concat=False)
        self.local2 = GATv2Conv(C,           C, heads=1, dropout=dropout, concat=False)
        self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        self.mid1 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)
        self.mid2 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)

        # ── Fusion: cat(local, mid) → logits ──────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.ELU(),           # ← matches original
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
        # ── Scale 1: local ────────────────────────────────────────────────
        skip = self.local_skip(x)
        h = F.elu(self.local1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.local2(h, edge_index))
        h_local = h + skip                              # residual  [N, C]

        # ── Scale 2: mid ──────────────────────────────────────────────────
        m = F.elu(self.mid1(h_local, edge_index))
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = F.elu(self.mid2(m, edge_index))
        h_mid = m + h_local                             # residual  [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        return self.fusion(
            torch.cat([h_local, h_mid], dim=-1)         # [N, 2C]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Ablation 2b: Hierarchical-only (no directed), 3 scales = 6 layers
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalOnlyGAT6(nn.Module):
    """
    Ablation: Hierarchical-only (no directed convolution), 6 layers.
    3-scale hierarchy using standard (undirected) GATv2Conv.
    Mirrors HierarchicalDirectedGAT exactly but without directed blocks.

    Scale 1 (local)  : 2 × GATv2Conv on raw x          → 1-2 hop
    Scale 2 (mid)    : 2 × GATv2Conv on local output   → 3-4 hop
    Scale 3 (global) : 2 × GATv2Conv on mid output     → 5-6 hop

    Fusion: cat(h_local, h_mid, h_global) → MLP → logits
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        # ── Scale 1: local (1-2 hops) ─────────────────────────────────────
        self.local1 = GATv2Conv(in_channels, C, heads=1, dropout=dropout, concat=False)
        self.local2 = GATv2Conv(C,           C, heads=1, dropout=dropout, concat=False)
        self.local_skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        # ── Scale 2: mid (3-4 hops) ───────────────────────────────────────
        self.mid1 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)
        self.mid2 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)

        # ── Scale 3: global (5-6 hops) ────────────────────────────────────
        self.global1 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)
        self.global2 = GATv2Conv(C, C, heads=1, dropout=dropout, concat=False)

        # ── Fusion: cat(local, mid, global) → logits ──────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(3 * C, C),
            nn.ELU(),           # ← matches original HierarchicalGAT
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
        # ── Scale 1: local ────────────────────────────────────────────────
        skip = self.local_skip(x)
        h = F.elu(self.local1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.local2(h, edge_index))
        h_local = h + skip                              # residual  [N, C]

        # ── Scale 2: mid ──────────────────────────────────────────────────
        m = F.elu(self.mid1(h_local, edge_index))
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = F.elu(self.mid2(m, edge_index))
        h_mid = m + h_local                             # residual  [N, C]

        # ── Scale 3: global ───────────────────────────────────────────────
        g = F.elu(self.global1(h_mid, edge_index))
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = F.elu(self.global2(g, edge_index))
        h_global = g + h_mid                            # residual  [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        return self.fusion(
            torch.cat([h_local, h_mid, h_global], dim=-1)  # [N, 3C]
        )

## directed + hierarchial GAT 
# ──────────────────────────────────────────────────────────────────────────────
# Full Model: Hierarchical + Directed, 3 scales = 6 layers
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalDirectedGAT_v2(nn.Module):
    """
    Full model: Hierarchical (3-scale) + Directed (fwd+bwd) GATv2.

    Each scale uses TWO DirectedGATBlocks, giving 2 message-passing hops
    per scale and 6 hops total.

    Scale 1 (local)  : 2 × DirectedGATBlock on raw x        → 1-2 hop
    Scale 2 (mid)    : 2 × DirectedGATBlock on local output → 3-4 hop
    Scale 3 (global) : 2 × DirectedGATBlock on mid output   → 5-6 hop

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
            nn.ELU(),
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
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        # ── Scale 1: local ────────────────────────────────────────────────
        skip = self.local_skip(x)
        h = F.elu(self.local1(x, edge_index, rev_edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.local2(h, edge_index, rev_edge_index))
        h_local = h + skip                              # residual  [N, C]

        # ── Scale 2: mid ──────────────────────────────────────────────────
        m = F.elu(self.mid1(h_local, edge_index, rev_edge_index))
        m = F.dropout(m, p=self.dropout, training=self.training)
        m = F.elu(self.mid2(m, edge_index, rev_edge_index))
        h_mid = m + h_local                             # residual  [N, C]

        # ── Scale 3: global ───────────────────────────────────────────────
        g = F.elu(self.global1(h_mid, edge_index, rev_edge_index))
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = F.elu(self.global2(g, edge_index, rev_edge_index))
        h_global = g + h_mid                            # residual  [N, C]

        # ── Fusion ────────────────────────────────────────────────────────
        return self.fusion(
            torch.cat([h_local, h_mid, h_global], dim=-1)  # [N, 3C]
        )




class DirectedOnlyGATWithGlobal(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        self.skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        self.layers = nn.ModuleList([
            DirectedGATBlock(in_channels if i == 0 else C, C, dropout=dropout)
            for i in range(4)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(C, C),
            nn.ELU(),
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

    def _add_global_node(self, x, edge_index):
        N = x.size(0)
        global_feat = x.mean(dim=0, keepdim=True)
        x_aug = torch.cat([x, global_feat], dim=0)

        real_ids = torch.arange(N, device=edge_index.device)
        global_id = torch.full((N,), N, device=edge_index.device)

        to_global = torch.stack([real_ids, global_id], dim=0)
        from_global = torch.stack([global_id, real_ids], dim=0)

        edge_index_aug = torch.cat([edge_index, to_global, from_global], dim=1)
        return x_aug, edge_index_aug, N

    def forward(self, x, edge_index, batch=None):
        x_aug, edge_index_aug, N = self._add_global_node(x, edge_index)

        row, col = edge_index_aug
        rev_edge_index = torch.stack([col, row], dim=0)

        skip = self.skip(x_aug)
        h = x_aug
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index_aug, rev_edge_index)
            h = F.elu(h)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = h + skip
        return self.classifier(h[:N])
