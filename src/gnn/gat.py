import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):  # keep signature same
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin3(x)
        return x

# # 4 layer gat
class gat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index)
        return x #F.log_softmax(x, dim=1)






class gatv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, dropout=0.1, concat = False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=2, dropout=0.1, concat = False)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1, concat = False)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, dropout=0.1, concat = False)

        # self.conv1 = GATv2Conv(in_channels,            hidden_channels,     heads=4, dropout=0.1, concat=True)
        # self.conv2 = GATv2Conv(hidden_channels * 4,    hidden_channels,     heads=4, dropout=0.1, concat=True)
        # self.conv3 = GATv2Conv(hidden_channels * 4,    hidden_channels,     heads=4, dropout=0.1, concat=True)
        # self.conv4 = GATv2Conv(hidden_channels * 4,    out_channels,        heads=4, dropout=0.1, concat=False)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index)
        return x



class GATv2_ResNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        # --- Layers ---
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=4, concat=False, dropout=dropout)

        # --- LayerNorms ---
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)

        # --- Input projection for residual (important!) ---
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x, edge_index):
        # ---- Layer 1 ----
        residual = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 2 ----
        residual = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 3 ----
        residual = x
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Output ----
        x = self.conv4(x, edge_index)
        return x


import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class gatv2_wEdges(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__()
        # edge_dim is the number of features per edge (e.g., 3 in your case)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index, edge_attr)
        return x




##################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels

        # GATv2 with concat=True to keep heads separate
        self.gat = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        # gating network (per head)
        self.gate = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)  # [N, heads * out_channels]

        N = x.size(0)

        # reshape to [N, heads, out_channels]
        x = x.view(N, self.heads, self.out_channels)

        # compute gate scores
        gate_scores = torch.sigmoid(self.gate(x))  # [N, heads, 1]

        # apply gating
        x = x * gate_scores

        # combine heads (sum or mean)
        x = x.sum(dim=1)  # [N, out_channels]

        return x

class GAAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()

        self.dropout = dropout

        self.conv1 = GAANLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GAANLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GAANLayer(hidden_channels, hidden_channels, heads=heads, dropout=dropout)

        # final layer → no gating needed (optional)
        self.conv4 = GATv2Conv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.conv4(x, edge_index)

        return x