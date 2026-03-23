import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv
import torch.nn as nn


class graphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)
        

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
        return x

from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE_ResNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        # --- Layers ---
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)

        # --- LayerNorms ---
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)

        # --- Input projection for first residual ---
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