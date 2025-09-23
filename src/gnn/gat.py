import torch.nn.functional as F 
from torch_geometric.nn import GATConv
import torch.nn as nn


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
        return F.log_softmax(x, dim=1)


# class gat(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, heads=4):
#         super().__init__()
#         self.dropout = dropout
#         self.heads = heads

#         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
#         self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
#         self.conv4 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

#         self.norm1 = nn.LayerNorm(hidden_channels * heads)
#         self.norm2 = nn.LayerNorm(hidden_channels * heads)
#         self.norm3 = nn.LayerNorm(hidden_channels * heads)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.norm1(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.norm2(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.norm3(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv4(x, edge_index)
#         return x  # No log_softmax