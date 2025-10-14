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
        return x #F.log_softmax(x, dim=1)



# class gat(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, attn_dropout=0.3):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, dropout=attn_dropout)
#         self.conv2 = GATConv(hidden_channels, hidden_channels, dropout=attn_dropout)
#         self.conv3 = GATConv(hidden_channels, hidden_channels, dropout=attn_dropout)
#         self.conv4 = GATConv(hidden_channels, out_channels, dropout=attn_dropout)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv4(x, edge_index)
#         return x  # raw logits