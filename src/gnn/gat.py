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



# In your GAT model, increase layers and receptive field
class gat_new(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # 4 layers instead of 2-3 (see further in the graph)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.6)
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.6)
        self.conv4 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels * 8)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 8)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 8)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv4(x, edge_index)
        
        return x

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