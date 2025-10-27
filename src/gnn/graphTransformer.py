import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GraphTransformer, self).__init__()

        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv4 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False)

        self.dropout = dropout
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        return x