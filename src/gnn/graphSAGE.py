import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv
import torch.nn as nn


class graphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels )
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)
        # self.skip = nn.Linear(in_channels, out_channels, bias=False)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # residual = self.skip(x) 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index)
        # x = x + residual 
        return  x

