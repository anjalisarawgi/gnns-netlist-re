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

# 3 layer gat
class gat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index)
        # x = F.relu(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index)
        return x #F.log_softmax(x, dim=1)






class gatv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # heads=1 is the default; dropout=0.1 here applies to the attention weights
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, dropout=0.1)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, dropout=0.1)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index)
        return x