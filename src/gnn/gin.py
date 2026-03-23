import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        
        # GIN requires an MLP (Multi-Layer Perceptron) inside each layer
        # to achieve its theoretical power. 
        def create_mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.BatchNorm1d(out_c), # CRITICAL: Keeps gradients alive
                nn.ReLU(),
                nn.Linear(out_c, out_c)
            )

        self.conv1 = GINConv(create_mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(create_mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(create_mlp(hidden_channels, hidden_channels))
        self.conv4 = GINConv(create_mlp(hidden_channels, out_channels))

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