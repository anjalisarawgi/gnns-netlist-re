from torch_geometric.nn import SAGEConv
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE_JK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # JK classifier
        self.classifier = nn.Linear(hidden_channels * num_layers, out_channels)

    def forward(self, x, edge_index):
        xs = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            xs.append(x)

            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Jumping Knowledge (concat)
        x = torch.cat(xs, dim=1)

        return self.classifier(x)