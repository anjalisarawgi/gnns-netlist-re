

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BiDirectedSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fwd = SAGEConv(in_channels, out_channels, project = True)
        self.conv_bwd = SAGEConv(in_channels, out_channels, project = True)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)
        x_b = self.conv_bwd(x, rev_edge_index)
        return x_f + x_b  # same as DirectedGATBlock


class BiDirectedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        self.layers = nn.ModuleList([
            BiDirectedSAGEBlock(in_channels if i == 0 else C, C)
            for i in range(num_layers)
        ])

        self.out = nn.Linear(C, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, rev_edge_index)
            h = F.relu(h)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return self.out(h)