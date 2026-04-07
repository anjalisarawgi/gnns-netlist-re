import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, JumpingKnowledge
import torch.nn as nn


class JK_GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1, jk_mode='cat'):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(in_channels, hidden_channels, project=True),
            SAGEConv(hidden_channels, hidden_channels, project=True),
            SAGEConv(hidden_channels, hidden_channels, project=True),
            SAGEConv(hidden_channels, hidden_channels, project=True),
            SAGEConv(hidden_channels, hidden_channels, project=True),
            SAGEConv(hidden_channels, hidden_channels, project=True),
        ])

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=6)

        # JK 'cat' concatenates all 6 layer outputs → hidden_channels * 6
        jk_out_channels = hidden_channels * 6 if jk_mode == 'cat' else hidden_channels
        self.lin = nn.Linear(jk_out_channels, out_channels)

        self.dropout = dropout
        self.jk_mode = jk_mode

    def forward(self, x, edge_index):
        layer_outs = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        x = self.jk(layer_outs)   # aggregate all layer representations
        x = self.lin(x)           # project to output dimension
        return x