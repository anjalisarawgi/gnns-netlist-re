import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, JumpingKnowledge

## here we combine both bi directions and jk setup for testing purposes
class BiDirectedSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fwd = SAGEConv(in_channels, out_channels, project=True)
        self.conv_bwd = SAGEConv(in_channels, out_channels, project=True)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)
        x_b = self.conv_bwd(x, rev_edge_index)
        return x_f + x_b


class BiDirectedJK_GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, dropout=0.1, jk_mode='cat'):
        super().__init__()
        self.dropout = dropout
        self.jk_mode = jk_mode

        # bidirec
        self.layers = nn.ModuleList([
            BiDirectedSAGEBlock(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers) # jk

        jk_out_channels = hidden_channels * num_layers if jk_mode == 'cat' else hidden_channels
        self.lin = nn.Linear(jk_out_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        layer_outs = []
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, rev_edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outs.append(h)

        h = self.jk(layer_outs)
        return self.lin(h)
