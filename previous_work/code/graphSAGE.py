import torch
from torch import nn
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.1):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:  # [:-1]:
            x = conv(x, edge_index)  # Apply SAGEConv layer
            x = torch.relu(x)  # Apply ReLU activation
            x = self.dropout(x)
        x = self.fc(x)
        return x
