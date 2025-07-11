import torch
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_layers=3, dropout=0.1):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 3):
            self.convs.append(GATConv((hidden_channels * heads), hidden_channels, heads))
        self.convs.append(GATConv((hidden_channels * heads), hidden_channels, concat=False))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.fc(x)
        return x
