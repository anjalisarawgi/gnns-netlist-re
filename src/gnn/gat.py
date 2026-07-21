import torch
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None): 
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin3(x)
        return x

# # 4 layer gat
class gat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
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
        return x #F.log_softmax(x, dim=1)






class gatv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.1):
        super().__init__()

        self.dropout = dropout

        # self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat = False)
        # self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat = False)
        # self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat = False)
        # self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, concat = False)

        self.conv1 = GATv2Conv(in_channels,            hidden_channels,     heads=8,  concat=True)
        self.conv2 = GATv2Conv(hidden_channels * 8,    hidden_channels,     heads=8, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * 8,    hidden_channels,     heads=8, concat=True)
        self.conv4 = GATv2Conv(hidden_channels * 8,    out_channels,        heads=8, concat=False)

    def forward(self, x, edge_index):
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
        return x


## GAAN
class GAAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.conv1 = GATv2Conv(in_channels,           hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, concat=True)
        self.conv4 = GATv2Conv(hidden_channels*heads, out_channels,    heads=heads, concat=False)

        self.gate1 = nn.Linear(in_channels,           heads)
        self.gate2 = nn.Linear(hidden_channels*heads, heads)
        self.gate3 = nn.Linear(hidden_channels*heads, heads)

    def _apply_gate(self, x_in, conv, gate_net, edge_index):
        g = torch.softmax(gate_net(x_in), dim=-1)         
        x_out = conv(x_in, edge_index)                
        x_out = x_out.view(-1, self.heads, self.hidden_channels)  
        x_out = x_out * g.unsqueeze(-1)                          
        x_out = x_out.view(-1, self.heads * self.hidden_channels)

        return x_out

    def forward(self, x, edge_index):
        x = self._apply_gate(x, self.conv1, self.gate1, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._apply_gate(x, self.conv2, self.gate2, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._apply_gate(x, self.conv3, self.gate3, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x
