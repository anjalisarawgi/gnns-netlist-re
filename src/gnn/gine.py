import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

class GINE(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super(GINE, self).__init__()
        
        # GINE also needs an MLP, but note: 
        # The input to GINEConv must match the node feature dimension.
        def create_mlp(out_c):
            return nn.Sequential(
                nn.Linear(out_c, out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c)
            )

        # We must project initial features to hidden_channels first 
        # because GINEConv expects x and edge_attr to be compatible 
        # or handled within the MLP.
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_dim, hidden_channels)

        self.conv1 = GINEConv(create_mlp(hidden_channels), edge_dim=hidden_channels)
        self.conv2 = GINEConv(create_mlp(hidden_channels), edge_dim=hidden_channels)
        self.conv3 = GINEConv(create_mlp(hidden_channels), edge_dim=hidden_channels)
        self.conv4 = GINEConv(create_mlp(hidden_channels), edge_dim=hidden_channels)
        
        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 1. Project features to a common embedding space
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # 2. Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # 3. Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # 4. Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # 5. Layer 4
        x = self.conv4(x, edge_index, edge_attr)
        
        # 6. Output mapping
        return self.final_lin(x)