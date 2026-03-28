import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv
import torch.nn as nn


class graphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index)
        return x


# from torch_geometric.nn import SAGEConv
# import torch.nn as nn
# import torch.nn.functional as F

# class GraphSAGE_ResNorm(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.1):
#         super().__init__()
#         self.dropout = dropout
#         self.num_layers = num_layers

#         # --- Convs ---
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()

#         # First layer
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
#         self.norms.append(nn.LayerNorm(hidden_channels))

#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels))
#             self.norms.append(nn.LayerNorm(hidden_channels))

#         # Output layer
#         self.convs.append(SAGEConv(hidden_channels, out_channels))

#         # Input projection for first residual
#         self.input_proj = (
#             nn.Linear(in_channels, hidden_channels)
#             if in_channels != hidden_channels
#             else nn.Identity()
#         )

#     def forward(self, x, edge_index):

#         # ---- First layer (special residual) ----
#         residual = self.input_proj(x)
#         x = self.convs[0](x, edge_index)
#         x = self.norms[0](x)
#         x = F.relu(x)
#         x = x + residual
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # ---- Middle layers ----
#         for i in range(1, self.num_layers - 1):
#             residual = x
#             x = self.convs[i](x, edge_index)
#             x = self.norms[i](x)
#             x = F.relu(x)
#             x = x + residual
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         # ---- Output layer (no residual, no norm) ----
#         x = self.convs[-1](x, edge_index)

#         return x