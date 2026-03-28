# from torch_geometric.nn import SAGEConv
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DirectedSAGEBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv_fwd = SAGEConv(in_channels, out_channels)
#         self.conv_bwd = SAGEConv(in_channels, out_channels)

#     def forward(self, x, edge_index, rev_edge_index):
#         x_f = self.conv_fwd(x, edge_index)
#         x_b = self.conv_bwd(x, rev_edge_index)

#         return x_f + x_b #torch.cat([x_f, x_b], dim=1) ## concet Optio 2: #x_f + x_b    # adding 

# ####### without resnorm- our main baseline
# class BiDirectedGraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()

#         self.conv1 = DirectedSAGEBlock(in_channels, hidden_channels)
#         self.conv2 = DirectedSAGEBlock(hidden_channels, hidden_channels)
#         self.conv3 = DirectedSAGEBlock(hidden_channels, hidden_channels)
#         self.conv4 = DirectedSAGEBlock(hidden_channels, out_channels)

#         self.dropout = 0.1

#     def forward(self, x, edge_index):
#         # Create reverse edges once
#         row, col = edge_index
#         rev_edge_index = torch.stack([col, row], dim=0)

#         x = self.conv1(x, edge_index, rev_edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index, rev_edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index, rev_edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv4(x, edge_index, rev_edge_index)

#         return x

# # class BiDirectedGraphSAGE_wResNorm(nn.Module):
# #     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.1):
# #         super().__init__()
# #         self.dropout = dropout
# #         self.num_layers = num_layers

# #         # --- Layers ---
# #         self.layers = nn.ModuleList()
# #         self.norms = nn.ModuleList()

# #         # First layer
# #         self.layers.append(DirectedSAGEBlock(in_channels, hidden_channels))
# #         self.norms.append(nn.LayerNorm(hidden_channels))

# #         # Hidden layers
# #         for _ in range(num_layers - 2):
# #             self.layers.append(DirectedSAGEBlock(hidden_channels, hidden_channels))
# #             self.norms.append(nn.LayerNorm(hidden_channels))

# #         # Output layer
# #         self.layers.append(DirectedSAGEBlock(hidden_channels, out_channels))

# #         # Input projection for first residual
# #         self.input_proj = (
# #             nn.Linear(in_channels, hidden_channels)
# #             if in_channels != hidden_channels
# #             else nn.Identity()
# #         )

# #     def forward(self, x, edge_index):

# #         # reverse edges
# #         row, col = edge_index
# #         rev_edge_index = torch.stack([col, row], dim=0)

# #         # ---- First layer ----
# #         residual = self.input_proj(x)
# #         x = self.layers[0](x, edge_index, rev_edge_index)
# #         x = self.norms[0](x)
# #         x = F.relu(x)
# #         x = x + residual
# #         x = F.dropout(x, p=self.dropout, training=self.training)

# #         # ---- Middle layers ----
# #         for i in range(1, self.num_layers - 1):
# #             residual = x
# #             x = self.layers[i](x, edge_index, rev_edge_index)
# #             x = self.norms[i](x)
# #             x = F.relu(x)
# #             x = x + residual
# #             x = F.dropout(x, p=self.dropout, training=self.training)

# #         # ---- Output ----
# #         x = self.layers[-1](x, edge_index, rev_edge_index)

# #         return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class BiDirectedSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fwd = SAGEConv(in_channels, out_channels)
        self.conv_bwd = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index, rev_edge_index):
        x_f = self.conv_fwd(x, edge_index)
        x_b = self.conv_bwd(x, rev_edge_index)
        return x_f + x_b  # same as DirectedGATBlock


class BiDirectedGraphSAGE(nn.Module):
    """
    Mirrors DirectedOnlyGAT structure:
    - input residual skip
    - N stacked BiDirectedSAGEBlocks with ELU + dropout
    - 2-layer classifier head
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.1):
        super().__init__()
        C = hidden_channels
        self.dropout = dropout

        self.skip = nn.Linear(in_channels, C, bias=False) if in_channels != C else nn.Identity()

        self.layers = nn.ModuleList([
            BiDirectedSAGEBlock(in_channels if i == 0 else C, C)
            for i in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(C, C),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(C, out_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        row, col = edge_index
        rev_edge_index = torch.stack([col, row], dim=0)

        skip = self.skip(x)

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, rev_edge_index)
            h = F.elu(h)
            if i < len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = h + skip
        return self.classifier(h)