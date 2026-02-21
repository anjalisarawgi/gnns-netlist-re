import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn


# class gat(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels)
#         self.conv2 = GATConv(hidden_channels, hidden_channels)
#         self.conv3 = GATConv(hidden_channels, hidden_channels)
#         self.conv4 = GATConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index, edge_attr=None, return_embeddings=False):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)

#         embeddings = x 
#         logits = self.conv4(embeddings, edge_index)
#         if return_embeddings:
#             return logits, embeddings

#         # x = self.conv4(x, edge_index)
#         return logits #F.log_softmax(x, dim=1)


class gat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        emb = F.relu(self.conv3(x, edge_index))
        logits = self.lin(emb)

        return (logits, emb) if return_embeddings else logits


# class gat(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
#         super().__init__()

#         # Layer 1
#         self.conv1 = GATConv(
#             in_channels,
#             hidden_channels,
#             heads=heads,
#             concat=True
#         )

#         # After concat: dim = hidden_channels * heads
#         self.conv2 = GATConv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             concat=True
#         )

#         self.conv3 = GATConv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             concat=True
#         )

#         # Final layer: usually single head, no concat
#         self.conv4 = GATConv(
#             hidden_channels * heads,
#             out_channels,
#             heads=1,
#             concat=False
#         )

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = F.relu(self.conv3(x, edge_index))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = self.conv4(x, edge_index)
#         return x


from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

class GAT_E(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=4):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv4 = GATConv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index, edge_attr)
        return x


# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
# import torch.nn as nn


# class gatE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  heads=4, edge_dim=5):
#         super().__init__()

#         self.conv1 = GATv2Conv(
#             in_channels,
#             hidden_channels,
#             heads=heads,
#             concat=True,
#             edge_dim=edge_dim
#         )

#         self.conv2 = GATv2Conv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             concat=True,
#             edge_dim=edge_dim
#         )

#         self.conv3 = GATv2Conv(
#             hidden_channels * heads,
#             hidden_channels,
#             heads=heads,
#             concat=True,
#             edge_dim=edge_dim
#         )

#         self.conv4 = GATv2Conv(
#             hidden_channels * heads,
#             out_channels,
#             heads=1,
#             concat=False,
#             edge_dim=edge_dim
#         )

#     def forward(self, x, edge_index, edge_attr):
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = F.relu(self.conv3(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.1, training=self.training)

#         x = self.conv4(x, edge_index, edge_attr)
#         return x

# # In your GAT model, increase layers and receptive field
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# import torch.nn as nn


# class gat(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.1):
#         super().__init__()
#         self.dropout = dropout

#         # layer 1: in_channels -> hidden_channels * heads
#         self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

#         # layer 2: hidden_channels * heads -> hidden_channels * heads
#         self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)

#         # layer 3: hidden_channels * heads -> hidden_channels * heads (reduced heads)
#         self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=4, dropout=dropout)

#         # layer 4 (output): hidden_channels * 4 -> out_channels (single head)
#         self.conv4 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=dropout)

#         # projection layers for residual connections (to match dimensions)
#         # skip from layer1 output -> layer3 input: hidden*8 -> hidden*8 (same, no proj needed)
#         # skip from layer2 output -> layer4 input: hidden*8 -> hidden*4 (need proj)
#         self.skip_proj = nn.Linear(hidden_channels * heads, hidden_channels * 4)

#     def forward(self, x, edge_index):
#         # layer 1
#         x1 = self.conv1(x, edge_index)          # -> [N, hidden * heads]
#         x1 = F.elu(x1)                           # ELU is standard for GAT (vs ReLU)
#         x1 = F.dropout(x1, p=self.dropout, training=self.training)

#         # layer 2
#         x2 = self.conv2(x1, edge_index)          # -> [N, hidden * heads]
#         x2 = F.elu(x2)
#         x2 = F.dropout(x2, p=self.dropout, training=self.training)

#         # layer 3 with skip from layer 1
#         x3 = self.conv3(x2 + x1, edge_index)    # -> [N, hidden * 4]
#         x3 = F.elu(x3)
#         x3 = F.dropout(x3, p=self.dropout, training=self.training)

#         # layer 4 with skip from layer 2 (projected to match dims)
#         x2_proj = self.skip_proj(x2)             # -> [N, hidden * 4]
#         x4 = self.conv4(x3 + x2_proj, edge_index)  # -> [N, out_channels]

#         return x4


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv

# class GNNRE_GAT(nn.Module):
#     """
#     PyG version of the GNN-RE GAT configuration:
#       - Multi-head attention K=8 with concatenation
#       - ReLU activation
#       - Dropout 0.1
#       - Final FC classifier (softmax applied outside for metrics)
#     Paper reference: Alrahis et al. (GNN-RE), Table III + text.  [oai_citation:2‡Alrahis et al. - 2021 - GNN-RE Graph Neural Networks for Reverse Engineer.pdf](sediment://file_00000000d74c720a97089b88d8cc93de)
#     """
#     def __init__(self, in_channels: int, hidden_dim: int = 256, out_channels: int = 2,
#                  heads: int = 8, dropout: float = 0.1, num_gat_layers: int = 4):
#         super().__init__()
#         assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads when using concat=True."
#         self.dropout = dropout
#         self.hidden_dim = hidden_dim
#         self.heads = heads

#         per_head = hidden_dim // heads  # e.g., 256//8 = 32 -> concat => 256

#         self.convs = nn.ModuleList()
#         # Layer 1: in -> hidden_dim (via per_head x heads concat)
#         self.convs.append(GATConv(in_channels, per_head, heads=heads, concat=True, dropout=dropout))

#         # Layers 2..num_gat_layers: hidden_dim -> hidden_dim
#         for _ in range(num_gat_layers - 1):
#             self.convs.append(GATConv(hidden_dim, per_head, heads=heads, concat=True, dropout=dropout))

#         # Final classifier (their "fc layer with softmax")
#         self.classifier = nn.Linear(hidden_dim, out_channels)

#     def forward(self, x, edge_index):
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = F.relu(x)  # paper uses ReLU  [oai_citation:3‡Alrahis et al. - 2021 - GNN-RE Graph Neural Networks for Reverse Engineer.pdf](sediment://file_00000000d74c720a97089b88d8cc93de)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         logits = self.classifier(x)
#         return logits  # use cross_entropy on logits