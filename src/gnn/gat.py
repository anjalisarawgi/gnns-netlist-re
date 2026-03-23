import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):  # keep signature same
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin3(x)
        return x

# 4 layer gat
class gat(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv3(x, edge_index)
        # x = F.relu(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv4(x, edge_index)
        return x #F.log_softmax(x, dim=1)






class gatv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # heads=1 is the default; dropout=0.1 here applies to the attention weights
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, dropout=0.1, concat = False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=4, dropout=0.1, concat = False)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=4, dropout=0.1, concat = False)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=4, dropout=0.1, concat = False)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index)
        return x



import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class gatv2_wEdges(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__()
        # edge_dim is the number of features per edge (e.g., 3 in your case)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, dropout=0.1, edge_dim=edge_dim)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=1, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index, edge_attr)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv

# class GaAN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
#         super().__init__()
#         self.heads = heads
        
#         # We use GATv2 as the base 'Attention' engine
#         self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.1, concat=True)
        
#         # The 'Gating' mechanism: A simple linear layer that looks at the 
#         # combined features and decides which head to trust.
#         # It maps from the concatenated output (hidden * heads) back to a weight for each head.
#         self.gate1 = nn.Sequential(
#             nn.Linear(hidden_channels * heads, heads),
#             nn.Sigmoid()
#         )
        
#         # Second layer (Transition)
#         self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1, concat=True)
#         self.gate2 = nn.Sequential(
#             nn.Linear(hidden_channels * heads, heads),
#             nn.Sigmoid()
#         )

#         # Output Layer (Reduced to out_channels)
#         self.out_proj = nn.Linear(hidden_channels * heads, out_channels)

#     def forward(self, x, edge_index):
#         # --- Layer 1 ---
#         # Get attention output from all heads
#         out1 = self.conv1(x, edge_index) # Shape: [Nodes, hidden_channels * heads]
        
#         # Calculate gate values for each head
#         g1 = self.gate1(out1) # Shape: [Nodes, heads]
        
#         # Apply gating: multiply each head's features by its gate value
#         # This is the 'GaAN' secret sauce!
#         out1 = out1.view(-1, self.heads, out1.size(-1) // self.heads)
#         out1 = out1 * g1.unsqueeze(-1)
#         out1 = out1.view(-1, out1.size(1) * out1.size(2)) # Flatten back
        
#         x = F.relu(out1)
#         x = F.dropout(x, p=0.1, training=self.training)

#         # --- Layer 2 ---
#         out2 = self.conv2(x, edge_index)
#         g2 = self.gate2(out2)
        
#         out2 = out2.view(-1, self.heads, out2.size(-1) // self.heads)
#         out2 = out2 * g2.unsqueeze(-1)
#         out2 = out2.view(-1, out2.size(1) * out2.size(2))
        
#         x = F.relu(out2)
#         x = F.dropout(x, p=0.1, training=self.training)

#         # Output projection
#         return self.out_proj(x)



class GATv2_ResNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        # --- Layers ---
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
        self.conv4 = GATv2Conv(hidden_channels, out_channels, heads=4, concat=False, dropout=dropout)

        # --- LayerNorms ---
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)

        # --- Input projection for residual (important!) ---
        self.input_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x, edge_index):
        # ---- Layer 1 ----
        residual = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 2 ----
        residual = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 3 ----
        residual = x
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = x + residual
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Output ----
        x = self.conv4(x, edge_index)
        return xs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GaAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        H = hidden_channels

        # ----- Layer 1 -----
        self.conv1 = GATv2Conv(
            in_channels, H,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.gate1 = nn.Sequential(
            nn.Linear(H * heads, heads),
            nn.Sigmoid()
        )
        self.proj1 = nn.Linear(H * heads, H)

        # ----- Layer 2 -----
        self.conv2 = GATv2Conv(
            H, H,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.gate2 = nn.Sequential(
            nn.Linear(H * heads, heads),
            nn.Sigmoid()
        )
        self.proj2 = nn.Linear(H * heads, H)

        # ----- Layer 3 -----
        self.conv3 = GATv2Conv(
            H, H,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.gate3 = nn.Sequential(
            nn.Linear(H * heads, heads),
            nn.Sigmoid()
        )
        self.proj3 = nn.Linear(H * heads, H)

        # ----- Layer 4 (output) -----
        self.conv4 = GATv2Conv(
            H, out_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )

    def _apply_head_gating(self, out, gate):
        # out: [N, H * heads]
        # gate: [N, heads]
        N = out.size(0)
        head_dim = out.size(1) // self.heads

        out = out.view(N, self.heads, head_dim)       # [N, heads, H]
        out = out * gate.unsqueeze(-1)                # [N, heads, H]
        out = out.reshape(N, self.heads * head_dim)   # [N, H * heads]
        return out

    def forward(self, x, edge_index):
        # ---- Layer 1 ----
        out = self.conv1(x, edge_index)
        gate = self.gate1(out)
        out = self._apply_head_gating(out, gate)
        x = self.proj1(out)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 2 ----
        out = self.conv2(x, edge_index)
        gate = self.gate2(out)
        out = self._apply_head_gating(out, gate)
        x = self.proj2(out)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 3 ----
        out = self.conv3(x, edge_index)
        gate = self.gate3(out)
        out = self._apply_head_gating(out, gate)
        x = self.proj3(out)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Output ----
        x = self.conv4(x, edge_index)
        return x