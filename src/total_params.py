import torch
import sys
sys.path.append("src")

from gnn.sage_bidirected import BiDirectedGraphSAGE  # use the correct class name
from gnn.graphSAGE import graphSAGE
# model = BiDirectedGraphSAGE(in_channels=31, hidden_channels=256, out_channels=2)
# state = torch.load("models/gnns/51lodo_31f_BiDirectedGraphSAGE_ce_weighted_fullgraph_per_design_tiny_aes/model.pt", map_location="cpu")

model = graphSAGE(in_channels=42, hidden_channels=256, out_channels=2)
state = torch.load("models/gnns/graphsage_ce_fullgraph_per_design_aes_core/model.pt", map_location="cpu")
fixed_state = {}
for k, v in state.items():
    if k.startswith("_orig_mod."):
        fixed_state[k.replace("_orig_mod.", "")] = v
    else:
        fixed_state[k] = v

model.load_state_dict(fixed_state)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")