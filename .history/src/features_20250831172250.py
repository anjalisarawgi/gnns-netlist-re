import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

def make_features(gml_path):
    G = nx.read_gml(gml_path)

    # leaving it as it is (convert to directed later if required!)
    if not G.is_directed():
        G_nx = G.to_directed() 

    node_ids = sorted(G.nodes(), key=int)
    num_nodes = len(node_ids)

    # feature matrix 
    x = torch.zeros((num_nodes, 1), dtype= torch.float)
    id_map = {int(nid): idx for idx, nid in enumerate(node_ids)} #  # Map node IDs to indices ( i dont understand this part)

    edge_index = []
    for src, dst in G_nx.edges():
        edge_index.append([id_map[int(src)], id_map[int(dst)]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    return data


upadted_gml = make_features(gml_path)
print(upadted_gml)