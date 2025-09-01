import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

def make_features(gml_path):
    # Load GML graph using NetworkX
    G_nx = nx.read_gml(gml_path)

    # Convert to directed if needed
    if not G_nx.is_directed():
        G_nx = G_nx.to_directed()

    # Create a list of nodes with their numeric ID
    node_id_list = []
    for node in G_nx.nodes(data=True):
        node_id = int(node[0]) if isinstance(node[0], (int, str)) and str(node[0]).isdigit() else None
        if node_id is not None:
            node_id_list.append(node[0])
        else:
            print(f"Skipping node with non-integer ID: {node[0]}")

    # Sort node IDs for consistent indexing
    node_id_list = sorted(node_id_list, key=lambda x: int(x))

    # Map node ID → index (0, 1, 2, ...)
    id_map = {nid: idx for idx, nid in enumerate(node_id_list)}

    # Create feature matrix (zero vector for now)
    num_nodes = len(id_map)
    x = torch.zeros((num_nodes, 1), dtype=torch.float)

    # Build edge_index using remapped indices
    edge_index = []
    for src, dst in G_nx.edges():
        if src in id_map and dst in id_map:
            edge_index.append([id_map[src], id_map[dst]])
        else:
            print(f"Skipping edge with unknown nodes: {src} -> {dst}")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Return PyG Data object
    data = Data(x=x, edge_index=edge_index)
    return data


upadted_gml = make_features(gml_path)
print(upadted_gml)