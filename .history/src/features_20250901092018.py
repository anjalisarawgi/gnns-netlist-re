import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

def make_features(gml_path):
    G_nx = nx.read_gml(gml_path)

    # Ensure it's directed
    if not G_nx.is_directed():
        G_nx = G_nx.to_directed()

    # Extract all node labels
    label_to_index = {}
    node_labels = []
    for i, (node_id, attrs) in enumerate(G_nx.nodes(data=True)):
        label = attrs.get("label", None)
        if label is None:
            continue
        label = label.strip("'")  # remove quotes around label
        label_to_index[label] = i
        node_labels.append(label)

    # Feature matrix: 1 zero feature per node
    num_nodes = len(label_to_index)
    x = torch.zeros((num_nodes, 1), dtype=torch.float)

    # Build edge index using labels
    edge_index = []
    for src, dst in G_nx.edges():
        src_label = G_nx.nodes[src].get("label", "").strip("'")
        dst_label = G_nx.nodes[dst].get("label", "").strip("'")

        if src_label in label_to_index and dst_label in label_to_index:
            edge_index.append([label_to_index[src_label], label_to_index[dst_label]])
        else:
            print(f"Skipping edge with unknown nodes: {src_label} -> {dst_label}")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data

upadted_gml = make_features(gml_path)
print(upadted_gml)