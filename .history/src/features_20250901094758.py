import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

G = nx.read_gml(gml_path)


# should we make it directed?
G = G.to_directed()
# should we create an encoding just for the type?

gate_types = ["INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR",  "NAND", "NOR", "INV", "BUF"]
gate2idx = {gate: idx for idx, gate in enumerate(gate_types)}
unknown_gates = []
def extract_gate_type(label):
    base = label.strip("'").split("_")[0]
    for gate in gate_types:
        if gate in base:
            return gate
    return "UNKNOWN"

for node in G.nodes():

    # label = G.nodes[node]["label"]
    label = G.nodes[node].get("label", "")
    gate_type = extract_gate_type(label)

    onehot = [0]*len(gate_types)
    if gate_type in gate2idx:
        onehot[gate2idx[gate_type]] = 1
    else:
        unknown_gates.append(label.strip("'").split("_")[0])
        print(f"Unknown gate type for label {label}")

    # in degree and out degree
    indeg = G.in_degree(node)
    outdeg = G.out_degree(node)
    # G.nodes[node]['features'] = [indeg, outdeg]

    # combine
    G.nodes[node]['features'] = onehot + [indeg, outdeg]
    

nx.write_gml(G, "aes_key_expand_128_gephi_wfeatures.gml")


import os 
import networkx as nx
import csv
# Save unknown gate types to CSV
if unknown_gates:
    unknown_counts = {}
    for gate in unknown_gates:
        unknown_counts[gate] = unknown_counts.get(gate, 0) + 1

    with open("unknown_gate_types.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gate_type", "count"])
        for gate, count in sorted(unknown_counts.items(), key=lambda x: -x[1]):
            writer.writerow([gate, count])

    print(f"\nSaved {len(unknown_counts)} unknown gate types to 'unknown_gate_types.csv'")