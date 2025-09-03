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

gate_types = [ "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR",  "NAND", "NOR", "INV", "BUF", "AOI", "OAI", "DFF", "MUX"] # not sure if we should handle dff like this 
gate2idx = {gate: idx for idx, gate in enumerate(gate_types)}
unknown_gates = []

def extract_gate_type(label):
    base = label.strip("'").split("_")[0]
    for gate in gate_types:
        if gate in base:
            return gate
    return "UNKNOWN"

log_lines = []

for node in G.nodes():
    raw_label = G.nodes[node].get("label", node)
    raw_partition = G.nodes[node].get("partition", node)
    parition_cleaned = raw_partition.strip("'")
    label = raw_label.strip("'")  # remove outer single quotes
    gate_type = extract_gate_type(label)

    onehot = [0] * len(gate_types)
    if gate_type in gate2idx:
        onehot[gate2idx[gate_type]] = 1
    else:
        base = label.strip("'").split("_")[0]
        unknown_gates.append(base)
        log_lines.append(f"Unknown gate type for label {label}")

    indeg = G.in_degree(node)
    outdeg = G.out_degree(node)


    # PI, PO, KEY
    is_pi = int(indeg == 0 )
    is_po = int(outdeg == 0 )

    clean_label = label.strip("'")
    print("clean_label", clean_label)
    is_key = int("key" in clean_label.lower())
    # print(f"Node: {node}, Label: {label}, is_key: {is_key}")
    G.nodes[node]['features'] = [is_pi, is_po, is_key] + onehot + [indeg, outdeg]
    if "INPUT" in clean_label:
        G.nodes[node]['is_IO'] = "INPUT"
    elif "OUTPUT" in clean_label:
        G.nodes[node]['is_IO'] = "OUTPUT"
    else:
        G.nodes[node]['subcircuit_id'] = parition_cleaned
        
    if "top+us00_round2" in parition_cleaned:
        G.nodes[node]['us00_round2'] = 1
    else:
        G.nodes[node]['us00_round2'] = 0
    
    if "top+u0+inst4" in parition_cleaned:
        G.nodes[node]['u0_inst4'] = 1
    else:
        G.nodes[node]['u0_inst4'] = 0



    G.nodes[node]['subcircuit'] = -1
    if "@top" in parition_cleaned:
        G.nodes[node]["subcircuit"] = 0
    elif "+us"  in parition_cleaned and parition_cleaned.endswith("round2"):
        G.nodes[node]["subcircuit"] = 1
    elif "inst" in parition_cleaned:
        G.nodes[node]["subcircuit"] = 2
    elif "+us" in parition_cleaned and not parition_cleaned.endswith("round2"):
        G.nodes[node]["subcircuit"] = 3
    elif parition_cleaned == "top+u0":
        G.nodes[node]['subcircuit'] = 4
    elif parition_cleaned.startswith("top+u0+u"):
        G.nodes[node]['subcircuit'] = 5




output_path = "aes_key_expand_features.gml"
nx.write_gml(G, output_path)
print(f"Saved modified GML to '{output_path}'")

# Save log file
if log_lines:
    with open("unknown_gate_types.log", "w") as f:
        f.write("\n".join(log_lines))
    print(f"Saved detailed log to 'unknown_gate_types.log'")
    
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
else:
    print("No unknown gate types found.")


