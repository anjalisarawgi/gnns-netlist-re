import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "graphs/raw/mips_16_latest/nangate/mips_16_core_top_gephi.gml"

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
        

    # if "top+us00_round2" in parition_cleaned:
    #     G.nodes[node]['us00_round2'] = 1
    # else:
    #     G.nodes[node]['us00_round2'] = 0
    
    # if "top+u0+inst4" in parition_cleaned:
    #     G.nodes[node]['u0_inst4'] = 1
    # else:
    #     G.nodes[node]['u0_inst4'] = 0





    
    # top - aes

    subcircuit_map = {
        "@top": 0,
        "top+us_round2": 1,
        "top+_+inst4": 2,
        "top+us_": 3,
        "top+u0": 4,
        "top+u0+u_": 5
    }

    def assign_subcircuit(p):
        if p == "top+u0":
            return 4
        elif p.startswith("top+u0+u"):
            return 5
        elif "+us" in p and p.endswith("round2"):
            return 1
        elif "+us" in p and not p.endswith("round2"):
            return 3
        elif "inst" in p:
            return 2
        elif "@top" in p:
            return 0
        return -1

    # def assign_subcircuit(p):
    #     if p == "@top":
    #         return 0
    #     elif "inst" in p:
    #         return 1
    #     elif "top+u" in p:
    #         return 2
    #     else:
    #         return -1

    def assign_subcircuit(p):
        if p == "@top" or p=="top" :
            return 0 
        elif "IF_stage" in p :
            return 1 
        elif "MEM_stage" in p:
            return 2 
        elif "EX_stage" in p:
            return 3
        elif "ID_stage" in p:
            return 3
        elif "hazard_detection" in p:
            return 4
        elif "register_file" in p:
            return 5 
        else:
            return -1
        

        
        

    G.nodes[node]['subcircuit'] = assign_subcircuit(parition_cleaned)


        # # key expand - aes
    # G.nodes[node]['subcircuit'] = -1
    # if "@top" in parition_cleaned:
    #     G.nodes[node]["subcircuit"] = 0
    # elif "inst" in parition_cleaned:
    #     G.nodes[node]["subcircuit"] = 2
    # elif "top+u" in parition_cleaned: 
    #     G.nodes[node]['subcircuit'] = 3


    # G.nodes[node]['subcircuit'] = -1
    # if "@top" in parition_cleaned:
    #     G.nodes[node]["subcircuit"] = 0
    # elif "+us"  in parition_cleaned and parition_cleaned.endswith("round2"):
    #     G.nodes[node]["subcircuit"] = 1
    # elif "inst" in parition_cleaned:
    #     G.nodes[node]["subcircuit"] = 2
    # elif "+us" in parition_cleaned and not parition_cleaned.endswith("round2"):
    #     G.nodes[node]["subcircuit"] = 3
    # elif parition_cleaned == "top+u0":
    #     G.nodes[node]['subcircuit'] = 4
    # elif parition_cleaned.startswith("top+u0+u"):
    #     G.nodes[node]['subcircuit'] = 5



output_path = "mips_16_core_top_gephi_test.gml"
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


#### for testing 
import os
import json
from collections import defaultdict

subcircuit_summary = defaultdict(set)

for node in G.nodes():
    partition = G.nodes[node].get("partition", "UNKNOWN").strip("'")
    subcircuit = G.nodes[node].get("subcircuit", -1)
    subcircuit_summary[subcircuit].add(partition)

subcircuit_summary = {k: sorted(list(v)) for k, v in sorted(subcircuit_summary.items())}
gml_basename = os.path.splitext(os.path.basename(gml_path))[0]
output_dir = os.path.join("results", gml_basename)
os.makedirs(output_dir, exist_ok=True)

json_path = os.path.join(output_dir, "test.json")
with open(json_path, "w") as f:
    json.dump(subcircuit_summary, f, indent=2)

print(f"Saved subcircuit mapping to '{json_path}'")