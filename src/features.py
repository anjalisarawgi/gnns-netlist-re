import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np
import csv
from collections import defaultdict


input_gml = "graphs/raw/aes_encryption_latest/osu035/aes_key_expand_128_gephi.gml" #aes_cipher_top_gephi aes_key_expand_128_gephi
output_gml = "graphs/processed/aes_encryption_latest/osu035/aes_key_expand_128_gephi_test8.gml"


# input_gml = "graphs/synthetic/raw/aes_encryption_latest/osu035/2/aes_cipher_top_gephi_mixedError_2.gml"
# output_gml = "graphs/synthetic/processed/aes_encryption_latest/osu035/2/aes_cipher_top_gephi_mixedError_2.gml"



G = nx.read_gml(input_gml)


# should we make it directed?
G = G.to_directed()
# should we create an encoding just for the type?

gate_types = [ "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR",  "NAND", "NOR", "INV", "BUF", "AOI", "OAI", "DFF", "MUX"] # not sure if we should handle dff like this 
# gate_types = [ "XOR", "XNOR", "AND", "OR",  "NAND", "NOR", "INV", "BUF", "AOI", "OAI", "DFF", "MUX"] 
gate2idx = {gate: idx for idx, gate in enumerate(gate_types)}
unknown_gates = []

def extract_gate_type(label):
    base = label.strip("'").split("_")[0]
    for gate in gate_types:
        if gate in base:
            return gate
    return "UNKNOWN"

log_lines = []

# # # top - aes ##
# def assign_subcircuit(p):
#     if p == "top+u0":
#         return 4
#     elif p.startswith("top+u0+u"):
#         return 5
#     elif "+us" in p and p.endswith("round2"):
#         return 1
#     elif "+us" in p and not p.endswith("round2"):
#         return 3
#     elif "inst" in p:
#         return 2
#     elif "@top" in p:
#         return 0
#     return -1


# def assign_subcircuit_name(p):
#     if p == "@top" or p=="top":
#         return "top"
#     elif p == "@top+u0" or p=="top+u0":
#         return "key_expand"
#     elif "inst" in p or "top+u0+" in p or "round2" in p or "top+us" in p :
#         return "sbox"
#     return -1


################################

# key expand - aes ##
def assign_subcircuit(p):
    if p == "@top" or  p=="top":
        return 0
    # elif p == "top+inst4":
    #     return 2
    elif "top+u" in p or "inst" in p :
        return 1
    return -1



def assign_subcircuit_name(p):
    if p == "@top" or p=="top":
        return "key_expand"
    # elif p == "top+inst4":
    #     return "rcon"
    elif "top+u" or "inst" in p :
        return "sbox"
    return -1

################################

# des ##
# def assign_subcircuit(p):
#     if p =="@top":
#         return 0 
#     elif p=="top":
#         return 1
#     elif p=="@top+u0":
#         return 2
#     elif p=="top+u0":
#         return 3
#     elif p=="top+u1":
#         return 4
#     elif "top+u0+" in p:
#         return 5
#     return -1 


# def assign_subcircuit_name(p):
#     if p == "@des" or p=="des":
#         return "des"
#     elif p == "@top+u0" or p=="top+u0":
#         return "crp"
#     elif p=="top+u1":
#         return "key_selh"
#     elif "top+u0+" in p:
#         return "sbox"
#     return -1

clustering = nx.clustering(G.to_undirected())
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
    # G.nodes[node]['features'] = [is_pi, is_po, is_key] + onehot + [indeg, outdeg]

    neighbor_gate_onehot = [0] * len(gate_types)

    for neighbor in G.predecessors(node):
        neighbor_label = G.nodes[neighbor].get("label", neighbor).strip("'")
        neighbor_gate = extract_gate_type(neighbor_label)
        if neighbor_gate in gate2idx:
            neighbor_gate_onehot[gate2idx[neighbor_gate]] += 1

    for neighbor in G.successors(node):
        neighbor_label = G.nodes[neighbor].get("label", neighbor).strip("'")
        neighbor_gate = extract_gate_type(neighbor_label)
        if neighbor_gate in gate2idx:
            neighbor_gate_onehot[gate2idx[neighbor_gate]] += 1


    # logic_cone_size = len(nx.ancestors(G, node))
    # transitive_fanout = len(nx.descendants(G, node))
    clustering_coeff = clustering.get(node, 0)

    # avg neighbor degrees
    neighbor_degrees = [G.degree(n) for n in G.predecessors(node)] + [G.degree(n) for n in G.successors(node)]
    avg_neighbor_degree = round(np.mean(neighbor_degrees), 3) if neighbor_degrees else 0

    # local reach - how big is the area where a node can reach in 2 hops
    neighbors = set(G.predecessors(node)) | set(G.successors(node))
    neighbors_2hop = set()
    for n in neighbors:
        neighbors_2hop.update(G.predecessors(n))
        neighbors_2hop.update(G.successors(n))
    local_reach = len(neighbors_2hop)



    gate_type_combined = [onehot[i] + neighbor_gate_onehot[i] for i in range(len(gate_types))] ### combine both 
    # G.nodes[node]['features'] = [indeg, outdeg, avg_neighbor_degree, local_reach] +  gate_type_combined #onehot + neighbor_gate_onehot
    G.nodes[node]['features'] = [indeg, outdeg, avg_neighbor_degree] + gate_type_combined

    # fan_io_ratio = indeg / (outdeg + 1e-5)  # avoid divide by zero
    
    # # clustering_coeff = clustering.get(node, 0)

    # # # Combine all into final features
    # G.nodes[node]['features'] = (
    #     # [is_pi, is_po, is_key] +
    #     onehot +
    #     [indeg, outdeg]
    #     #  +
    #     # [fan_io_ratio, logic_cone_size, transitive_fanout]
    # )

    if "INPUT" in clean_label:
        G.nodes[node]['is_IO'] = 1
    elif "OUTPUT" in clean_label:
        G.nodes[node]['is_IO'] = 2
    else:
        G.nodes[node]['is_IO'] = 0
        G.nodes[node]['subcircuit_id'] = parition_cleaned
    # else:
    #     G.nodes[node]['subcircuit_id'] = parition_cleaned
        
    G.nodes[node]['subcircuit'] = assign_subcircuit(parition_cleaned) # subciruit_id -
    G.nodes[node]['subcircuit_name'] = assign_subcircuit_name(parition_cleaned) # subciruit_id -


    

partition_set = sorted(set(G.nodes[node].get("partition", "UNKNOWN").strip("'") for node in G.nodes()))
partition2id = {part: idx for idx, part in enumerate(partition_set)}

# Store original subcircuit label ID
for node in G.nodes():
    raw_partition = G.nodes[node].get("partition", "UNKNOWN")
    cleaned_partition = raw_partition.strip("'")
    G.nodes[node]['subcircuit_original'] = partition2id[cleaned_partition]



output_dir = os.path.dirname(output_gml)
os.makedirs(output_dir, exist_ok=True)
nx.write_gml(G, output_gml)
print(f"Saved modified GML to '{output_gml}'")

if log_lines:
    log_path = os.path.join(os.path.dirname(output_gml), "unknown_gate_types.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"Saved detailed logs to {log_path}")
    

if unknown_gates:
    unknown_counts = {}
    for gate in unknown_gates:
        unknown_counts[gate] = unknown_counts.get(gate, 0) + 1

    log_path = os.path.join(os.path.dirname(output_gml), "unknown_gate_types.csv")
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gate_type", "count"])
        for gate, count in sorted(unknown_counts.items(), key=lambda x: -x[1]):
            writer.writerow([gate, count])
    print(f"\nSaved {len(unknown_counts)} unknown gate types to 'unknown_gate_types.csv'")
else:
    print("No unknown gate types found.")

partition_map_path = os.path.join(os.path.dirname(output_gml), "partition2id.json")
with open(partition_map_path, "w") as f:
    json.dump(partition2id, f, indent=2)

print(f"Saved partition-to-ID mapping to '{partition_map_path}'")


# Build subcircuit map from partition names
subcircuit_map = defaultdict(list)

for node in G.nodes():
    raw_partition = G.nodes[node].get("partition", "UNKNOWN")
    cleaned_partition = raw_partition.strip("'")
    subcircuit_id = assign_subcircuit(cleaned_partition)
    subcircuit_map[subcircuit_id].append(cleaned_partition)

# Remove duplicates and sort partition names
for k in subcircuit_map:
    subcircuit_map[k] = sorted(list(set(subcircuit_map[k])))

# Save to JSON
subcircuit_map_path = os.path.join(os.path.dirname(output_gml), "subcircuit_map.json")
with open(subcircuit_map_path, "w") as f:
    json.dump(subcircuit_map, f, indent=2)

print(f"Saved subcircuit-to-partition mapping to '{subcircuit_map_path}'")




    # def assign_subcircuit(p):
    #     if p == "@top":
    #         return 0
    #     elif "inst" in p:
    #         return 1
    #     elif "top+u" in p:
    #         return 2
    #     else:
    #         return -1

    # def assign_subcircuit(p):
    #     if p == "@top" or p=="top" :
    #         return 0 
    #     elif "IF_stage" in p :
    #         return 1 
    #     elif "MEM_stage" in p:
    #         return 2 
    #     elif "EX_stage" in p:
    #         return 3
    #     elif "ID_stage" in p:
    #         return 3
    #     elif "hazard_detection" in p:
    #         return 4
    #     elif "register_file" in p:
    #         return 5 
    #     else:
    #         return -1