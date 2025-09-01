import os
import networkx as nx
import csv

# Load graph
gml_path = "mwe/aes_key_expand_128_gephi.gml"
G = nx.read_gml(gml_path)
G = G.to_directed()

# Known gate types (add DFFPOS here)
gate_types = ["INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR", "INV", "BUF", "DFFPOS"]
gate2idx = {gate: idx for idx, gate in enumerate(gate_types)}

unknown_gates = []

def extract_gate_type(label):
    base = label.strip("'").split("_")[0]
    for gate in gate_types:
        if base.startswith(gate):
            return gate
    return "UNKNOWN"

# Process nodes
for node in G.nodes():
    label = G.nodes[node].get("label", "")
    gate_type = extract_gate_type(label)

    onehot = [0] * len(gate_types)
    if gate_type in gate2idx:
        onehot[gate2idx[gate_type]] = 1
    else:
        base = label.strip("'").split("_")[0]
        unknown_gates.append(base)
        print(f"Unknown gate type for label {label}")

    indeg = G.in_degree(node)
    outdeg = G.out_degree(node)

    G.nodes[node]['features'] = onehot + [indeg, outdeg]

# Save the modified graph
nx.write_gml(G, "aes_key_expand_128_gephi_wfeatures.gml")

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