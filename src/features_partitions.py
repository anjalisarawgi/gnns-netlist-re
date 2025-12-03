import os
import json
import csv
import numpy as np
import networkx as nx
from collections import defaultdict

############################################################
# STEP 1: Put YOUR existing processing code into this function
############################################################
def is_graph_connected(G):
    """Return True if the graph is fully connected (weakly)."""
    return nx.is_weakly_connected(G)

def has_boundary_labels(G):
    """Return True if the graph contains at least one boundary label."""
    for _, data in G.nodes(data=True):
        if "boundary" in data:
            return True
    return False

def process_single_gml(input_gml, output_gml):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml)
    G = G.to_directed()

    gate_types = [
        "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR",
        "INV", "BUF", "AOI", "OAI", "DFF", "MUX"
    ]
    gate2idx = {g: i for i, g in enumerate(gate_types)}
    unknown_gates = []
    log_lines = []

    def extract_gate_type(label):
        base = label.strip("'").split("_")[0]
        for gate in gate_types:
            if gate in base:
                return gate
        return "UNKNOWN"

    clustering = nx.clustering(G.to_undirected())

    # === YOUR NODE FEATURE EXTRACTION CODE ===
    for node in G.nodes():
        raw_label = G.nodes[node].get("label", node)
        raw_partition = G.nodes[node].get("partition", node)

        parition_cleaned = raw_partition.strip("'")
        label = raw_label.strip("'")
        gate_type = extract_gate_type(label)

        onehot = [0] * len(gate_types)
        if gate_type in gate2idx:
            onehot[gate2idx[gate_type]] = 1
        else:
            unknown_gates.append(gate_type)

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)

        boundary_flag = int(G.nodes[node].get("boundary", 0))
        avg_neighbor_degree = 0
        clustering_coeff = clustering.get(node, 0)

        G.nodes[node]['features'] = [indeg, outdeg, avg_neighbor_degree, boundary_flag]

        clean_label = label.strip("'")
        if "INPUT" in clean_label:
            G.nodes[node]['is_IO'] = 1
        elif "OUTPUT" in clean_label:
            G.nodes[node]['is_IO'] = 2
        else:
            G.nodes[node]['is_IO'] = 0
            G.nodes[node]['subcircuit_id'] = parition_cleaned

    # === Save processed GML ===
    os.makedirs(os.path.dirname(output_gml), exist_ok=True)
    nx.write_gml(G, output_gml)

    print(f"Saved processed GML to {output_gml}")


############################################################
# STEP 2: Auto process ALL folders + save processed paths
############################################################

ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_v2"

# This dictionary will store all processed directories
processed_dirs = {}
usable_graphs = []
unusable_graphs_connectivity = []
unusable_graphs_no_boundary = []

for design in os.listdir(ROOT_RAW):
    design_path = os.path.join(ROOT_RAW, design)
    if not os.path.isdir(design_path):
        continue

    processed_dirs[design] = {}  # add entry for this design

    for tech in os.listdir(design_path):
        tech_path = os.path.join(design_path, tech)
        if not os.path.isdir(tech_path):
            continue

        print(f"\n=== Processing design: {design} | tech: {tech} ===\n")

        out_dir = os.path.join(ROOT_OUT, design, tech)
        os.makedirs(out_dir, exist_ok=True)

        # Save processed folder path
        processed_dirs[design][tech] = out_dir  

        gml_files = [f for f in os.listdir(tech_path) if f.endswith(".gml")]

        for fname in gml_files:
            input_gml = os.path.join(tech_path, fname)
            output_gml = os.path.join(out_dir, fname)

            # Load graph first (before processing)
            G = nx.read_gml(input_gml)

            # 1. Connectivity check
            if not is_graph_connected(G):
                print(f"[ERROR 1] Graph NOT connected → unusable (type 1): {input_gml}")
                unusable_graphs_connectivity.append(input_gml)
                continue

            # 2. Boundary presence check
            if not has_boundary_labels(G):
                print(f"[ERROR 2] Graph has NO boundary labels → unusable (type 2): {input_gml}")
                unusable_graphs_no_boundary.append(input_gml)
                continue

            # If passes all checks -> usable
            usable_graphs.append(output_gml)

            # Only process if graph is usable
            process_single_gml(input_gml, output_gml)

############################################################
# STEP 3: Save all processed directory paths to JSON
############################################################


partitions_json_path = "graphs/processed_v2/partitions_path.json"

# Convert dict-of-dicts → dict-of-lists
cleaned_processed_dirs = {
    design: list(tech_paths.values())
    for design, tech_paths in processed_dirs.items()
}

with open(partitions_json_path, "w") as f:
    json.dump(cleaned_processed_dirs, f, indent=4)

print(f"\nSaved all processed directory paths to {partitions_json_path}\n")

# Save validation results
with open("graphs/processed_v2/usable_graphs.json", "w") as f:
    json.dump(usable_graphs, f, indent=4)

with open("graphs/processed_v2/unusable_graphs_connectivity.json", "w") as f:
    json.dump(unusable_graphs_connectivity, f, indent=4)

with open("graphs/processed_v2/unusable_graphs_no_boundary.json", "w") as f:
    json.dump(unusable_graphs_no_boundary, f, indent=4)

print("\n=== CONNECTIVITY VALIDATION ===")
print("Usable graphs                    :", len(usable_graphs))
print("Unusable (not connected)         :", len(unusable_graphs_connectivity))
print("Unusable (no boundary labels)    :", len(unusable_graphs_no_boundary))