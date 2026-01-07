import os
import json
import numpy as np
import networkx as nx

############################################################
# STEP 1: Helper checks
############################################################

def is_graph_connected(G):
    """Return True if the graph is fully connected."""
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)

def has_boundary_labels(G):
    """Return True if at least one node has boundary == 1."""
    for _, data in G.nodes(data=True):
        if int(data.get("boundary", 0)) == 1:
            return True
    return False

def is_io_label(label: str) -> bool:
    s = str(label).upper()
    return ("INPUT" in s) or ("OUTPUT" in s)

############################################################
# STEP 2: Single GML processing
############################################################

def process_single_gml(input_gml, output_gml):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml).to_directed()

    gate_types = [
        "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR",
        "INV", "BUF", "AOI", "OAI", "DFF", "MUX"
    ]
    gate2idx = {g: i for i, g in enumerate(gate_types)}

    def extract_gate_type(label):
        base = str(label).strip("'").split("_")[0].upper()
        for gate in gate_types:
            if gate in base:
                return gate
        return "UNKNOWN"

    clustering = nx.clustering(G.to_undirected())

    for node in G.nodes():
        raw_label = G.nodes[node].get("label", node)
        raw_partition = G.nodes[node].get("partition", node)

        label = str(raw_label).strip("'")
        partition_cleaned = str(raw_partition).strip("'")

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)

        boundary_flag = int(G.nodes[node].get("boundary", 0))
        clustering_coeff = float(clustering.get(node, 0.0))

        nbrs = set(G.predecessors(node)) | set(G.successors(node))

        if nbrs:
            avg_neighbor_degree = float(np.mean([G.degree(n) for n in nbrs]))
            boundary_nbrs = sum(
                int(G.nodes[n].get("boundary", 0)) for n in nbrs
            )
            frac_boundary_nbrs = boundary_nbrs / len(nbrs)
            is_connected_to_io = float(
                any(is_io_label(G.nodes[n].get("label", "")) for n in nbrs)
            )
        else:
            avg_neighbor_degree = 0.0
            frac_boundary_nbrs = 0.0
            is_connected_to_io = 0.0

        in_out_ratio = (indeg + 1) / (outdeg + 1)

        G.nodes[node]["features"] = [
            float(indeg),
            float(outdeg),
            avg_neighbor_degree,
            clustering_coeff,
            in_out_ratio,
            frac_boundary_nbrs,
            float(boundary_flag),
            is_connected_to_io,
        ]

        clean_label = label.upper()
        if "INPUT" in clean_label:
            G.nodes[node]["is_IO"] = 1
        elif "OUTPUT" in clean_label:
            G.nodes[node]["is_IO"] = 2
        else:
            G.nodes[node]["is_IO"] = 0
            G.nodes[node]["subcircuit_id"] = partition_cleaned

    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G, output_gml)
    print(f"[INFO] Saved processed GML to {output_gml}")

############################################################
# STEP 3: Batch processing
############################################################

ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_v3"

processed_dirs = {}
usable_graphs = []
unusable_graphs_connectivity = []
unusable_graphs_no_boundary = []

for design in os.listdir(ROOT_RAW):
    design_path = os.path.join(ROOT_RAW, design)
    if not os.path.isdir(design_path):
        continue

    processed_dirs[design] = {}

    for tech in os.listdir(design_path):
        tech_path = os.path.join(design_path, tech)
        if not os.path.isdir(tech_path):
            continue

        print(f"\n=== Processing design: {design} | tech: {tech} ===\n")

        out_dir = os.path.join(ROOT_OUT, design, tech)
        os.makedirs(out_dir, exist_ok=True)
        processed_dirs[design][tech] = out_dir

        for fname in os.listdir(tech_path):
            if not fname.endswith(".gml"):
                continue

            input_gml = os.path.join(tech_path, fname)
            output_gml = os.path.join(out_dir, fname)

            G = nx.read_gml(input_gml)

            if not is_graph_connected(G):
                print(f"[ERROR] Not connected → {input_gml}")
                unusable_graphs_connectivity.append(input_gml)
                continue

            if not has_boundary_labels(G):
                print(f"[ERROR] No boundary → {input_gml}")
                unusable_graphs_no_boundary.append(input_gml)
                continue

            try:
                process_single_gml(input_gml, output_gml)
                usable_graphs.append(output_gml)
            except Exception as e:
                print(f"[CRASH] {input_gml}: {e}")

############################################################
# STEP 4: Save metadata / validation results
############################################################

os.makedirs(ROOT_OUT, exist_ok=True)

with open(os.path.join(ROOT_OUT, "partitions_path.json"), "w") as f:
    json.dump(
        {k: list(v.values()) for k, v in processed_dirs.items()},
        f,
        indent=4
    )

with open(os.path.join(ROOT_OUT, "usable_graphs.json"), "w") as f:
    json.dump(usable_graphs, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_connectivity.json"), "w") as f:
    json.dump(unusable_graphs_connectivity, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_no_boundary.json"), "w") as f:
    json.dump(unusable_graphs_no_boundary, f, indent=4)

print("\n=== VALIDATION SUMMARY ===")
print("Usable graphs                 :", len(usable_graphs))
print("Unusable (not connected)      :", len(unusable_graphs_connectivity))
print("Unusable (no boundary labels) :", len(unusable_graphs_no_boundary))