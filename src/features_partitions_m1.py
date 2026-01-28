import os
import json
import numpy as np
import networkx as nx

############################################################
# STEP 1: Helper checks
############################################################

def is_graph_connected(G):
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)

def has_boundary_labels(G):
    for _, data in G.nodes(data=True):
        if int(data.get("boundary", 0)) == 1:
            return True
    return False

############################################################
# STEP 2: Single GML processing (M1)
############################################################

def process_single_gml(input_gml, output_gml):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml).to_directed()
    G.remove_edges_from(nx.selfloop_edges(G))

    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    for node in G.nodes():
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        deg = G.degree(node)

        nbrs = set(G.predecessors(node)) | set(G.successors(node))

        # ------------------------
        # Local neighborhood stats
        # ------------------------
        if nbrs:
            nbr_degs = np.array([G.degree(n) for n in nbrs], dtype=np.float32)
            avg_neighbor_degree = float(nbr_degs.mean())

            deg_minus_nbr_mean = float(deg - avg_neighbor_degree)
            deg_over_nbr_mean  = float(deg / (avg_neighbor_degree + 1e-6))
            norm_density_contrast = float(
                (deg - avg_neighbor_degree) / (avg_neighbor_degree + 1e-6)
            )

            H = G_undirected.subgraph(list(nbrs))
            m = H.number_of_edges()
            k = len(nbrs)
            ego_edges_frac = float(m / (k * (k - 1) / 2)) if k > 1 else 0.0
        else:
            avg_neighbor_degree = 0.0
            deg_minus_nbr_mean = 0.0
            deg_over_nbr_mean  = 0.0
            norm_density_contrast = 0.0
            ego_edges_frac = 0.0

        # ------------------------
        # Directed flow features
        # ------------------------
        deg_imbalance = abs(indeg - outdeg)
        flow_asym = (indeg - outdeg) / (indeg + outdeg + 1.0)

        is_source = float(indeg == 0)
        is_sink   = float(outdeg == 0)

        clustering_coeff = float(clustering.get(node, 0.0))

        # ------------------------
        # FINAL M1 FEATURE VECTOR
        # ------------------------
        G.nodes[node]["features"] = [
            float(indeg),
            float(outdeg),
            float(np.log1p(indeg)),
            float(np.log1p(outdeg)),

            float(indeg / (outdeg + 1e-6)),  # in/out ratio
            float(deg_imbalance),
            float(flow_asym),

            float(is_source),
            float(is_sink),

            float(avg_neighbor_degree),
            float(deg_minus_nbr_mean),
            float(deg_over_nbr_mean),
            float(norm_density_contrast),

            float(clustering_coeff),
            float(ego_edges_frac),
        ]

    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G, output_gml)
    print(f"[INFO] Saved processed GML → {output_gml}")

############################################################
# STEP 3: Batch processing
############################################################

# ROOT_RAW = "new_graphs_crypto/raw/raw"
# ROOT_OUT = "new_graphs_crypto/processed_m1/"

ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_jan27_m1"


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
                print(f"[SKIP] Not connected → {input_gml}")
                unusable_graphs_connectivity.append(input_gml)
                continue

            if not has_boundary_labels(G):
                print(f"[SKIP] No boundary → {input_gml}")
                unusable_graphs_no_boundary.append(input_gml)
                continue

            try:
                process_single_gml(input_gml, output_gml)
                usable_graphs.append(output_gml)

                G_out = nx.read_gml(output_gml)
                any_node = next(iter(G_out.nodes()))
                print("Feature dim:", len(G_out.nodes[any_node]["features"]))

            except Exception as e:
                print(f"[CRASH] {input_gml}: {e}")

############################################################
# STEP 4: Save metadata
############################################################

os.makedirs(ROOT_OUT, exist_ok=True)

with open(os.path.join(ROOT_OUT, "usable_graphs.json"), "w") as f:
    json.dump(usable_graphs, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_connectivity.json"), "w") as f:
    json.dump(unusable_graphs_connectivity, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_no_boundary.json"), "w") as f:
    json.dump(unusable_graphs_no_boundary, f, indent=4)

print("\n=== SUMMARY ===")
print("Usable graphs                 :", len(usable_graphs))
print("Unusable (not connected)      :", len(unusable_graphs_connectivity))
print("Unusable (no boundary labels) :", len(unusable_graphs_no_boundary))