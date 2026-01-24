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

    clustering = nx.clustering(G.to_undirected())
    G_undirected = G.to_undirected()

    # =========================================================
    # GRAPH-LEVEL STATISTICS (computed once per graph)
    # =========================================================
    degrees = np.array([G.degree(n) for n in G.nodes()])
    avg_graph_degree = float(degrees.mean()) if len(degrees) > 0 else 1.0

    # ================= GRAPH-LEVEL FEATURES =================
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    deg_std = float(degrees.std()) if len(degrees) > 0 else 0.0
    avg_clustering_graph = float(nx.average_clustering(G_undirected))

    num_io = sum(
        1 for _, d in G.nodes(data=True)
        if is_io_label(d.get("label", ""))
    )
    frac_io = num_io / num_nodes if num_nodes > 0 else 0.0

    graph_features = [
        np.log1p(num_nodes),     # graph size
        avg_graph_degree,        # density
        deg_std,                 # degree variance
        avg_clustering_graph,    # structure
        frac_io                  # IO fraction
    ]

    # store once per graph
    G.graph["graph_features"] = graph_features
    # ========================================================


    # Betweenness centrality (approximate for scalability)
    n = G_undirected.number_of_nodes()
    if n <= 1:
        betweenness = {node: 0.0 for node in G.nodes()}
    else:
        k = min(100, n)
        betweenness = nx.betweenness_centrality(
            G_undirected, normalized=True, k=k
        )


    avg_betweenness = (
        float(np.mean(list(betweenness.values())))
        if len(betweenness) > 0 else 1.0
    )

    # ---------------------------------------------------------
    # Define dense core: top 10% nodes by degree
    # ---------------------------------------------------------
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    k_core = max(1, int(0.1 * len(sorted_nodes)))
    core_nodes = set(sorted_nodes[:k_core])


    # IO nodes
    io_nodes = [
        n for n, d in G.nodes(data=True)
        if is_io_label(d.get("label", ""))
    ]

    dist_to_core = {}
    if core_nodes:
        super_core = "__CORE__"
        G_undirected.add_node(super_core)
        for c in core_nodes:
            G_undirected.add_edge(super_core, c)

        dist_to_core = nx.single_source_shortest_path_length(
            G_undirected, super_core
        )

        G_undirected.remove_node(super_core)


    min_io_distances = {}
    if io_nodes:
        super_io = "__IO__"
        G_undirected.add_node(super_io)
        for io in io_nodes:
            G_undirected.add_edge(super_io, io)

        min_io_distances = nx.single_source_shortest_path_length(
            G_undirected, super_io
        )

        G_undirected.remove_node(super_io)
        
    # Min distance to IO
    for node in G.nodes():
        raw_label = G.nodes[node].get("label", node)
        raw_partition = G.nodes[node].get("partition", node)

        label = str(raw_label).strip("'")
        partition_cleaned = str(raw_partition).strip("'")

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        # ---------------------------------------------------------
        # Graph-normalized features (relative importance)
        # ---------------------------------------------------------
        norm_indeg = indeg / (avg_graph_degree + 1e-6)
        norm_outdeg = outdeg / (avg_graph_degree + 1e-6)
        norm_betweenness = betweenness.get(node, 0.0) / (avg_betweenness + 1e-6)

        clustering_coeff = float(clustering.get(node, 0.0))

        nbrs = set(G.predecessors(node)) | set(G.successors(node))

        # ------------------------
        # Neighborhood statistics
        # ------------------------
        if nbrs:
            nbr_degs = [G.degree(n) for n in nbrs]
            avg_neighbor_degree = float(np.mean(nbr_degs))
            # ---------------------------------------------------------
            # Density contrast (boundary gradient cue)
            # ---------------------------------------------------------
            density_contrast = G.degree(node) - avg_neighbor_degree
            norm_density_contrast = density_contrast / (avg_neighbor_degree + 1e-6)

            is_connected_to_io = float(
                any(is_io_label(G.nodes[n].get("label", "")) for n in nbrs)
            )

            # NEW: neighbor degree entropy
            vals, counts = np.unique(nbr_degs, return_counts=True)
            probs = counts / counts.sum()
            neighbor_degree_entropy = float(
                -np.sum(probs * np.log(probs + 1e-8))
            )

            core_distance = dist_to_core.get(node, -1)

        else:
            avg_neighbor_degree = 0.0
            is_connected_to_io = 0.0
            neighbor_degree_entropy = 0.0

        # ------------------------
        # NEW: two-hop boundary exposure
        # ------------------------
        two_hop = set()
        for n in nbrs:
            two_hop |= set(G.predecessors(n)) | set(G.successors(n))
        two_hop.discard(node)

        # ------------------------
        # NEW: flow / cut features
        # ------------------------
        deg_imbalance = abs(indeg - outdeg)
        flow_asym = (indeg - outdeg) / (indeg + outdeg + 1)
        cut_proxy = indeg * outdeg

        
        # ------------------------
        # Final feature vector
        # ------------------------
        # gf = G.graph["graph_features"]
        G.nodes[node]["features"] = [
            float(indeg),
            float(outdeg),
            float(norm_indeg),              # NEW
            float(norm_outdeg),             # NEW
            float(deg_imbalance),
            float(avg_neighbor_degree),
            float(norm_density_contrast),   # NEW
            float(clustering_coeff),
            float(cut_proxy),
            float(flow_asym),
            float(neighbor_degree_entropy),
            float(is_connected_to_io),
            float(betweenness.get(node, 0.0)),
            float(norm_betweenness),        # NEW
            float(min_io_distances.get(node, -1)),
            float(core_distance),           # NEW
        ] # + gf

        ############ 
        # jan 18
        ############
        # G.nodes[node]["features"] = [
        #     float(indeg),
        #     float(outdeg),
        #     float(deg_imbalance),           # NEW
        #     float(avg_neighbor_degree),
        #     float(clustering_coeff),
        #     float(cut_proxy),               # NEW
        #     float(flow_asym),               # NEW
        #     float(neighbor_degree_entropy), # NEW
        #     float(is_connected_to_io),
        #     float(betweenness.get(node, 0.0)),
        #     float(min_io_distances.get(node, -1)),
            
        # ]

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

# ROOT_RAW = "new_graphs_crypto/raw/raw"
# ROOT_OUT = "new_graphs_crypto/processed_jan23_w_graphFeatures"
ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_jan23_w_graphFeatures"

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
            
            # print("Feature dim per node:", len(G.nodes[list(G.nodes())[0]]["features"]))
            G_out = nx.read_gml(output_gml)
            any_node = next(iter(G_out.nodes()))
            print("Feature dim per node:", len(G_out.nodes[any_node]["features"]))

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