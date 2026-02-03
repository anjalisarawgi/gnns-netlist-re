import os
import json
import numpy as np
import networkx as nx
from collections import deque

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
# STEP 2: Small helpers for new features
############################################################

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def neighbors_undirected(G, node):
    # treat digraph as undirected neighborhood for local structure stats
    return set(G.predecessors(node)) | set(G.successors(node))

def two_hop_neighbors(G, node):
    """
    Returns the set of 2-hop neighbors (excluding the node itself).
    Uses undirected neighbor notion (pred ∪ succ).
    """
    one_hop = neighbors_undirected(G, node)
    two_hop = set()
    for n1 in one_hop:
        two_hop.update(neighbors_undirected(G, n1))
    two_hop.discard(node)
    return two_hop

############################################################
# STEP 3: Single GML processing (M1 + improved features)
############################################################

def process_single_gml(input_gml, output_gml):
    print("Processing:", input_gml)

    # keep IDs stable: your other scripts sometimes use label="id"
    # Here we read default, then force directed for consistent in/out
    G = nx.read_gml(input_gml).to_directed()
    G.remove_edges_from(nx.selfloop_edges(G))

    # For clustering + ego density we want undirected view
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)

    # Precompute degrees for speed
    deg_dict = dict(G.degree())
    indeg_dict = dict(G.in_degree())
    outdeg_dict = dict(G.out_degree())

    # If partition attribute exists, we can compute external exposure.
    # We’ll try a few common keys; if none exists, we fallback to 0.0.
    # (You can add/remove keys depending on your partition2gephi output.)
    partition_keys = ["partition", "cluster", "community", "part", "subcircuit", "subcircuit_id"]
    node_part = {}
    detected_key = None
    for k in partition_keys:
        # check if at least one node has it
        any_has = any(k in data for _, data in G.nodes(data=True))
        if any_has:
            detected_key = k
            break

    if detected_key is not None:
        for n, data in G.nodes(data=True):
            node_part[n] = data.get(detected_key, None)
        print(f"[INFO] Using partition key: '{detected_key}' for external-neighbor features")
    else:
        print("[INFO] No partition key found; external-neighbor features will be 0.0")


    # ------------------------
    # NEW: Bridge + SCC precomputation (cheap, global)
    # ------------------------
    try:
        bridge_edges = set(nx.bridges(G_undirected))
    except nx.NetworkXError:
        bridge_edges = set()

    # Directed SCCs
    sccs = list(nx.strongly_connected_components(G))
    scc_id = {}
    scc_sizes = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            scc_id[n] = i
        scc_sizes[i] = len(comp)

            
    for node in G.nodes():
        indeg = indeg_dict.get(node, 0)
        outdeg = outdeg_dict.get(node, 0)
        deg = deg_dict.get(node, 0)

        nbrs = neighbors_undirected(G, node)

        # ------------------------
        # Local neighborhood stats (1-hop)
        # ------------------------
        if nbrs:
            nbr_degs = np.array([deg_dict.get(n, 0) for n in nbrs], dtype=np.float32)
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

        # Simple “regime” flags (often very helpful)
        # thresholds are intentionally mild
        is_low_in_high_out = float(indeg <= 1 and outdeg >= 3)
        is_high_in_low_out = float(outdeg <= 1 and indeg >= 3)

        clustering_coeff = float(clustering.get(node, 0.0))

        # ------------------------
        # NEW: 2-hop neighborhood contrast
        # ------------------------
        two_hop = two_hop_neighbors(G, node)
        if two_hop:
            twohop_degs = np.array([deg_dict.get(n, 0) for n in two_hop], dtype=np.float32)
            twohop_avg_degree = float(twohop_degs.mean())
        else:
            twohop_avg_degree = 0.0

        hop_contrast = float(abs(avg_neighbor_degree - twohop_avg_degree))

        # ------------------------
        # NEW: External neighbor fraction (partition-aware)
        # ------------------------
        # ext_frac = 0.0
        # if detected_key is not None:
        #     p0 = node_part.get(node, None)
        #     if p0 is None:
        #         ext_frac = 0.0
        #     else:
        #         ext = 0
        #         for nb in nbrs:
        #             if node_part.get(nb, None) != p0:
        #                 ext += 1
        #         ext_frac = float(ext / (len(nbrs) + 1e-6))
        # else:
        #     ext_frac = 0.0

        # ------------------------
        # NEW: Bridge fraction (undirected)
        # ------------------------
        if nbrs:
            bridge_incident = 0
            for nb in nbrs:
                e = (node, nb)
                er = (nb, node)
                if e in bridge_edges or er in bridge_edges:
                    bridge_incident += 1
            bridge_frac = float(bridge_incident / (len(nbrs) + 1e-6))
        else:
            bridge_frac = 0.0

        # ------------------------
        # NEW: SCC features (directed)
        # ------------------------
        sid = scc_id.get(node, None)
        if sid is not None:
            scc_size = float(scc_sizes.get(sid, 1))
            is_trivial_scc = float(scc_size == 1)
        else:
            scc_size = 1.0
            is_trivial_scc = 1.0

        #### 
        ego = set(nbrs) | {node}

        cut_edges = 0
        seen = set()
        for u in ego:
            for v in neighbors_undirected(G, u):
                e = tuple(sorted((u, v)))
                if e in seen:
                    continue
                seen.add(e)
                if v not in ego:
                    cut_edges += 1

        ego_cut_ratio = float(cut_edges / (len(ego) + 1e-6))

        # ------------------------
        # NEW: Neighbor flow variance (heterogeneity proxy)
        # ------------------------
        if nbrs:
            nbr_flow = [
                indeg_dict.get(n, 0) - outdeg_dict.get(n, 0)
                for n in nbrs
            ]
            nbr_flow_var = float(np.var(nbr_flow))
        else:
            nbr_flow_var = 0.0
        # ------------------------
        # FINAL FEATURE VECTOR (M1 + 3 additions)
        # ------------------------
        # G.nodes[node]["features"] = [
        #     float(indeg),
        #     float(outdeg),
        #     float(np.log1p(indeg)),
        #     float(np.log1p(outdeg)),

        #     float(indeg / (outdeg + 1e-6)),   # in/out ratio
        #     float(deg_imbalance),
        #     float(flow_asym),

        #     float(is_source),
        #     float(is_sink),

        #     float(avg_neighbor_degree),
        #     float(deg_minus_nbr_mean),
        #     float(deg_over_nbr_mean),
        #     float(norm_density_contrast),

        #     float(clustering_coeff),
        #     float(ego_edges_frac),

        #     # ---- NEW (recommended minimal set) ----
        #     # float(ext_frac), # removed 
        #     float(twohop_avg_degree),
        #     float(hop_contrast),

        #     # ---- NEW tiny binary regime flags ----
        #     float(is_low_in_high_out),
        #     float(is_high_in_low_out),

        #     # # ---- NEW - feb 1 (global structure features) ----
        #     float(bridge_frac),
        #     float(scc_size),
        #     float(is_trivial_scc),
        # ]

        G.nodes[node]["features"] = [
            float(indeg),
            float(outdeg),
            float(np.log1p(indeg)),
            float(np.log1p(outdeg)),

            float(indeg / (outdeg + 1e-6)),
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

            # ---- NEW (partition-free boundary cues) ----
            float(ego_cut_ratio),
            float(twohop_avg_degree),
            float(hop_contrast),
            float(nbr_flow_var),

            # ---- regime flags ----
            float(is_low_in_high_out),
            float(is_high_in_low_out),

            # ---- global structure ----
            float(bridge_frac),
            float(scc_size),
            float(is_trivial_scc),
        ]

    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G, output_gml)
    print(f"[INFO] Saved processed GML → {output_gml}")

############################################################
# STEP 4: Batch processing
############################################################

# ROOT_RAW = "graphs/raw_v2/raw"
# ROOT_OUT = "graphs/processed_feb1_m1_new"


ROOT_RAW = "new_graphs_crypto/raw/raw"
ROOT_OUT = "new_graphs_crypto/processed_feb1_m1_new"


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
# STEP 5: Save metadata
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