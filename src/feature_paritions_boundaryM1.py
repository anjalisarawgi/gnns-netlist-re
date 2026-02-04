import os
import json
import numpy as np
import networkx as nx
from collections import deque
import re

def parse_cell_name(label_copy):
    """
    Returns a dict of semantic gate features from label_copy.
    """
    feats = {
        "is_INPUT": 0.0,
       "is_OUTPUT": 0.0,
 
        "is_AND": 0.0,
        "is_OR": 0.0,
        "is_XOR": 0.0,
        "is_INV": 0.0,
        "is_BUF": 0.0,
        "is_MUX": 0.0,
        "is_SEQ": 0.0,
        "is_ff": 0.0,
        "is_OTHER": 0.0,
        "input_count": -1.0,
        "is_arith": 0.0,
    }

    if not label_copy:
        feats["is_OTHER"] = 1.0
        return feats

    name = label_copy.strip("'").upper()


    # ---- Primary IO ----
    if "INPUT" in name:
        feats["is_INPUT"] = 1.0
        feats["input_count"] = 0.0
        return feats

    if "OUTPUT" in name:
        feats["is_OUTPUT"] = 1.0
        feats["input_count"] = 1.0
        return feats

    # --- Sequential ---
    if "DFF" in name or "FF" in name:
        feats["is_SEQ"] = 1.0
        feats["is_ff"] = 1.0
        feats["input_count"] = 1.0
        return feats

    if "LATCH" in name:
        feats["is_SEQ"] = 1.0
        feats["input_count"] = 1.0
        return feats

    # --- Gate families ---
    # --- Composite first ---
    if "XNOR" in name:
        feats["is_XOR"] = 1.0
        feats["is_INV"] = 1.0
        feats["is_arith"] = 1.0
    elif "XOR" in name:
        feats["is_XOR"] = 1.0
        feats["is_arith"] = 1.0
    elif "NAND" in name:
        feats["is_AND"] = 1.0
        feats["is_INV"] = 1.0
    elif "NOR" in name:
        feats["is_OR"] = 1.0
        feats["is_INV"] = 1.0
    elif "AND" in name:
        feats["is_AND"] = 1.0
    elif "OR" in name:
        feats["is_OR"] = 1.0

    if "INV" in name:
        feats["is_INV"] = 1.0
    if "BUF" in name:
        feats["is_BUF"] = 1.0
    if "MUX" in name or "MX" in name:
        feats["is_MUX"] = 1.0
        feats["is_arith"] = 1.0

    # --- Input count (best-effort) ---
    if feats["is_XOR"] or feats["is_AND"] or feats["is_OR"]:
        m = re.search(r'(\d)', name)
        if m:
            feats["input_count"] = float(m.group(1))


    # fallback
    if sum(feats[k] for k in ["is_AND","is_OR","is_XOR","is_INV","is_BUF","is_MUX","is_SEQ"]) == 0:
        feats["is_OTHER"] = 1.0

    for k in feats:
        if k != "input_count":
            feats[k] = float(1.0 if feats[k] else 0.0)

    return feats


def is_graph_connected(G):
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)

def has_boundary_labels(G):
    for _, data in G.nodes(data=True):
        try:
            if int(data.get("boundary", 0)) == 1:
                return True
        except Exception:
            pass
    return False

def _safe_mean_std(values):
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())

def _ego_nodes_khop_undirected(G_und, start, k=2):
    # BFS up to depth k in undirected graph
    visited = {start}
    q = deque([(start, 0)])
    while q:
        v, d = q.popleft()
        if d == k:
            continue
        for nb in G_und.neighbors(v):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return visited

def _ego_density_undirected(G_und, nodeset):
    # density = 2E / (N(N-1)) for undirected
    n = len(nodeset)
    if n <= 1:
        return 0.0
    sub = G_und.subgraph(nodeset)
    e = sub.number_of_edges()
    return float((2.0 * e) / (n * (n - 1)))

def _multi_source_bfs_undirected(G_und, sources):
    # returns dist dict from nearest source for nodes reached
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        v = q.popleft()
        dv = dist[v]
        for nb in G_und.neighbors(v):
            if nb not in dist:
                dist[nb] = dv + 1
                q.append(nb)
    return dist

def _forward_reach_within_k(G_dir, start, k=3):
    # directed forward BFS up to k hops, count unique reached excluding start
    visited = {start}
    q = deque([(start, 0)])
    while q:
        v, d = q.popleft()
        if d == k:
            continue
        for nb in G_dir.successors(v):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return len(visited) - 1

def process_single_gml(input_gml, output_gml, reach_k=3, ego_k=2):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml)
    # Ensure directed view if possible
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G

    # Undirected approximation for some metrics
    G_und = G_dir.to_undirected()

    # Precompute degrees
    indeg = dict(G_dir.in_degree())
    outdeg = dict(G_dir.out_degree())
    deg_und = dict(G_und.degree())


    # k-core number (undirected)
    try:
        core_num = nx.core_number(G_und)
    except Exception:
        # fallback: all zeros if core_number fails
        core_num = {n: 0 for n in G_dir.nodes()}

    # PageRank (directed)
    # For large graphs, this can still be okay; if it becomes heavy, we can switch to approximate PR.
    try:
        pr = nx.pagerank(G_dir, alpha=0.85, max_iter=200, tol=1e-06)
    except Exception:
        pr = {n: 0.0 for n in G_dir.nodes()}

    # IO nodes for distance feature
    io_nodes = []
    for n, data in G_dir.nodes(data=True):
        try:
            if int(data.get("is_IO", 0)) == 1:
                io_nodes.append(n)
        except Exception:
            pass

    # Multi-source BFS distances to IO (undirected)
    if len(io_nodes) > 0:
        dist_to_io = _multi_source_bfs_undirected(G_und, io_nodes)
    else:
        dist_to_io = {}

    for node in G_dir.nodes():
        in_d = float(indeg.get(node, 0))
        out_d = float(outdeg.get(node, 0))

        

        # fan-in/fan-out ratio (smoothed)
        fan_ratio = float((in_d + 1.0) / (out_d + 1.0))

        # neighbor degree stats
        in_neigh = list(G_dir.predecessors(node))
        out_neigh = list(G_dir.successors(node))

        in_neigh_deg = [deg_und.get(u, 0) for u in in_neigh]
        out_neigh_deg = [deg_und.get(u, 0) for u in out_neigh]

        in_mean, in_std = _safe_mean_std(in_neigh_deg)
        out_mean, out_std = _safe_mean_std(out_neigh_deg)

        # 2-hop ego density (undirected)
        ego_nodes = _ego_nodes_khop_undirected(G_und, node, k=ego_k)
        ego_density = _ego_density_undirected(G_und, ego_nodes)

        # k-core
        kcore = float(core_num.get(node, 0))

        # pagerank
        pager = float(pr.get(node, 0.0))

        # distance to IO (min hops). If unreachable or no IO, set -1
        d_io = float(dist_to_io.get(node, -1))

        # forward reach count within 3 hops (directed)
        f_reach = float(_forward_reach_within_k(G_dir, node, k=reach_k))

        cell_feats = parse_cell_name(G_dir.nodes[node].get("label_copy", ""))

        if indeg.get(node, 0) == 0 and outdeg.get(node, 0) > 0:
            cell_feats["is_INPUT"] = 1.0

        if indeg.get(node, 0) > 0 and outdeg.get(node, 0) == 0:
            cell_feats["is_OUTPUT"] = 1.0


        G_dir.nodes[node]["features"] = [
            in_d, out_d, fan_ratio,
            in_mean, in_std,
            out_mean, out_std,
            float(ego_density),
            kcore, pager, d_io, f_reach,

            ### input outpue below: 
            # cell_feats["is_INPUT"],
            # cell_feats["is_OUTPUT"],
            ### other gates below: 
            cell_feats["is_AND"],
            cell_feats["is_OR"],
            cell_feats["is_XOR"],
            cell_feats["is_INV"],
            cell_feats["is_BUF"],
            cell_feats["is_MUX"],
            cell_feats["is_SEQ"],
            cell_feats["is_ff"],
            cell_feats["is_OTHER"],
            cell_feats["input_count"],
            cell_feats["is_arith"],
        ]


    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G_dir, output_gml)
    print(f"[INFO] Saved processed GML → {output_gml}")


ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_partitions_boundaryM1_oneHot"
# ROOT_RAW = "new_graphs_crypto/raw/raw"
# ROOT_OUT = "new_graphs_crypto/processed_partitions_boundaryM1_oneHot"

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
                process_single_gml(input_gml, output_gml, reach_k=3, ego_k=2)
                usable_graphs.append(output_gml)

                G_out = nx.read_gml(output_gml)
                any_node = next(iter(G_out.nodes()))
                print("Feature dim:", len(G_out.nodes[any_node]["features"]))

            except Exception as e:
                print(f"[CRASH] {input_gml}: {e}")

# save lists
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
