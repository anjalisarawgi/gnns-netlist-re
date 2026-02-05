import os
import json
import numpy as np
import networkx as nx
from collections import deque
import re

import torch

GATE_TYPES = [
    "INPUT", "OUTPUT",
    "AND", "OR", "NAND", "NOR", "XOR", "XNOR",
    "INV", "AOI", "OAI", "MUX", "DFF",
    "UNKNOWN"
]

gate2id = {g: i for i, g in enumerate(GATE_TYPES)}

def parse_gate_from_label(label_copy: str):
    if label_copy is None:
        return "UNKNOWN"

    name = label_copy.strip("'").upper()

    # explicit first
    if "INPUT" in name:
        return "INPUT"
    if "OUTPUT" in name:
        return "OUTPUT"

    # logic gates by prefix
    for g in GATE_TYPES:
        if g not in ("INPUT", "OUTPUT", "UNKNOWN") and name.startswith(g):
            return g

    return "UNKNOWN"



def encode_gate(gate_type: str):
    gate_onehot = torch.zeros(len(GATE_TYPES), dtype=torch.float32)

    if gate_type in gate2id:
        gate_onehot[gate2id[gate_type]] = 1.0
    else:
        gate_onehot[gate2id["UNKNOWN"]] = 1.0

    return gate_onehot


def count_1hop_gate_types(node, G_und, G_dir):
    """
    Returns a vector of length |GATE_TYPES|
    where each entry counts 1-hop neighbors of that gate type.
    """
    counts = torch.zeros(len(GATE_TYPES), dtype=torch.float32)

    for nb in G_und.neighbors(node):
        nb_data = G_dir.nodes[nb]
        nb_label = nb_data.get("label_copy", "")
        nb_gate = parse_gate_from_label(nb_label)

        idx = gate2id.get(nb_gate, gate2id["UNKNOWN"])
        counts[idx] += 1.0

    return counts

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


        node_data = G_dir.nodes[node]
        label_copy = node_data.get("label_copy", "")

        # gate_type = parse_gate_from_label(label_copy)
        # gate_feats = encode_gate(gate_type)
        gate_type = parse_gate_from_label(label_copy)

        # one-hot of THIS node
        gate_onehot = encode_gate(gate_type)

        # 1-hop neighbor gate-type counts
        gate_1hop_counts = count_1hop_gate_types(node, G_und, G_dir)

        # combined gate encoding
        gate_feats = torch.cat([gate_onehot, gate_1hop_counts])

        # structural / graph features
        struct_feats = torch.tensor([
            in_d, out_d, fan_ratio,
            in_mean, in_std,
            out_mean, out_std,
            float(ego_density),
            kcore, pager, d_io, f_reach,
        ], dtype=torch.float32)

        x = torch.cat([gate_feats, struct_feats])
        G_dir.nodes[node]["features"] = x.tolist()

    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G_dir, output_gml)
    print(f"[INFO] Saved processed GML → {output_gml}")


ROOT_RAW = "graphs/raw_v2/raw"
ROOT_OUT = "graphs/processed_partitions_boundaryM1_oneHotAdd"
# ROOT_RAW = "new_graphs_crypto/raw/raw"
# ROOT_OUT = "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd"

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
