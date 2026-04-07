import os
import json
import numpy as np
import networkx as nx
from collections import deque
import re
import community.community_louvain as community_louvain
import torch
import igraph as ig
import leidenalg

GATE_TYPES = ["INPUT", "OUTPUT", "AND", "OR", "NAND", "NOR", "XOR", "XNOR", "INV", "AOI", "OAI", "MUX", "DFF", "UNKNOWN"]
gate2id = {g: i for i, g in enumerate(GATE_TYPES)}

def parse_gate_from_label(label_copy: str):
    if label_copy is None:
        return "UNKNOWN"

    name = label_copy.strip("'").upper()

    # check if input / output gate
    if "INPUT" in name:
        return "INPUT"
    if "OUTPUT" in name:
        return "OUTPUT"

    # if not then one hot for other gate names
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
def normalize_counts(counts: torch.Tensor):
    total = counts.sum()
    if total > 0:
        return counts / total
    return counts  # stays zero if no neighbors
    

def count_2hop_gate_types(node, G_und, G_dir):
    """
    Returns counts of gate types exactly 2 hops away.
    """
    counts = torch.zeros(len(GATE_TYPES), dtype=torch.float32)

    neighbors_1 = set(G_und.neighbors(node))

    neighbors_2 = set()
    for nb in neighbors_1:
        neighbors_2.update(G_und.neighbors(nb))

    # remove 1-hop neighbors and the node itself
    neighbors_2 = neighbors_2 - neighbors_1
    neighbors_2.discard(node)

    for nb in neighbors_2:
        nb_data = G_dir.nodes[nb]
        nb_label = nb_data.get("label_copy", "")
        nb_gate = parse_gate_from_label(nb_label)

        idx = gate2id.get(nb_gate, gate2id["UNKNOWN"])
        counts[idx] += 1.0

    return counts

# def count_pi_po_connections(node, G_dir):
#     pi_connections = 0
#     po_connections = 0

#     for nb in G_dir.predecessors(node):
#         label = str(G_dir.nodes[nb].get("label_copy", "")).upper()
#         if "INPUT" in label:
#             pi_connections += 1

#     for nb in G_dir.successors(node):
#         label = str(G_dir.nodes[nb].get("label_copy", "")).upper()
#         if "OUTPUT" in label:
#             po_connections += 1

#     return float(pi_connections), float(po_connections)

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

def _backward_reach_within_k(G_dir, start, k=3):
    visited = {start}
    q = deque([(start, 0)])
    while q:
        v, d = q.popleft()
        if d == k:
            continue
        for nb in G_dir.predecessors(v):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return len(visited) - 1

def _mean_neighbor_ego_density(G_und, node, k=2):
    densities = []
    for nb in G_und.neighbors(node):
        nb_ego = _ego_nodes_khop_undirected(G_und, nb, k=k)
        densities.append(_ego_density_undirected(G_und, nb_ego))
    if len(densities) == 0:
        return 0.0
    return float(np.mean(densities))

def _clean_partition(p):
    if p is None:
        return None
    return str(p).strip("'")

def fraction_same_partition_1hop(node, G_und, G_dir):
    p0 = _clean_partition(G_dir.nodes[node].get("partition"))
    if p0 is None:
        return 0.0

    neighs = list(G_und.neighbors(node))
    if len(neighs) == 0:
        return 0.0

    same = 0
    for nb in neighs:
        p_nb = _clean_partition(G_dir.nodes[nb].get("partition"))
        if p_nb == p0:
            same += 1

    return same / len(neighs)


def fraction_same_partition_2hop(node, G_und, G_dir):
    p0 = _clean_partition(G_dir.nodes[node].get("partition"))
    if p0 is None:
        return 0.0

    # reuse your ego function
    nodes_2hop = _ego_nodes_khop_undirected(G_und, node, k=2)
    nodes_2hop.discard(node)

    if len(nodes_2hop) == 0:
        return 0.0

    same = 0
    for nb in nodes_2hop:
        p_nb = _clean_partition(G_dir.nodes[nb].get("partition"))
        if p_nb == p0:
            same += 1

    return same / len(nodes_2hop)

def compute_structural_edge_features(G_dir, edge_index_list):
    """
    Generates features for every wire (edge) in the circuit.
    """
    edge_attrs = []
    
    # Pre-calculate node degrees for speed
    in_degrees = dict(G_dir.in_degree())
    out_degrees = dict(G_dir.out_degree())
    
    for u, v in edge_index_list:
        # Feature 1: Fan-out of the source (How many gates does this signal feed?)
        src_fan_out = float(out_degrees.get(u, 1))
        
        # Feature 2: Fan-in of the destination (How many signals feed this gate?)
        dst_fan_in = float(in_degrees.get(v, 1))
        
        # Feature 3: Leverage Ratio (Does a big driver feed a small gate?)
        leverage = src_fan_out / (dst_fan_in + 1e-6)
        
        # Feature 4: Is it a "Feedback" edge? (Heuristic: u has higher index than v)
        # This can help find loops in sequential logic.
        is_feedback = 1.0 if str(u) > str(v) else 0.0

        edge_feat = [src_fan_out, dst_fan_in, leverage, is_feedback]
        edge_attrs.append(edge_feat)
        
    return edge_attrs

def get_unique_partition_count(G_dir):
    partitions = set()

    for _, data in G_dir.nodes(data=True):
        p = _clean_partition(data.get("partition"))
        if p is not None:
            partitions.add(p)

    return float(len(partitions))

def process_single_gml(input_gml, output_gml, tech,  reach_k=3, ego_k=2):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml)
    if not G.is_directed():
        G_dir = G.to_directed()
    else:
        G_dir = G

    G_und = G_dir.to_undirected()
    indeg = dict(G_dir.in_degree())
    outdeg = dict(G_dir.out_degree())
    deg_und = dict(G_und.degree())

    # unsuerpvised - features test:
    partition_map = community_louvain.best_partition(G_und)
    print(f"[INFO] Louvain found {len(set(partition_map.values()))} clusters")



    # Leiden community detection (unsupervised)
    g_ig = ig.Graph.from_networkx(G_und)
    leiden_partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, seed=42)
    leiden_map = {}
    for cluster_id, node_list in enumerate(leiden_partition):
        for node_idx in node_list:
            original_node = g_ig.vs[node_idx]["_nx_name"]
            leiden_map[original_node] = cluster_id
    print(f"[INFO] Leiden found {len(leiden_partition)} clusters")

    try:
        core_num = nx.core_number(G_und)
    except Exception:
        core_num = {n: 0 for n in G_dir.nodes()}

    # disable if it itakes ery long
    try:
        pr = nx.pagerank(G_dir, alpha=0.85, max_iter=200, tol=1e-06)
    except Exception:
        pr = {n: 0.0 for n in G_dir.nodes()}

    for node in G_dir.nodes():
        label_copy = str(G_dir.nodes[node].get("label_copy", "")).strip("'").upper()
        
        if label_copy.startswith("INPUT") or "INPUT" in label_copy:
            G_dir.nodes[node]["is_input"] = 1
            G_dir.nodes[node]["is_output"] = 0
            G_dir.nodes[node]["is_IO"] = 1
        elif label_copy.startswith("OUTPUT") or "OUTPUT" in label_copy:
            G_dir.nodes[node]["is_input"] = 0
            G_dir.nodes[node]["is_output"] = 1
            G_dir.nodes[node]["is_IO"] = 1
        else:
            G_dir.nodes[node]["is_input"] = 0
            G_dir.nodes[node]["is_output"] = 0
            G_dir.nodes[node]["is_IO"] = 0


    io_nodes = []
    for n, data in G_dir.nodes(data=True):
        try:
            if int(data.get("is_IO", 0)) == 1:
                io_nodes.append(n)
        except Exception:
            pass

    if len(io_nodes) > 0:
        dist_to_io = _multi_source_bfs_undirected(G_und, io_nodes)
        default_dist_io = max(dist_to_io.values()) + 1 if dist_to_io else 0.0
    else:
        dist_to_io = {}
        default_dist_io = 0.0


    num_unique_partitions = get_unique_partition_count(G_dir)

    num_nodes = float(G_dir.number_of_nodes())
    num_edges = float(G_dir.number_of_edges())
    graph_density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1))


    unknown_gate_labels = {}
    for node in G_dir.nodes():
        in_d = float(indeg.get(node, 0))
        out_d = float(outdeg.get(node, 0))
        fan_ratio = float((in_d + 1.0) / (out_d + 1.0))
        in_neigh = list(G_dir.predecessors(node))
        out_neigh = list(G_dir.successors(node))
        in_neigh_deg = [deg_und.get(u, 0) for u in in_neigh]
        out_neigh_deg = [deg_und.get(u, 0) for u in out_neigh]
        in_mean, in_std = _safe_mean_std(in_neigh_deg)
        out_mean, out_std = _safe_mean_std(out_neigh_deg)

        # other features
        # ego_nodes = _ego_nodes_khop_undirected(G_und, node, k=ego_k)
        # ego_density = _ego_density_undirected(G_und, ego_nodes)
        ego_nodes = _ego_nodes_khop_undirected(G_und, node, k=ego_k)
        ego_density = _ego_density_undirected(G_und, ego_nodes)
        # ego_density_nb_mean = _mean_neighbor_ego_density(G_und, node, k=ego_k)
        # ego_density_contrast = ego_density - ego_density_nb_mean

        kcore = float(core_num.get(node, 0))
        pager = float(pr.get(node, 0.0))
        d_io = float(dist_to_io.get(node, default_dist_io)) # d_io = float(dist_to_io.get(node, -1))


        f_reach = float(_forward_reach_within_k(G_dir, node, k=reach_k))
        b_reach = float(_backward_reach_within_k(G_dir, node, k=reach_k))
        reach_asym = (f_reach - b_reach) / (f_reach + b_reach + 1.0)


        node_data = G_dir.nodes[node]
        label_copy = node_data.get("label_copy", "")

        # one hot encoding section ---:::::::
        gate_type = parse_gate_from_label(label_copy)


        if gate_type == "UNKNOWN":
            key = str(label_copy).strip("'")
            unknown_gate_labels[key] = unknown_gate_labels.get(key, 0) + 1

        gate_onehot = encode_gate(gate_type)
        gate_1hop_counts = count_1hop_gate_types(node, G_und, G_dir)
        gate_1hop_frac = normalize_counts(gate_1hop_counts)

        # gate_2hop_counts = count_2hop_gate_types(node, G_und, G_dir)

        # gate_feats = torch.cat([gate_onehot, gate_1hop_counts, gate_2hop_counts]) #### check
        gate_feats = torch.cat([gate_onehot, gate_1hop_frac]) #### check

        neighbor_degs = [deg_und.get(nb, 0) for nb in G_und.neighbors(node)]
        deg_contrast = deg_und.get(node, 0) - (np.mean(neighbor_degs) if neighbor_degs else 0.0)

        # structural / graph features
        struct_feats = torch.tensor([
            in_d, out_d, fan_ratio, 
            in_mean, in_std, 
            out_mean, out_std, 
            float(ego_density),
            kcore, pager, 
            f_reach, b_reach, reach_asym, deg_contrast, 
            # d_io, # input output infoa
        ], dtype=torch.float32)

        x = torch.cat([gate_feats, struct_feats])
        G_dir.nodes[node]["features"] = x.tolist()

        # partition features 
        frac_same_p1 = fraction_same_partition_1hop(node, G_und, G_dir)
        frac_same_p2 = fraction_same_partition_2hop(node, G_und, G_dir)

        G_dir.nodes[node]["partition_features"] = [
            float(frac_same_p1),
            float(frac_same_p2),
            int(num_unique_partitions) # basically total partitions
        ]

        # graph features
        graph_feats = torch.tensor([num_nodes, num_edges, graph_density], dtype=torch.float32)
        G_dir.nodes[node]["graph_features"] = graph_feats.tolist()

        # unsupervised - feature test
        my_cluster = partition_map.get(node, 0)
        neighbors_und = list(G_und.neighbors(node))
        louvain_frac_1hop = (
            sum(1 for nb in neighbors_und if partition_map.get(nb) == my_cluster) / len(neighbors_und)
            if neighbors_und else 0.0
        )
        # G_dir.nodes[node]["unsupervised_partition_feats"] = [float(louvain_frac_1hop)]
        two_hop = _ego_nodes_khop_undirected(G_und, node, k=2)
        two_hop.discard(node)
        louvain_frac_2hop = (
            sum(1 for nb in two_hop if partition_map.get(nb) == my_cluster) / len(two_hop)
            if two_hop else 0.0
        )
        G_dir.nodes[node]["unsup_louvain_1hop"] = float(louvain_frac_1hop)
        G_dir.nodes[node]["unsup_louvain_2hop"] = float(louvain_frac_2hop)
        G_dir.nodes[node]["unsup_louvain_cluster"] = int(my_cluster)

        # Leiden unsupervised features
        my_leiden_cluster = leiden_map.get(node, 0)
        leiden_frac_1hop = (
            sum(1 for nb in neighbors_und if leiden_map.get(nb) == my_leiden_cluster) / len(neighbors_und)
            if neighbors_und else 0.0
        )
        leiden_two_hop = _ego_nodes_khop_undirected(G_und, node, k=2)
        leiden_two_hop.discard(node)
        leiden_frac_2hop = (
            sum(1 for nb in leiden_two_hop if leiden_map.get(nb) == my_leiden_cluster) / len(leiden_two_hop)
            if leiden_two_hop else 0.0
        )
        G_dir.nodes[node]["unsup_leiden_1hop"] = float(leiden_frac_1hop)
        G_dir.nodes[node]["unsup_leiden_2hop"] = float(leiden_frac_2hop)
        G_dir.nodes[node]["unsup_leiden_cluster"] = int(my_leiden_cluster)

    # edge features
    for u, v in G_dir.edges():
        # 1. Gate Logic Symmetry (The "Bundle" Signal)
        gate_u = parse_gate_from_label(G_dir.nodes[u].get("label_copy", ""))
        gate_v = parse_gate_from_label(G_dir.nodes[v].get("label_copy", ""))
        is_same_gate_type = 1.0 if gate_u == gate_v else 0.0

        # 3. Connectivity Delta (The "Flow" Signal)
        deg_u = float(deg_und.get(u, 0))
        deg_v = float(deg_und.get(v, 0))
        delta_deg = deg_u - deg_v

        # 4. Neighborhood Overlap (The "Ribbon" Signal)
        neigh_u = set(G_und.neighbors(u))
        neigh_v = set(G_und.neighbors(v))
        shared_count = float(len(neigh_u.intersection(neigh_v)))

        edge_feat = [
            is_same_gate_type,    # Does logic type persist?If the source and destination gates are the same type (AND→AND, INV→INV, etc.), the feature becomes 1. Otherwise it’s 0. 
            delta_deg,            # Is there a density change? whether the edge moves from a highly connected region to a less connected region, or the reverse.
            shared_count,         # Are they tightly coupled? This tells us whether the two nodes live inside the same local cluster.
        ]

        G_dir.edges[u, v]["edge_features"] = edge_feat
   
    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G_dir, output_gml)
    print(f"[INFO] Saved processed GML → {output_gml}")

    if len(unknown_gate_labels) > 0:
        import csv
        csv_path = output_gml.replace(".gml", "_unknown_gates.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gate_label", "count"])

            for gate, count in sorted(unknown_gate_labels.items(), key=lambda x: -x[1]):
                writer.writerow([gate, count])

        print(f"[INFO] Unknown gate report saved → {csv_path}")
    else:
        print(f"[INFO] unknown gates not found")



# ROOT_RAW = "graphs/raw_v2/raw"
# ROOT_OUT = "graphs/processed_boundaryDetection_march23"
# ROOT_RAW = "new_graphs_crypto/raw/raw/"
# ROOT_OUT = "new_graphs_crypto/processed_boundaryDetection_march23"


# ROOT_RAW = "crypto_graphs_final/raw_final/rawv4"
# ROOT_OUT = "crypto_graphs_final/processed_final/"
ROOT_RAW = "other_graphs_final/raw_final/rawv4"
ROOT_OUT = "other_graphs_final/processed_final/"

processed_dirs = {}
usable_graphs = []
unusable_graphs_connectivity = []
unusable_graphs_no_boundary = []
missing_boundary_graphs = {} 
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
                process_single_gml(input_gml, output_gml, tech, reach_k=3, ego_k=2)
                usable_graphs.append(output_gml)

                # Check for nodes with missing boundary labels
                G_out = nx.read_gml(output_gml)
                any_node = next(iter(G_out.nodes()))
                print("Feature dim:", len(G_out.nodes[any_node]["features"]))

                missing_boundary = []
                for node in G_out.nodes():
                    val = G_out.nodes[node].get("boundary", "MISSING")
                    if val == "MISSING":
                        missing_boundary.append(node)

                if missing_boundary:
                    print(f"[WARN] {len(missing_boundary)} nodes with NO boundary label in {output_gml}")
                    print(f"       Example nodes: {missing_boundary[:5]}")

                    # store full info
                    missing_boundary_graphs[output_gml] = {
                        "num_missing": len(missing_boundary),
                        "nodes": missing_boundary
                    }

                else:
                    print(f"[OK] All nodes have boundary labels in {output_gml}")

            except Exception as e:
                print(f"[CRASH] {input_gml}: {e}")


os.makedirs(ROOT_OUT, exist_ok=True)

with open(os.path.join(ROOT_OUT, "usable_graphs.json"), "w") as f:
    json.dump(usable_graphs, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_connectivity.json"), "w") as f:
    json.dump(unusable_graphs_connectivity, f, indent=4)

with open(os.path.join(ROOT_OUT, "unusable_graphs_no_boundary.json"), "w") as f:
    json.dump(unusable_graphs_no_boundary, f, indent=4)

with open(os.path.join(ROOT_OUT, "graphs_missing_boundary_labels.json"), "w") as f:
    json.dump(missing_boundary_graphs, f, indent=4)

print("Usable graphs:", len(usable_graphs))
print("Unusable (not connected) :", len(unusable_graphs_connectivity))
print("Unusable (no boundary labels) :", len(unusable_graphs_no_boundary))
