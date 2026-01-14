import os
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

    clustering = nx.clustering(G.to_undirected())
    G_undirected = G.to_undirected()

    # Betweenness centrality (sampled) — make k safe for small graphs
    k = min(100, G_undirected.number_of_nodes())
    betweenness = nx.betweenness_centrality(G_undirected, normalized=True, k=k)

    # IO nodes
    io_nodes = [n for n, d in G.nodes(data=True) if is_io_label(d.get("label", ""))]

    # Min distance to any IO node
    min_io_distances = {}
    for node in G.nodes():
        try:
            dists = [
                nx.shortest_path_length(G_undirected, source=node, target=io)
                for io in io_nodes
                if nx.has_path(G_undirected, node, io)
            ]
            min_io_distances[node] = min(dists) if dists else -1
        except Exception:
            min_io_distances[node] = -1

    for node in G.nodes():
        raw_label = G.nodes[node].get("label", node)
        raw_partition = G.nodes[node].get("partition", node)

        label = str(raw_label).strip("'")
        partition_cleaned = str(raw_partition).strip("'")

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)

        clustering_coeff = float(clustering.get(node, 0.0))

        nbrs = set(G.predecessors(node)) | set(G.successors(node))

        if nbrs:
            avg_neighbor_degree = float(np.mean([G.degree(n) for n in nbrs]))
            boundary_nbrs = sum(int(G.nodes[n].get("boundary", 0)) for n in nbrs)
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
            is_connected_to_io,
            float(betweenness.get(node, 0.0)),
            float(min_io_distances.get(node, -1)),
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
    print(f"[INFO] Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

############################################################
# STEP 3: Run on your one file (processed_v2 -> processed_v4)
############################################################

if __name__ == "__main__":
    input_gml = "graphs/processed_v2/openmsp430_latest/nangate/altera_reset_controller_combined_m1.gml"
    output_gml = "graphs/processed_v4/openmsp430_latest/nangate/altera_reset_controller_combined_m1.gml"

    # Optional validation before processing (recommended)
    G_check = nx.read_gml(input_gml)
    if not is_graph_connected(G_check):
        raise RuntimeError(f"Graph not connected: {input_gml}")
    if not has_boundary_labels(G_check):
        raise RuntimeError(f"No boundary labels: {input_gml}")

    process_single_gml(input_gml, output_gml)