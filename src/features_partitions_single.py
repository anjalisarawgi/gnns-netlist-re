import os
import json
import numpy as np
import networkx as nx

def is_graph_connected(G):
    if G.is_directed():
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)

def has_boundary_labels(G):
    for _, data in G.nodes(data=True):
        if int(data.get("boundary", 0)) == 1:
            return True
    return False

def is_io_label(label: str) -> bool:
    s = str(label).upper()
    return ("INPUT" in s) or ("OUTPUT" in s)

def process_single_gml(input_gml, output_gml):
    print("Processing:", input_gml)

    G = nx.read_gml(input_gml).to_directed()

    if not is_graph_connected(G):
        raise ValueError("Graph is not connected")

    if not has_boundary_labels(G):
        raise ValueError("Graph has no boundary labels")

    gate_types = [
        "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR",
        "INV", "BUF", "AOI", "OAI", "DFF", "MUX"
    ]

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
            avg_neighbor_degree = float(
                np.mean([G.degree(n) for n in nbrs])
            )
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


if __name__ == "__main__":
    INPUT_GML = "graphs/processed_v2/tiny_aes_latest/osu035/aes_128_combined_m2.gml"
    OUTPUT_GML =  "graphs/processed_v3/tiny_aes_latest/osu035/aes_128_combined_m2.gml"

    process_single_gml(INPUT_GML, OUTPUT_GML)