import os
import networkx as nx
import numpy as np

def is_graph_connected(G):
    """Return True if the graph is fully connected (weakly)."""
    return nx.is_weakly_connected(G)

def has_boundary_labels(G):
    """Return True if the graph contains at least one boundary label attribute."""
    for _, data in G.nodes(data=True):
        if "boundary" in data:
            return True
    return False



def is_io_label(lbl):
    lbl = str(lbl).strip("'")
    return ("INPUT" in lbl) or ("OUTPUT" in lbl)


def process_single_gml(input_gml: str, output_gml: str) -> None:
    print("[INFO] Processing:", input_gml)

    G = nx.read_gml(input_gml)
    G = G.to_directed()

    gate_types = [
        "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR",
        "INV", "BUF", "AOI", "OAI", "DFF", "MUX"
    ]
    gate2idx = {g: i for i, g in enumerate(gate_types)}

    def extract_gate_type(label: str) -> str:
        base = str(label).strip("'").split("_")[0]
        for gate in gate_types:
            if gate in base:
                return gate
        return "UNKNOWN"

    clustering = nx.clustering(G.to_undirected())

    for node in G.nodes():
        raw_label = G.nodes[node].get("label", node)
        raw_partition = G.nodes[node].get("partition", node)

        partition_cleaned = str(raw_partition).strip("'")
        label = str(raw_label).strip("'")
        gate_type = extract_gate_type(label)

        # one-hot (currently unused, but kept so your pipeline stays consistent)
        onehot = [0] * len(gate_types)
        if gate_type in gate2idx:
            onehot[gate2idx[gate_type]] = 1

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)

        boundary_flag = int(G.nodes[node].get("boundary", 0))
        clustering_coeff = round(float(clustering.get(node, 0.0)), 2)

        # 1-hop neighbors (directed-aware)
        nbrs = set(G.predecessors(node)) | set(G.successors(node))

        # avg neighbor degree
        if nbrs:
            avg_neighbor_degree = round(float(np.mean([G.degree(n) for n in nbrs])), 2)
        else:
            avg_neighbor_degree = 0.0

        # in/out ratio + imbalance
        in_out_ratio = round((indeg + 1) / (outdeg + 1), 2)
        abs_in_out_diff = float(abs(indeg - outdeg))  # optional

        # fraction of boundary neighbors
        if nbrs:
            boundary_nbrs = sum(int(G.nodes[n].get("boundary", 0)) for n in nbrs)
            frac_boundary_nbrs = round(boundary_nbrs / len(nbrs), 2)
        else:
            frac_boundary_nbrs = 0.0

        is_connected_to_io = float(any(is_io_label(G.nodes[n].get("label", "")) for n in nbrs))

        G.nodes[node]["features"] = [
            float(indeg),
            float(outdeg),
            avg_neighbor_degree,
            clustering_coeff,
            in_out_ratio,
            frac_boundary_nbrs,
            float(boundary_flag),
            # abs_in_out_diff,  # include if you want it
            is_connected_to_io,
        ]

        clean_label = label.strip("'")
        if "INPUT" in clean_label:
            G.nodes[node]["is_IO"] = 1
        elif "OUTPUT" in clean_label:
            G.nodes[node]["is_IO"] = 2
        else:
            G.nodes[node]["is_IO"] = 0
            G.nodes[node]["subcircuit_id"] = partition_cleaned

    os.makedirs(os.path.dirname(output_gml) or ".", exist_ok=True)
    nx.write_gml(G, output_gml)
    print("[INFO] Saved processed GML to:", output_gml)

if __name__ == "__main__":
    # --- set these paths ---
    input_gml = "graphs/raw_v2/raw/aes-master/osu035/aes_core_combined_m2.gml"
    output_gml = "test.gml"

    # --- checks (same as your batch script) ---
    G_raw = nx.read_gml(input_gml)

    if not is_graph_connected(G_raw):
        raise RuntimeError(f"Graph NOT weakly connected: {input_gml}")

    if not has_boundary_labels(G_raw):
        raise RuntimeError(f"Graph has NO boundary labels: {input_gml}")

    process_single_gml(input_gml, output_gml)