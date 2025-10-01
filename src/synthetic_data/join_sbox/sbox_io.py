import networkx as nx
import re

def extract_bit_index(label):
    """Extracts bit index from strings like 'OUTPUT_\\out[3]' or 'INPUT_\\a[3]'."""
    match = re.search(r'\[(\d+)\]', label)
    return int(match.group(1)) if match else None

def is_input_node(label):
    return 'INPUT' in label

def is_output_node(label):
    return 'OUTPUT' in label

def join_gml_by_io(gml_path, output_path):
    # Load original graph
    G = nx.read_gml(gml_path)

    # Version 1
    G1 = G.copy()

    # Version 2 with renamed nodes
    G2 = nx.relabel_nodes(G.copy(), lambda x: f"{x}_v2")

    # Add is_io flags to all nodes
    def tag_io(graph, suffix=""):
        for node in graph.nodes:
            label = str(node)
            if is_input_node(label):
                graph.nodes[node]['is_io'] = 1
            elif is_output_node(label):
                graph.nodes[node]['is_io'] = 2
            else:
                graph.nodes[node]['is_io'] = 0

    tag_io(G1)
    tag_io(G2)

    # Combine
    G_combined = nx.compose(G1, G2)

    # Match output[i] → input[i]
    output_nodes = [(n, extract_bit_index(n)) for n in G1.nodes if is_output_node(n)]
    input_nodes = [(n, extract_bit_index(n)) for n in G2.nodes if is_input_node(n)]

    match_count = 0
    for out_node, out_idx in output_nodes:
        for in_node, in_idx in input_nodes:
            if out_idx is not None and out_idx == in_idx:
                G_combined.add_edge(out_node, in_node)
                match_count += 1

    print(f"Connected {match_count} output-input node pairs.")
    print(f"Tagged all nodes with is_io for Gephi.")

    # Save
    nx.write_gml(G_combined, output_path)
    print(f"Saved to: {output_path}")

# Example usage
join_gml_by_io(
    gml_path="mwe/aes_sbox_gephi.gml",
    output_path="synthetic_sbox_joined_by_io_tagged.gml"
)