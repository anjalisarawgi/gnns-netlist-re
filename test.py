import networkx as nx
import random

def join_two_versions(gml_path, output_path):
    # Load the original GML
    G = nx.read_gml(gml_path)

    # Copy 1: Original
    G1 = G.copy()

    # Copy 2: Renamed nodes to avoid collisions
    G2 = nx.relabel_nodes(G.copy(), lambda x: f"{x}_v2")

    # Combine both graphs
    G_combined = nx.compose(G1, G2)

    # Connect a random node from each graph
    node1 = random.choice(list(G1.nodes))
    node2 = random.choice(list(G2.nodes))
    G_combined.add_edge(node1, node2)

    # Save the result
    nx.write_gml(G_combined, output_path)
    print(f"Saved combined graph to: {output_path}")

# Example usage
join_two_versions(
    gml_path="mwe/aes_sbox_gephi.gml",
    output_path="synthetic_combined_sbox.gml"
)