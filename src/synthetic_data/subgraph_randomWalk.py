import os
import random
import networkx as nx

def graphsaint_sampler(G, num_subgraphs=10, walk_steps=5, output_dir="subgraphs_graphsaint", seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    nodes = list(G.nodes())

    for i in range(num_subgraphs):
        start_node = random.choice(nodes)
        walk_nodes = [start_node]

        current = start_node
        for _ in range(walk_steps):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            walk_nodes.append(next_node)
            current = next_node

        walk_nodes_set = set(walk_nodes)
        subG = G.subgraph(walk_nodes_set).copy()
        out_path = os.path.join(output_dir, f"subgraph_{i+1}.gml")
        nx.write_gml(subG, out_path)
        print(f"[✓] Saved subgraph {i+1}: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges -> {out_path}")




if __name__ == "__main__":
    gml_path = "graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_randomError_1.gml"
    output_dir = "graphs/processed/aes_encryption_latest/osu035/subgraphs_graphsaint"

    print(f"Loading graph from: {gml_path}")
    G = nx.read_gml(gml_path)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    graphsaint_sampler(
        G,
        num_subgraphs=10,
        walk_steps=1000,
        output_dir=output_dir,
        seed=42
    )