import os
import random
import networkx as nx

def generate_k_hop_subgraphs(G, num_subgraphs=10, radius=2, output_dir="subgraphs_khop", seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    nodes = list(G.nodes())

    for i in range(num_subgraphs):
        center = random.choice(nodes)
        subG = nx.ego_graph(G, center, radius=radius)

        out_path = os.path.join(output_dir, f"subgraph_{i+1}.gml")
        nx.write_gml(subG, out_path)
        print(f"[✓] Subgraph {i+1}: center={center}, {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges -> {out_path}")


# === Set your paths and config ===

# INPUT_GML_PATH = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml"
INPUT_GML_PATH = "graphs/synthetic/processed/aes_encryption_latest/osu035/1/aes_cipher_top_gephi_mixedError_1.gml"
OUTPUT_DIR = "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/"
NUM_SUBGRAPHS = 10
RADIUS = 20 
SEED = 42

# === Run ===

if __name__ == "__main__":
    print(f"Loading graph from: {INPUT_GML_PATH}")
    G = nx.read_gml(INPUT_GML_PATH)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    generate_k_hop_subgraphs(
        G,
        num_subgraphs=NUM_SUBGRAPHS,
        radius=RADIUS,
        output_dir=OUTPUT_DIR,
        seed=SEED
    )