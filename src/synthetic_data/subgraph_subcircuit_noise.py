import networkx as nx
import os 
from collections import defaultdict
import random

gml_file_path = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml"
output_file_path = "graphs/processed/aes_encryption_latest/osu035/subgraphs_subcircuits_noisy/"
os.makedirs(output_file_path, exist_ok=True)

G = nx.read_gml(gml_file_path)

partition_to_nodes = defaultdict(list)
for node_id, node_data in G.nodes(data=True):
    subcircuit_id = node_data.get('subcircuit_id')
    if subcircuit_id:
        partition_to_nodes[subcircuit_id].append(node_id)

print("Total partitions:", len(partition_to_nodes))

n_noise = 100  # noise nodes per subgraph

all_nodes_set = set(G.nodes)

for label, node_list in partition_to_nodes.items():
    sg = G.subgraph(node_list).copy()
    existing_nodes = list(sg.nodes)

    noise_candidates = list(all_nodes_set - set(node_list))
    if len(noise_candidates) < n_noise:
        print(f"Warning: not enough noise candidates for {label}, using {len(noise_candidates)} instead of {n_noise}")
    selected_noise_nodes = random.sample(noise_candidates, min(n_noise, len(noise_candidates)))

    for noise_node in selected_noise_nodes:
        sg.add_node(noise_node, **G.nodes[noise_node])
        target_node = random.choice(existing_nodes)
        if G.has_edge(noise_node, target_node):
            sg.add_edge(noise_node, target_node, **G.get_edge_data(noise_node, target_node))
        else:
            sg.add_edge(noise_node, target_node)  

    filename = f"{label.replace('+', '_').replace('/', '_')}.gml"
    path = os.path.join(output_file_path, filename)
    nx.write_gml(sg, path)
    print(f"Saved subgraph '{label}' with {len(sg.nodes)} nodes (incl. noise) to: {path}")

print("Complete.")