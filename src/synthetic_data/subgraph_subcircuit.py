import networkx as nx
import os 
import math 
from collections import defaultdict


gml_file_path = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml"
output_file_path = "graphs/processed/aes_encryption_latest/osu035/subgraphs/"

G = nx.read_gml(gml_file_path)

partition_to_nodes = defaultdict(list) ###?
for node_id, node_data in G.nodes(data = True):
    subcircuit_id = node_data.get('subcircuit_id', None)
    if subcircuit_id:
        partition_to_nodes[subcircuit_id].append(node_id)

print("total paritions:", len(partition_to_nodes))

# subgraphs 
subgraphs = {}
for partition_label, node_list in partition_to_nodes.items():
    subgraph = G.subgraph(node_list).copy()
    subgraphs[partition_label] = subgraph


for label, sg in subgraphs.items():
    filename = f"{label.replace('+', '_').replace('/', '_')}.gml"
    path = os.path.join(output_file_path, filename)
    nx.write_gml(sg, path)
    print(f"Saved subgraph '{label}' with {len(sg.nodes)} nodes to: {path}")

print("complete")