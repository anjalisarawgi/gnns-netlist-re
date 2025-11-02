import networkx as nx
import community as community_louvain
import numpy as np


graph_path = "graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi_test6.gml"
G = nx.read_gml(graph_path)


print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


G_undirected = G.to_undirected() # converting to undirected for Louvain
partition = community_louvain.best_partition(G_undirected)
print(f"Detected {len(set(partition.values()))} communities")
nx.set_node_attributes(G, partition, "community_id") # Adding community IDs back to original (directed) graph

out_path = graph_path.replace(".gml", "_comm.gml")
nx.write_gml(G, out_path)
print(f"Saved community-annotated graph to {out_path}")