import networkx as nx

G = nx.read_gml("aes_cipher_top_gephi_features.gml")

# Specify the subcircuit ID you want to extract
target_subcircuit_id = "top+us20"

# Filter nodes that belong to the target subcircuit
sub_nodes = [n for n, attr in G.nodes(data=True) if attr.get("subcircuit_id") == target_subcircuit_id]

# Create subgraph
subgraph = G.subgraph(sub_nodes).copy()

# Save the subgraph to a new GML file
subgraph_path = f"subgraph_{target_subcircuit_id.replace('+', '_')}.gml"
nx.write_gml(subgraph, subgraph_path)

print(f"Saved subgraph with subcircuit_id='{target_subcircuit_id}' to '{subgraph_path}'")

import networkx as nx

# For directed graphs use weakly connected components
components = list(nx.weakly_connected_components(G))  # If G is directed
# If G was undirected, use nx.connected_components(G)

print(f"Total components: {len(components)}")
for i, c in enumerate(components):
    print(f"Component {i}: {len(c)} nodes")