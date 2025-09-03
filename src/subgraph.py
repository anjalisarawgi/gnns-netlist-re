import networkx as nx
import os 


G = nx.read_gml("aes_cipher_top_gephi_features.gml")


# # based on subcircuit_id
# target_subcircuit_id = "top+us20"
# sub_nodes = [n for n, attr in G.nodes(data=True) if attr.get("subcircuit_id") == target_subcircuit_id]
# subgraph = G.subgraph(sub_nodes).copy()
# subgraph_path = f"subgraph_{target_subcircuit_id.replace('+', '_')}.gml"
# nx.write_gml(subgraph, subgraph_path)

# print(f"Saved subgraph with subcircuit_id='{target_subcircuit_id}' to '{subgraph_path}'")

# components = list(nx.weakly_connected_components(G)) 
# print(f"Total components: {len(components)}")
# for i, c in enumerate(components):
#     print(f"Component {i}: {len(c)} nodes")



# based on subcircuit 
# ## Loop through subcircuit values 0 to 3
# output_dir = "results/subcircuit_subgraphs"
# os.makedirs(output_dir, exist_ok=True)
# for target_subcircuit in [-1, 0, 1, 2, 3, 4, 5]:
#     sub_nodes = [n for n, attr in G.nodes(data=True) if attr.get("subcircuit") == target_subcircuit]
#     subgraph = G.subgraph(sub_nodes).copy()
#     subgraph_path = os.path.join(output_dir, f"subcircuit_{target_subcircuit}.gml")
#     nx.write_gml(subgraph, subgraph_path)
#     print(f"Saved subgraph with subcircuit={target_subcircuit} to '{subgraph_path}'")



output_dir = "results/subcircuit_subgraphs"
os.makedirs(output_dir, exist_ok=True)
subcircuit_dict = {n: attr.get("subcircuit") for n, attr in G.nodes(data=True)}

for target_subcircuit in [1, 2, 3, 4, 5]:
    target_nodes = {n for n, s in subcircuit_dict.items() if s == target_subcircuit}
    if not target_nodes:
        print(f"Skipping subcircuit {target_subcircuit} (no nodes)")
        continue

    connected_aux_nodes = set()
    for node in target_nodes:
        for neighbor in G.neighbors(node):
            if subcircuit_dict.get(neighbor) in [0, -1]:
                connected_aux_nodes.add(neighbor)
        for neighbor in G.predecessors(node):
            if subcircuit_dict.get(neighbor) in [0, -1]:
                connected_aux_nodes.add(neighbor)

    combined_nodes = target_nodes.union(connected_aux_nodes)
    subgraph = G.subgraph(combined_nodes).copy()

    subgraph_path = os.path.join(output_dir, f"subcircuit_{target_subcircuit}_with_connected_0_and_-1.gml")
    nx.write_gml(subgraph, subgraph_path)
    print(f"Saved subgraph: subcircuit={target_subcircuit} + connected 0/-1 → {subgraph_path}")