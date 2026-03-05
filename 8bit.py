# import networkx as nx
# from collections import defaultdict, Counter

# G = nx.read_gml('new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_march2Edge/aes_core/nangate/aes_cipher_top_combined_m1.gml')

# def neighbor_shape(G, node_id):
#     my_in  = G.in_degree(node_id)
#     my_out = G.out_degree(node_id)
#     in_neighbor_shapes  = tuple(sorted(
#         (G.in_degree(n), G.out_degree(n)) for n in G.predecessors(node_id)
#     ))
#     out_neighbor_shapes = tuple(sorted(
#         (G.in_degree(n), G.out_degree(n)) for n in G.successors(node_id)
#     ))
#     return (my_in, my_out, in_neighbor_shapes, out_neighbor_shapes)


# bus_candidates = defaultdict(list)
# for node_id, data in G.nodes(data=True):
#     # if data.get('boundary', 0) == 1:
#     sig = neighbor_shape(G, node_id)
#     bus_candidates[sig].append(node_id)

# # Initialize all nodes
# for node_id in G.nodes():
#     G.nodes[node_id]['bus_id']    = -1
#     G.nodes[node_id]['bus_width'] = 1
#     G.nodes[node_id]['is_bus']    = 0

# bus_id = 0
# bus_sizes = [8, 16, 32, 64]

# for sig, nodes in sorted(bus_candidates.items(), key=lambda x: -len(x[1])):
#     w = len(nodes)
#     if w in bus_sizes:
#         for node_id in nodes:
#             G.nodes[node_id]['bus_id']    = bus_id
#             G.nodes[node_id]['bus_width'] = w
#             G.nodes[node_id]['is_bus']    = 1
#         print(f"Bus {bus_id:3d} | width={w}")
#         bus_id += 1

# # === SANITY CHECK ===
# bus_nodes    = [(nid, d) for nid, d in G.nodes(data=True) if d.get('is_bus') == 1]
# all_boundary = [(nid, d) for nid, d in G.nodes(data=True) if d.get('boundary', 0) == 1]

# boundary_1 = sum(1 for _, d in bus_nodes if d.get('boundary', 0) == 1)
# boundary_0 = sum(1 for _, d in bus_nodes if d.get('boundary', 0) == 0)

# print("\n=== SANITY CHECK ===")
# print(f"Total bus nodes:             {len(bus_nodes)}")
# print(f"  boundary=1 (expected):     {boundary_1}")
# print(f"  boundary=0 (UNEXPECTED):   {boundary_0}")

# if boundary_0 == 0:
#     print("✓ PASS: All bus nodes are boundary nodes")
# else:
#     print("✗ FAIL: Some bus nodes are NOT boundary nodes!")
#     print(f"    {boundary_1} ({100*boundary_1/len(bus_nodes):.1f}%) are true boundary nodes")
#     print(f"    {boundary_0} ({100*boundary_0/len(bus_nodes):.1f}%) are NOT boundary nodes")

#     # for nid, d in bus_nodes:
#         # if d.get('boundary', 0) == 0:
#             # print(f"    node {nid} | label={d.get('label_copy')} | bus_id={d.get('bus_id')}")

# boundary_not_in_bus = sum(1 for _, d in all_boundary if d.get('is_bus') == 0)
# print(f"\nTotal boundary=1 nodes:      {len(all_boundary)}")
# print(f"  in a bus:                   {boundary_1}")
# print(f"  NOT in any bus (ungrouped): {boundary_not_in_bus}")
# print(f"  bus coverage:               {100*boundary_1/len(all_boundary):.1f}%")
# print(f"\nTotal buses found:           {bus_id}")
# print(f"Total bus nodes:             {sum(1 for _, d in G.nodes(data=True) if d.get('is_bus') == 1)}")

# nx.write_gml(G, 'encoded_bus_graph_no_partition.gml')
# print("\nSaved to encoded_bus_graph_no_partition.gml")