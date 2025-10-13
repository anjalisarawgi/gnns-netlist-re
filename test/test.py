import networkx as nx

G = nx.read_gml("../graphs/processed/des_latest/osu035/des_gephi.gml")
G = nx.DiGraph(G)

output_nodes = [n for n in G.nodes if G.out_degree(n) > 3]

input_nodes = set()
for n in output_nodes:
    input_nodes.update(G.successors(n))

for n in G.nodes:
    if n in output_nodes:
        G.nodes[n]["candidate"] = "output"
    # elif n in input_nodes:
    #     G.nodes[n]["candidate"] = "input"
    else:
        G.nodes[n]["candidate"] = "none"

nx.write_gml(G, "candidates.gml")

print(f"Found {len(output_nodes)} outputs and {len(input_nodes)} inputs")


# # ##############################


# import networkx as nx

# G = nx.read_gml("../graphs/processed/des_latest/osu035/des_gephi.gml")
# G = nx.DiGraph(G)

# fanout_threshold = 3

# outputs = [n for n in G.nodes if G.out_degree(n) > fanout_threshold]
# partitions = {}
# for out_node in outputs:
#     inputs = list(G.successors(out_node))
#     partitions[out_node] = inputs

# for n in G.nodes:
#     if n in outputs:
#         G.nodes[n]["role"] = "output_partition"
#     elif any(n in ins for ins in partitions.values()):
#         G.nodes[n]["role"] = "input_partition"
#     else:
#         G.nodes[n]["role"] = "none"

# nx.write_gml(G, "chip_graph_partition_pairs.gml")
# print(f"Found {len(outputs)} output partitions")

import networkx as nx

G = nx.read_gml("../graphs/processed/des_latest/osu035/des_gephi.gml")
G = nx.DiGraph(G)

fanout_threshold = 3
outputs = [n for n in G.nodes if G.out_degree(n) > fanout_threshold]

partitions = {}
for out_node in outputs:
    inputs = list(G.successors(out_node))
    partitions[out_node] = inputs

for n in G.nodes:
    if n in outputs:
        G.nodes[n]["role"] = "output_partition"
    elif any(n in ins for ins in partitions.values()):
        G.nodes[n]["role"] = "input_partition"
    else:
        G.nodes[n]["role"] = "none"

nx.write_gml(G, "chip_graph_partition_pairs.gml")
print(f"Found {len(outputs)} output partitions")




import networkx as nx

G = nx.read_gml("chip_graph_partition_pairs.gml")
G = nx.DiGraph(G)

# Step 1: choose stricter threshold
fanout_threshold = 5
outputs = [n for n in G.nodes if G.out_degree(n) >= fanout_threshold]

# Step 2: build candidate groups (output + its cone)
groups = []
for out_node in outputs:
    members = set()
    members.add(out_node)
    members.update(nx.ancestors(G, out_node))
    members.update(G.successors(out_node))
    groups.append(members)

# Step 3: merge overlapping groups (>60% overlap)
merged = []
for g in groups:
    merged_flag = False
    for mg in merged:
        if len(g & mg) / min(len(g), len(mg)) > 0.6:
            mg |= g
            merged_flag = True
            break
    if not merged_flag:
        merged.append(g)

# Step 4: assign partition ids
partitions = {}
for i, grp in enumerate(merged, 1):
    for n in grp:
        partitions[n] = f"part_{i}"

for n in G.nodes:
    G.nodes[n]["partition"] = partitions.get(n, "none")

nx.write_gml(G, "chip_graph_partitions_merged.gml")

print(f"{len(merged)} merged partitions written")