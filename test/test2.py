import networkx as nx
import os
from collections import deque


# === CONFIG ===
input_file = "candidates.gml"  # change this to your file name
output_dir = "partitions_fixed"
attribute_name = "candidate"
green_label = "output"

# === Load graph ===
print(f"Loading {input_file}...")
G = nx.read_gml(input_file)
print(f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

os.makedirs(output_dir, exist_ok=True)

# === Identify green nodes ===
green_nodes = [n for n, d in G.nodes(data=True) if d.get(attribute_name) == green_label]
print(f"Found {len(green_nodes)} green nodes")

visited = set()
partitions = []
partition_id = 0

for green in green_nodes:
    if green in visited:
        continue

    partition_id += 1
    sub_nodes = set()
    queue = deque([green])
    local_visited = set([green])

    while queue:
        node = queue.popleft()
        sub_nodes.add(node)
        visited.add(node)

        for neighbor in G.neighbors(node):
            if neighbor not in local_visited:
                # Stop traversal when another green node is encountered
                if (
                    neighbor in green_nodes
                    and neighbor != green
                ):
                    continue
                queue.append(neighbor)
                local_visited.add(neighbor)

    subG = G.subgraph(sub_nodes).copy()
    partitions.append(subG)
    nx.write_gml(subG, os.path.join(output_dir, f"partition_{partition_id}.gml"))
    print(f"✅ partition_{partition_id}.gml -> {subG.number_of_nodes()} nodes")

print(f"\n✅ Done: {len(partitions)} partitions saved in '{output_dir}'")