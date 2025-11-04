import networkx as nx
import community as community_louvain
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
graph_path = "results/aes_to_des/aes_cipher_top_gephi_test6_predictions.gml"
save_graph = True       # whether to save annotated GML
draw_graphs = False     # set True if you want visualization
resolution = 1.0        # Louvain resolution parameter
# ----------------------------

# ---------- LOAD GRAPH ----------
G = nx.read_gml(graph_path)
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Louvain needs undirected graph
G_undirected = G.to_undirected()

# ---------- COMMUNITY DETECTION ----------
partition = community_louvain.best_partition(G_undirected, resolution=resolution)
num_comms = len(set(partition.values()))
print(f"Detected {num_comms} communities (resolution={resolution})")

# Add community IDs back to original (directed) graph
nx.set_node_attributes(G, partition, "community_id")

# -----------------------------------------------------
# Compare community ID vs true subcircuit label per node
# -----------------------------------------------------
mistakes = 0
for n in G.nodes():
    true_label = str(G.nodes[n].get("subcircuit_original", "unknown"))
    comm_id = str(G.nodes[n]["community_id"])
    G.nodes[n]["misclassified"] = 0  # default

    # Find dominant label of this community
    # (we already computed these earlier, but recompute here for clarity)
    # or load from precomputed dictionary if you prefer speed
    # For now, we'll mark as misclassified if the node label doesn't match
    # the dominant label of its community.

# Step 1: build mapping from community -> dominant true label
from collections import Counter, defaultdict

community_to_labels = defaultdict(list)
for n in G.nodes():
    cid = G.nodes[n]["community_id"]
    lbl = str(G.nodes[n].get("subcircuit_original", "unknown"))
    community_to_labels[cid].append(lbl)

community_dominant = {}
for cid, labels in community_to_labels.items():
    most_common_label, _ = Counter(labels).most_common(1)[0]
    community_dominant[cid] = most_common_label

# Step 2: mark misclassified nodes
for n in G.nodes():
    cid = G.nodes[n]["community_id"]
    lbl = str(G.nodes[n].get("subcircuit_original", "unknown"))
    if lbl != community_dominant[cid]:
        G.nodes[n]["misclassified"] = 1
        mistakes += 1

print(f"Total misclassified nodes: {mistakes} ({mistakes / G.number_of_nodes():.2%})")


# ---------- COMPUTE METRICS ----------
labels_true = []
labels_pred = []

for n in G.nodes():
    if "label" in G.nodes[n] and "community_id" in G.nodes[n]:
        labels_true.append(G.nodes[n]["label"])
        labels_pred.append(G.nodes[n]["community_id"])

if len(set(labels_true)) > 1:
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
else:
    ami = np.nan
print(f"Adjusted Mutual Information (AMI): {ami:.3f}")

# ---------- COMMUNITY PURITY ----------
community_to_labels = defaultdict(list)
for n in G.nodes():
    cid = G.nodes[n]["community_id"]
    # lbl = G.nodes[n].get("label", "unknown")
    lbl = G.nodes[n].get("subcircuit_original", "unknown")
    community_to_labels[cid].append(lbl)

purities = []
for cid, labels in community_to_labels.items():
    most_common_label, count = Counter(labels).most_common(1)[0]
    purity = count / len(labels)
    purities.append(purity)
    print(f"Community {cid:3d}: size={len(labels):5d}, purity={purity:.2f}, dominant={most_common_label}")

avg_purity = np.mean(purities)
print(f"\nAverage community purity: {avg_purity:.3f}")

# ---------- VISUALIZATION ----------
if draw_graphs:
    plt.figure(figsize=(10,8))
    nx.draw_spring(
        G_undirected,
        node_color=[G.nodes[n]['community_id'] for n in G.nodes()],
        node_size=10,
        with_labels=False
    )
    plt.title("Detected Communities (Louvain)")
    plt.show()

# ---------- SAVE ----------
if save_graph:
    out_path = graph_path.replace(".gml", "_comm.gml")
    nx.write_gml(G, out_path)
    print(f"Saved community-annotated graph to: {os.path.abspath(out_path)}")

