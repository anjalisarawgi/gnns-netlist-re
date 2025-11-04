import networkx as nx
from collections import Counter
import os

# === CONFIG ===
graph_path = "graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi_test6_comm.gml"
target_subcircuit = "31"
output_suffix = "_focus31_combined.gml"

# === LOAD GRAPH ===
G = nx.read_gml(graph_path)
print(f"Loaded graph with {G.number_of_nodes()} nodes")

# === FIND DOMINANT COMMUNITY FOR THIS SUBCIRCUIT ===
sub_nodes = [n for n in G.nodes() if str(G.nodes[n].get("subcircuit_original")) == target_subcircuit]
comm_ids = [str(G.nodes[n]["community_id"]) for n in sub_nodes]
dominant_comm = Counter(comm_ids).most_common(1)[0][0]
print(f"Subcircuit {target_subcircuit} mostly lies in community {dominant_comm}")

# === ADD ANNOTATION FIELDS ===
for n in G.nodes():
    sid = str(G.nodes[n].get("subcircuit_original", "unknown"))
    cid = str(G.nodes[n].get("community_id", "unknown"))

    # --- binary annotations for Gephi filtering ---
    G.nodes[n]["true_subcircuit_flag"] = 1 if sid == target_subcircuit else 0
    G.nodes[n]["community_flag"] = 1 if cid == dominant_comm else 0

    # --- categorical annotation for quick color view ---
    if sid == target_subcircuit and cid == dominant_comm:
        G.nodes[n]["focus_type"] = "overlap"          # both true & detected
        G.nodes[n]["viz_color"] = "#4daf4a"           # green
    elif sid == target_subcircuit:
        G.nodes[n]["focus_type"] = "true_subcircuit"  # only true
        G.nodes[n]["viz_color"] = "#e41a1c"           # red
    elif cid == dominant_comm:
        G.nodes[n]["focus_type"] = "detected_comm"    # only community
        G.nodes[n]["viz_color"] = "#377eb8"           # blue
    else:
        G.nodes[n]["focus_type"] = "other"            # neither
        G.nodes[n]["viz_color"] = "#cccccc"           # grey

# === SAVE GML ===
out_path = graph_path.replace(".gml", output_suffix)
nx.write_gml(G, out_path)
print(f"Saved combined-annotated graph to: {os.path.abspath(out_path)}")

# === SUMMARY ===
num_true = sum(nx.get_node_attributes(G, "true_subcircuit_flag").values())
num_comm = sum(nx.get_node_attributes(G, "community_flag").values())
num_overlap = sum(1 for n in G.nodes()
                  if G.nodes[n]["true_subcircuit_flag"] == 1 and G.nodes[n]["community_flag"] == 1)

overlap_ratio = num_overlap / num_true if num_true > 0 else 0
print(f"\nSummary for subcircuit {target_subcircuit}:")
print(f"  Nodes in true subcircuit     : {num_true}")
print(f"  Nodes in dominant community  : {num_comm}")
print(f"  Nodes in overlap (both)      : {num_overlap}")
print(f"  Overlap fraction             : {overlap_ratio:.3f}")
