# compare_communities_to_true_subcircuits.py
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd
import os

# ---------- CONFIG ----------
graph_path = "results/aes_to_des/aes_key_expand_128_gephi_test6_predictions_comm.gml"
subcircuit_key = "subcircuit_original"   # true module id
community_key = "community_id"           # unsupervised cluster id
# --------------------------------

# ---------- LOAD ----------
G = nx.read_gml(graph_path)
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------- EXTRACT LABELS ----------
nodes = list(G.nodes())
true_labels = [str(G.nodes[n].get(subcircuit_key, "unknown")) for n in nodes]
comm_labels = [str(G.nodes[n].get(community_key, "unknown")) for n in nodes]

num_true = len(set(true_labels))
num_comms = len(set(comm_labels))
print(f"True subcircuits: {num_true} | Detected communities: {num_comms}")

# ---------- GLOBAL METRIC ----------
if len(set(true_labels)) > 1 and len(set(comm_labels)) > 1:
    ami = adjusted_mutual_info_score(true_labels, comm_labels)
else:
    ami = np.nan
print(f"Adjusted Mutual Information (AMI): {ami:.3f}")

# ---------- PER-COMMUNITY MATCH TABLE ----------
comm_to_true = defaultdict(list)
for n in nodes:
    cid = str(G.nodes[n].get(community_key, "unknown"))
    sid = str(G.nodes[n].get(subcircuit_key, "unknown"))
    comm_to_true[cid].append(sid)

rows = []
for cid, sids in comm_to_true.items():
    counts = Counter(sids)
    dominant_sid, dominant_count = counts.most_common(1)[0]
    purity = dominant_count / len(sids)

    # --- find dominant subcircuit type ---
    # sample any node with this subcircuit ID to read its name / type
    sample_node = next((n for n in G.nodes() 
                        if str(G.nodes[n].get(subcircuit_key)) == dominant_sid), None)
    if sample_node is not None:
        sub_name = G.nodes[sample_node].get("subcircuit_name", "")
        true_label = G.nodes[sample_node].get("true_label", "")
        # prefer true_label if available, else fallback to name
        dominant_type = true_label if true_label else sub_name
    else:
        dominant_type = "unknown"

    # --- list all subcircuits in this community ---
    sorted_subs = [f"{sid}({cnt})" for sid, cnt in counts.most_common()]
    subs_summary = ", ".join(sorted_subs)

    rows.append({
        "community_id": cid,
        "size": len(sids),
        "dominant_subcircuit": dominant_sid,
        "dominant_type": dominant_type,    # <--- NEW COLUMN
        "purity": round(purity, 3),
        "num_true_subcircuits_in_comm": len(counts),
        "subcircuits_in_comm": subs_summary
    })

df = pd.DataFrame(rows).sort_values("purity", ascending=False)
avg_purity = df["purity"].mean()
print(f"Average community purity: {avg_purity:.3f}")

# ---------- ONE-TO-ONE MATCH CHECK ----------
# each true subcircuit can appear in multiple communities;
# we count how many communities map uniquely 1-to-1 to one true subcircuit.
dominant_pairs = df.groupby("dominant_subcircuit")["community_id"].nunique()
one_to_one_matches = (dominant_pairs == 1).sum()
print(f"Unique one-to-one matches: {one_to_one_matches} / {num_true}")

# ---------- SAVE SUMMARY ----------
out_path = graph_path.replace(".gml", "_community_vs_subcircuit_summary.csv")
df.to_csv(out_path, index=False)
print(f"Saved detailed table to:\n{os.path.abspath(out_path)}")

# ---------- PRINT TOP FEW ----------
print("\nTop 10 communities by purity:")
# print(df.head(10).to_string(index=False))
print(df.to_string(index=False))


# === GLOBAL ACCURACY: compare dominant community mapping vs true subcircuit ===

# Step 1. Build mapping: community_id -> dominant subcircuit
comm_to_true = {}
for n in G.nodes():
    cid = str(G.nodes[n]["community_id"])
    sid = str(G.nodes[n].get("subcircuit_original", "unknown"))
    comm_to_true.setdefault(cid, []).append(sid)

comm_dominant = {}
for cid, sids in comm_to_true.items():
    dominant_sid = Counter(sids).most_common(1)[0][0]
    comm_dominant[cid] = dominant_sid

# Step 2. Compute how many nodes match their community's dominant true subcircuit
correct = 0
for n in G.nodes():
    cid = str(G.nodes[n]["community_id"])
    true_sid = str(G.nodes[n].get("subcircuit_original", "unknown"))
    predicted_sid = comm_dominant.get(cid, "unknown")
    if predicted_sid == true_sid:
        correct += 1

total = G.number_of_nodes()
global_acc = correct / total if total > 0 else 0

print(f"\n=== GLOBAL COMMUNITY ACCURACY ===")
print(f"  Total nodes: {total}")
print(f"  Correctly matched to dominant subcircuit: {correct}")
print(f"  Global community accuracy: {global_acc:.3f}")
