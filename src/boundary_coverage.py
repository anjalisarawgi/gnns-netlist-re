import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load graph
G = nx.read_gml("results/aes_to_des/aes_cipher_top_com_predictions.gml")

# Extract labels
y_true = [1 if G.nodes[n].get("true_label") == "boundary" else 0 for n in G.nodes()]
y_pred = [1 if G.nodes[n].get("predicted_label") == "boundary" else 0 for n in G.nodes()]

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Coverage check
true_boundary_nodes = {n for n in G.nodes() if G.nodes[n].get("true_label") == "boundary"}
pred_boundary_nodes = {n for n in G.nodes() if G.nodes[n].get("predicted_label") == "boundary"}
all_covered = true_boundary_nodes.issubset(pred_boundary_nodes)
covered_count = len(true_boundary_nodes & pred_boundary_nodes)
coverage_ratio = covered_count / len(true_boundary_nodes) if true_boundary_nodes else 0

# Print everything neatly
print(f"--- Boundary Detection Summary ---")
print(f"Total nodes: {len(G.nodes())}")
print(f"True boundary nodes: {len(true_boundary_nodes)}")
print(f"Predicted boundary nodes: {len(pred_boundary_nodes)}")
print()
print(f"True Positives (correctly detected boundaries): {tp}")
print(f"False Negatives (missed boundaries): {fn}")
print(f"False Positives (non-boundaries predicted as boundary): {fp}")
print(f"True Negatives (correctly predicted non-boundaries): {tn}")
print()
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print()
print(f"Coverage ratio: {coverage_ratio:.3f} ({covered_count}/{len(true_boundary_nodes)})")
print(f"All true boundary nodes covered? {all_covered}")