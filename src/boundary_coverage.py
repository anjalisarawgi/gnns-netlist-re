import networkx as nx

G = nx.read_gml("results/aes_to_des/aes_cipher_top_com_predictions.gml")

y_true = [int(G.nodes[n].get("boundary", 0)) for n in G.nodes()]
y_pred = [1 if G.nodes[n].get("predicted_label") == "boundary" else 0 for n in G.nodes()]

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")