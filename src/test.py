import torch
import os
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from main import (
    run_training,
    evaluate_binary,
    set_seed,
    save_predictions_to_gml
)
import wandb
import random
import networkx as nx
from preprocessing import normalize_features
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from gnn.gat import gat


wandb.init(project="gnn-subcircuit-detection", name="aes_to_des_test")
# For reproducibility
set_seed(42)


def load_aisec_single_gml(gml_path, label_type="subcircuit", binary_label=False, positive_class=None, remove_edges=False):
    print("calling gnn from path:", gml_path)
    random.seed(42)

    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node]

        feat = list(map(int, attr.get("features", [])))
        features.append(feat)

        # === LABEL ASSIGNMENT ===
        if binary_label:
            # Use subcircuit_name directly
            label_name = attr.get("subcircuit_name", "unknown")
            if label_name == "sbox":
                labels.append(1)   # positive
            else:
                labels.append(0)   # negative
        elif label_type == "subcircuit":
            labels.append(attr.get("subcircuit", -1))
        else:
            labels.append(int(attr.get("subcircuit_original", -1)))

    # Normalize features
    features = normalize_features(np.array(features))

    # === ID2LABEL mapping ===
    if binary_label:
        id2label = {0: "not_sbox", 1: "sbox"}
        labels = torch.tensor(labels, dtype=torch.long)
    elif label_type == "subcircuit":
        label_set = sorted(set(labels))
        label_map = {v: i for i, v in enumerate(label_set)}
        labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
        id2label = {i: l for l, i in label_map.items()}
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        id2label = {int(l): str(l) for l in sorted(set(labels.tolist()))}

    # === Print distribution ===
    print(f"Total labels: {len(labels)}")
    print(f"Unique labels: {sorted(set(labels.tolist()))}")
    label_counts = Counter(labels.tolist())
    print("Label counts:", label_counts)

    # === Build edge_index ===
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # === Masks ===
    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_cutoff = int(0.6 * num_nodes)
    val_cutoff = train_cutoff + int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_cutoff]] = True
    val_mask[indices[train_cutoff:val_cutoff]] = True
    test_mask[indices[val_cutoff:]] = True

    # === Optionally remove cross-split edges ===
    if remove_edges:
        print("Removing cross-split edges for inductive setup")
        train_nodes = set(torch.where(train_mask)[0].tolist())
        val_nodes = set(torch.where(val_mask)[0].tolist())
        test_nodes = set(torch.where(test_mask)[0].tolist())

        new_edges = []
        for src, dst in edge_index.t().tolist():
            if (src in train_nodes and dst in train_nodes) or \
               (src in val_nodes and dst in val_nodes) or \
               (src in test_nodes and dst in test_nodes):
                new_edges.append([src, dst])

        edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    else:
        print("Keeping all edges")

    # === Final Data object ===
    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, id2label



# === Train on AES ===
aes_data, id2label = load_aisec_single_gml(
    gml_path="graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml",
    binary_label=True,   # sbox vs not_sbox
)
# Debugging: check AES label distribution
print("\n[DEBUG] AES dataset:")
print("id2label:", id2label)
print("Total nodes:", aes_data.num_nodes)
print("SBOX nodes:", (aes_data.y == 1).sum().item())
print("Not-SBOX nodes:", (aes_data.y == 0).sum().item())

# Double-check training nodes come only from AES
print("\n[DEBUG] AES train/val/test split:")
print("  Train nodes:", aes_data.train_mask.sum().item())
print("  Val nodes:", aes_data.val_mask.sum().item())
print("  Test nodes:", aes_data.test_mask.sum().item())

# Sanity: make sure all masks add up to total AES nodes
print("  Total AES nodes:", aes_data.num_nodes)
print("  Sum of masks   :", (aes_data.train_mask.sum() +
                              aes_data.val_mask.sum() +
                              aes_data.test_mask.sum()).item())

aes_loader = GraphSAINTRandomWalkSampler(
    aes_data,
    batch_size=int(0.3 * aes_data.num_nodes),
    walk_length=8,
    shuffle=True,
)

model = run_training(
    data=aes_data,
    train_loader=aes_loader,
    in_dim=aes_data.num_features,
    out_dim=2,   # binary classification
    id2name=id2label,
    model_name="gat",
    use_weighted_loss=True,
)

# === Test on DES ===
des_data, _ = load_aisec_single_gml(
    gml_path="graphs/processed/des_latest/osu035/des_gephi.gml",
    binary_label=True,   # same labeling logic
)

# Debugging: check DES label distribution
print("\n[DEBUG] DES dataset:")
print("Total nodes:", des_data.num_nodes)
print("SBOX nodes:", (des_data.y == 1).sum().item())
print("Not-SBOX nodes:", (des_data.y == 0).sum().item())

# Ensure no DES nodes are used for training
print("\n[DEBUG] DES dataset check (should have *no* training here):")
print("  Train nodes:", des_data.train_mask.sum().item())
print("  Val nodes:", des_data.val_mask.sum().item())
print("  Test nodes:", des_data.test_mask.sum().item())

# Use all DES nodes for testing
mask = torch.ones_like(des_data.y, dtype=torch.bool)

f1, precision, recall = evaluate_binary(model, des_data, mask)
print(f"\n=== Cross-graph test (AES→DES) ===")
print(f"F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

# === Per-class and total accuracy on DES ===
model.eval()
out = model(des_data.x, des_data.edge_index)
pred = out.argmax(dim=1)

y_true = des_data.y.cpu().numpy()
y_pred = pred.cpu().numpy()

total_correct = (y_true == y_pred).sum()
total_acc = total_correct / len(y_true)

sbox_mask = (y_true == 1)
not_sbox_mask = (y_true == 0)

sbox_acc = (y_pred[sbox_mask] == y_true[sbox_mask]).sum() / sbox_mask.sum()
not_sbox_acc = (y_pred[not_sbox_mask] == y_true[not_sbox_mask]).sum() / not_sbox_mask.sum()

print("\n=== DES Accuracy Breakdown ===")
print(f"Total Accuracy   : {total_acc:.4f}")
print(f"SBOX Accuracy    : {sbox_acc:.4f}")
print(f"Not-SBOX Accuracy: {not_sbox_acc:.4f}")

# Save DES graph with predictions
output_dir = "results/aes_to_des"
os.makedirs(output_dir, exist_ok=True)
print("\n[DEBUG] Saving DES predictions to results/aes_to_des/des_predictions.gml")
save_predictions_to_gml(
    original_gml_path="graphs/processed/des_latest/osu035/des_gephi.gml",
    data=des_data,
    model=model,
    id2name={0: "not_sbox", 1: "sbox"},
    output_gml_path=os.path.join(output_dir, "des_predictions.gml"),
)