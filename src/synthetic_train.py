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



def load_multiple_gmls(gml_paths, binary_label=True, remove_edges=False):
    all_x = []
    all_y = []
    all_edge_index = []
    all_train_mask = []
    all_val_mask = []
    all_test_mask = []
    offset = 0

    for path in gml_paths:
        data, _ = load_aisec_single_gml(
            gml_path=path,
            binary_label=binary_label,
            remove_edges=remove_edges,
        )

        all_x.append(data.x)
        all_y.append(data.y)

        # Offset edge indices
        edge_idx = data.edge_index + offset
        all_edge_index.append(edge_idx)

        # Adjust masks
        all_train_mask.append(data.train_mask)
        all_val_mask.append(data.val_mask)
        all_test_mask.append(data.test_mask)

        offset += data.num_nodes

    # Concatenate everything
    merged_data = Data(
        x=torch.cat(all_x, dim=0),
        edge_index=torch.cat(all_edge_index, dim=1),
        y=torch.cat(all_y, dim=0),
        train_mask=torch.cat(all_train_mask),
        val_mask=torch.cat(all_val_mask),
        test_mask=torch.cat(all_test_mask)
    )

    return merged_data


# Load training data from 3 AES graphs
train_graph_paths = [
    "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml",
    "graphs/processed/aes_encryption_latest/osu035/aes_mixcolumns_gephi.gml",
    "graphs/processed/aes_encryption_latest/osu035/aes_key_schedule_gephi.gml"
]
train_data = load_multiple_gmls(train_graph_paths, binary_label=True)

# Use GraphSAINT sampler
train_loader = GraphSAINTRandomWalkSampler(
    train_data,
    batch_size=int(0.3 * train_data.num_nodes),
    walk_length=5,
    shuffle=True,
)

# Train model
model = run_training(
    data=train_data,
    train_loader=train_loader,
    in_dim=train_data.num_features,
    out_dim=2,
    id2name={0: "not_sbox", 1: "sbox"},
    model_name="graphsage",
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