import torch
import os
import random
import numpy as np
import networkx as nx
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler

import wandb

from main import (
    run_training,
    evaluate_binary,
    set_seed,
    save_predictions_to_gml
)
from preprocessing import normalize_features
from gnn.gat import gat
from glob import glob

def load_aisec_multiple_gmls(gml_paths, label_type="subcircuit", binary_label=False):
    print(f"[DEBUG] Loading {len(gml_paths)} GMLs and combining...")

    all_features = []
    all_labels = []
    all_edges = []
    is_error_flags = [] 
    offset = 0

    for path in gml_paths:
        print(f"[DEBUG] Loading GML: {path}")
        G = nx.read_gml(path, label="id")
        nodes = list(G.nodes())
        node_map = {node: idx + offset for idx, node in enumerate(nodes)}

        for node in nodes:
            attr = G.nodes[node]
            feat = list(map(int, attr.get("features", [])))
            all_features.append(feat)

            if binary_label:
                label_name = attr.get("subcircuit_name", "unknown")
                all_labels.append(1 if label_name == "sbox" else 0)
            elif label_type == "subcircuit":
                all_labels.append(attr.get("subcircuit", -1))
            else:
                all_labels.append(attr.get("subcircuit_original", -1))

            is_error_flags.append(int(attr.get("is_error", 0)))

        for src, dst in G.edges():
            all_edges.append((node_map[src], node_map[dst]))

        offset += len(nodes)

    print(f"[DEBUG] Total nodes: {offset}, Total edges: {len(all_edges)}")
    print(f"[DEBUG] Label distribution: {Counter(all_labels)}")

    all_features = normalize_features(np.array(all_features))

    if binary_label:
        labels = torch.tensor(all_labels, dtype=torch.long)
        is_error_tensor = torch.tensor(is_error_flags, dtype=torch.long)
        id2label_global = {0: "not_sbox", 1: "sbox"}
        print("After binarization:", Counter(labels.tolist()))
    else:
        label_set = sorted(set(all_labels))
        label_map = {v: i for i, v in enumerate(label_set)}
        all_labels = [label_map[l] for l in all_labels]
        labels = torch.tensor(all_labels, dtype=torch.long)
        id2label_global = {i: l for l, i in label_map.items()}

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    num_nodes = len(all_features)

    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_cutoff = int(0.9 * num_nodes)
    print(f"[CHECK DEBUG]Total num_nodes = {num_nodes}")
    print("[CHECK DEBUG] train_cutoff ", train_cutoff)
    print("[CHECK DEBUG]int(0.2 * num_nodes)", int(0.05 * num_nodes))
    val_cutoff = train_cutoff + int(0.05 * num_nodes)
    print("[CHECK DEBUG] train_cutoff ", val_cutoff)

    train_indices = set(indices[:train_cutoff])
    val_indices = set(indices[train_cutoff:val_cutoff])
    test_indices = set(indices[val_cutoff:])

    print("[CHECK DEBUG] Overlap between train and val:", len(train_indices & val_indices))
    print("[CHECK DEBUG] Overlap between train and test:", len(train_indices & test_indices))
    print("[CHECK DEBUG] Overlap between val and test:", len(val_indices & test_indices))


    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_cutoff]] = True
    val_mask[indices[train_cutoff:val_cutoff]] = True
    test_mask[indices[val_cutoff:]] = True


    print(f"[DEBUG] Mask sizes — Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    data = Data(
        x=torch.tensor(all_features, dtype=torch.float),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask, 
        is_error=is_error_tensor 
    )

    return data, id2label_global

def save_predictions_to_gml(original_gml_path, data, model, id2name, output_gml_path="gml_results.gml"):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # ✅ grab is_error from data (if present)
    is_error_tensor = getattr(data, "is_error", None)

    # Load original GML graph with attributes
    G = nx.read_gml(original_gml_path, label="id")
    node_map = {node: idx for idx, node in enumerate(G.nodes())}

    for node in G.nodes():
        idx = node_map[node]
        true_id = data.y[idx].item()
        pred_id = pred[idx].item()

        true_label = id2name[true_id]
        pred_label = id2name[pred_id]
        correct = (true_id == pred_id)

        G.nodes[node]["true_label"] = true_label
        G.nodes[node]["predicted_label"] = pred_label
        G.nodes[node]["correct"] = correct

        # ✅ save is_error to GML if we have it
        if is_error_tensor is not None:
            G.nodes[node]["is_error"] = int(is_error_tensor[idx].item())

        # ✅ also keep split info
        if data.train_mask[idx]:
            G.nodes[node]["split"] = "train"
        elif data.val_mask[idx]:
            G.nodes[node]["split"] = "val"
        elif data.test_mask[idx]:
            G.nodes[node]["split"] = "test"
        else:
            G.nodes[node]["split"] = "unknown"

    nx.write_gml(G, output_gml_path)
    print(f"Saved GML with predictions to: {output_gml_path}")

if __name__ == "__main__":
    wandb.init(project="gnn-subcircuit-detection", name="aes_errors_to_mixed_eval")
    set_seed(42)

    train_paths = [
        "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml",
        "graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_randomError_1.gml",
        "graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_layoutError_1.gml",
        "graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_speckledError_1.gml",
    ]

    train_data, id2label = load_aisec_multiple_gmls(train_paths, binary_label=True)

    print("[DEBUG] Creating GraphSAINT sampler...")
    train_loader = GraphSAINTRandomWalkSampler(
        train_data,
        batch_size=int(0.3 * train_data.num_nodes),
        walk_length=5,
        shuffle=True,
    )

    for i, batch in enumerate(train_loader):
        print(f"\n[DEBUG] === Sampled Batch {i+1} ===")
        print(f"  Nodes in batch: {batch.num_nodes}")
        print(f"  Edges in batch: {batch.num_edges}")
        print(f"  Label counts: {Counter(batch.y.tolist())}")
        if hasattr(batch, 'n_id'):
            print(f"  Sample node IDs: {batch.n_id[:5].tolist()}...")
        if i >= 1:
            break

    model = run_training(
        data=train_data,
        train_loader=train_loader,
        in_dim=train_data.num_features,
        out_dim=2,
        id2name=id2label,
        model_name="graphsage",
        use_weighted_loss=True,
    )

    def run_eval(gml_path, tag):
        print(f"[DEBUG] Evaluating on: {gml_path}")
        G_data, _ = load_aisec_multiple_gmls([gml_path], binary_label=True)
        mask = torch.ones_like(G_data.y, dtype=torch.bool)

        model.eval()
        out = model(G_data.x, G_data.edge_index)
        pred = out.argmax(dim=1)
        labels = G_data.y

        correct = (pred == labels).sum().item()
        accuracy = correct / len(labels)

        f1, precision, recall = evaluate_binary(model, G_data, mask)
        print(f"\n=== Eval on {tag} ===")
        print(f"Accuracy = {accuracy:.4f}")
        print(f"F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

        # ✅ check misclassified error nodes
        if hasattr(G_data, "is_error"):
            misclassified = (pred != labels)
            is_error_mask = (G_data.is_error == 1)
            misclassified_error_nodes = (misclassified & is_error_mask).sum().item()
            total_error_nodes = is_error_mask.sum().item()

            print(f"Total is_error nodes: {total_error_nodes}")
            print(f"Misclassified is_error nodes: {misclassified_error_nodes}")

        save_predictions_to_gml(
            original_gml_path=gml_path,
            data=G_data,
            model=model,
            id2name=id2label,
            output_gml_path=f"{tag}_predictions.gml"
        )

    run_eval("graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_mixedError_1.gml", "MixedError_1")
    run_eval("graphs/processed/aes_encryption_latest/osu035/synth/1_perc/aes_cipher_top_gephi_mixedError_2.gml", "MixedError_2")

