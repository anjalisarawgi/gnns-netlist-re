import torch
import joblib
import numpy as np
import networkx as nx
import json
import os
import random

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from torch_geometric.data import Data
from gnn.gat import gat
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.graphTransformer import GraphTransformer
from collections import Counter


# ------------------------------------------------------------
# Load Graph
# ------------------------------------------------------------
def load_single_gml(
    gml_path,
    remove_edges=False,
    use_partition_features=False,
    use_graph_features=False
):
    print("[INFO] Calling gml from path:", gml_path)

    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node]
        feat = attr.get("features", [])

        if use_partition_features:
            partition_feat = attr.get("partition_features", [0.0, 0.0])
            feat = list(feat) + list(partition_feat)

        if use_graph_features:
            graph_feat = attr.get("graph_features", [0.0, 0.0])
            feat = list(feat) + list(graph_feat[:2])

        features.append(feat)

        boundary_value = attr.get("boundary", 0)
        try:
            label = int(boundary_value)
        except:
            label = 0

        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)
    labels[labels == -1] = 0

    # Build edge_index
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Edge features
    edge_attrs = []
    for src, dst in G.edges():
        attr = G[src][dst]
        e_feat = attr.get("edge_features", [0.0, 0.0, 0.0, 0.0])
        edge_attrs.append(e_feat)

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    features = torch.tensor(np.array(features, dtype=np.float32))

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )

    return data


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
def load_model(model_path, metadata_path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    in_dim = meta["final_feature_dim"]

    if model_type == "gat":
        model = gat(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "gcn":
        model = GCN(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "graphTransformer":
        model = GraphTransformer(in_channels=in_dim, hidden_channels=256, out_channels=2)
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# ------------------------------------------------------------
# Get probabilities
# ------------------------------------------------------------
@torch.no_grad()
def get_probs(model, data):
    out = model(data.x, data.edge_index, data.edge_attr)
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    labels = data.y.cpu().numpy()
    return probs, labels


# ------------------------------------------------------------
# Compute AUC metrics
# ------------------------------------------------------------
def compute_auc_metrics(probs, labels):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    return float(roc_auc), float(pr_auc)


# ------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------
def sweep_thresholds(probs, labels):
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        predicted_ratio = preds.mean()
        true_ratio = labels.mean()
        

        results.append({
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "predicted_ratio": float(predicted_ratio),
            "true_ratio": float(true_ratio),
        })

    return results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    model_dir = "models/wPartitions/fullgraph_per_design_ce_soft_500ep_for_aes_128_combined_m1+aes_inv_cipher_top_combined_m1+more"
    model_path = os.path.join(model_dir, "model.pt")
    metadata_path = os.path.join(model_dir, "metadata.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    test_graphs = [
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/tiny_aes_latest/osu035/aes_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_inv_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/nangate/aes_key_expand_128_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/sha1-master/osu035/sha1_core_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/des_latest/gscl45nm/des_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_key_expand_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/trigonometric_functions_in_double_fpu_latest/gscl45nm/top_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/mips32r1_latest/gscl45nm/ALU_combined_m1.gml",
    ]
    os.makedirs("test", exist_ok=True)

    model = load_model(model_path, metadata_path)
    scaler = joblib.load(scaler_path)

    for gml_path in test_graphs:

        print("\n======================================")
        print("Processing:", gml_path)
        print("======================================")

        data = load_single_gml(
            gml_path,
            use_partition_features=True,
            use_graph_features=False
        )

        # Apply scaler
        data.x = torch.tensor(
            scaler.transform(data.x.cpu().numpy()),
            dtype=torch.float32
        )

        probs, labels = get_probs(model, data)

        # ---- Compute AUC metrics ----
        roc_auc, pr_auc = compute_auc_metrics(probs, labels)
        mean_prob = np.mean(probs)
        p95_prob = np.percentile(probs, 95)
        print(f"[AUC] ROC-AUC: {roc_auc:.4f}")
        print(f"[AUC] PR-AUC (AP): {pr_auc:.4f}")
        print("Mean probability:", mean_prob)
        print("95th percentile probability:", p95_prob)


        # ---- Sweep thresholds ----
        results = sweep_thresholds(probs, labels)

        # ---- Save output ----
        parts = os.path.splitext(gml_path)[0].split(os.sep)
        graph_name = "__".join(parts[-3:])
        save_path = os.path.join("test", f"{graph_name}.json")

        mean_prob = float(np.mean(probs))
        p95_prob  = float(np.percentile(probs, 95))

        output = {
            "graph_name": graph_name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "results": results,
            "mean_prob": mean_prob,
            "p95_prob": p95_prob
        }

        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print("Saved:", save_path)

    print("\nAll threshold sweeps completed.")