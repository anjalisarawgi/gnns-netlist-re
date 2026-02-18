# # import yaml

# # INPUT_YAML  = "config/train_new_graphs_crypto_v4.yml"          # your original YAML
# # OUTPUT_YAML = "config/train_new_graphs_crypto_v4.yml"  # filtered YAML

# # with open(INPUT_YAML, "r") as f:
# #     cfg = yaml.safe_load(f)

# # def keep_only(paths, tag):
# #     if paths is None:
# #         return paths
# #     return [p for p in paths if tag in p]

# # # filter rules
# # cfg["train_gml"] = keep_only(cfg.get("train_gml"), "_m2.gml")
# # cfg["val_gml"]   = keep_only(cfg.get("val_gml"), "_m1.gml")
# # cfg["test_gml"]  = keep_only(cfg.get("test_gml"), "_m1.gml")

# # with open(OUTPUT_YAML, "w") as f:
# #     yaml.safe_dump(cfg, f, sort_keys=False)

# # print("Saved filtered config to:", OUTPUT_YAML)
# # print("Train graphs:", len(cfg["train_gml"]))
# # print("Val graphs:", len(cfg.get("val_gml", [])))
# # print("Test graphs:", len(cfg.get("test_gml", [])))


# import json

# with open("graphs/processed_jan27_m1/usable_graphs.json", "r") as f:
#     files = json.load(f)

# filtered = [p for p in files if p.endswith("_m1.gml")]

# with open("graphs/processed_jan27_m1/usable_graphs_m1.json", "w") as f:
#     json.dump(filtered, f, indent=2)
import torch
import joblib
import numpy as np
import networkx as nx
import json
import os

from sklearn.metrics import f1_score
from torch_geometric.data import Data
from gnn.gat import gat
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.graphTransformer import GraphTransformer
from sklearn.metrics import f1_score, precision_score, recall_score

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
TARGET_RATIO = 0.15   # adjust if needed
FIXED_THRESHOLD = 0.5


# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------
def load_single_gml(gml_path):
    G = nx.read_gml(gml_path, label="id")

    features = []
    labels = []

    for node in G.nodes():
        attr = G.nodes[node]
        feat = attr.get("features", [])
        partition_feat = attr.get("partition_features", [0.0, 0.0])
        feat = list(feat) + list(partition_feat)
        features.append(feat)

        boundary_value = attr.get("boundary", 0)
        labels.append(int(boundary_value) if boundary_value != -1 else 0)

    labels = torch.tensor(labels, dtype=torch.long)

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[s], node_map[d]) for s, d in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_attrs = []
    for s, d in G.edges():
        edge_attrs.append(G[s][d].get("edge_features", [0, 0, 0, 0]))

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    features = torch.tensor(np.array(features, dtype=np.float32))

    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
def load_model(model_path, metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)

    in_dim = meta["final_feature_dim"]
    model_type = meta["model_type"]

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


@torch.no_grad()
def get_probs(model, data):
    out = model(data.x, data.edge_index, data.edge_attr)
    return torch.softmax(out, dim=1)[:, 1].cpu().numpy()


# ------------------------------------------------------------
# Decision rules
# ------------------------------------------------------------
def predict_fixed(probs, threshold):
    return (probs >= threshold).astype(int)


def predict_topk(probs, ratio):
    n = len(probs)
    k = max(1, int(round(ratio * n)))
    idx = np.argsort(probs)[::-1]
    preds = np.zeros(n, dtype=int)
    preds[idx[:k]] = 1
    return preds


def best_f1_oracle(probs, labels):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_t = 0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_f1, best_t


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
        # "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_inv_cipher_top_combined_m1.gml",
        # "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/nangate/aes_key_expand_128_combined_m1.gml",
        # "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/sha1-master/osu035/sha1_core_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/des_latest/gscl45nm/des_combined_m1.gml",
        # "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_key_expand_128_combined_m1.gml",
        # "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml",
        # "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/trigonometric_functions_in_double_fpu_latest/gscl45nm/top_combined_m1.gml",
        # "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/mips32r1_latest/gscl45nm/ALU_combined_m1.gml",
    ]

    model = load_model(model_path, metadata_path)
    scaler = joblib.load(scaler_path)

    results = []

    for gml_path in test_graphs:

        print("\nProcessing:", gml_path)

        data = load_single_gml(gml_path)
        data.x = torch.tensor(
            scaler.transform(data.x.cpu().numpy()),
            dtype=torch.float32
        )

        probs = get_probs(model, data)
        labels = data.y.cpu().numpy()

        # Fixed threshold
        preds_fixed = predict_fixed(probs, FIXED_THRESHOLD)
        f1_fixed = f1_score(labels, preds_fixed, zero_division=0)
        prec_fixed = precision_score(labels, preds_fixed, zero_division=0)
        rec_fixed = recall_score(labels, preds_fixed, zero_division=0)

        # Ranking-based
        preds_topk = predict_topk(probs, TARGET_RATIO)
        f1_topk = f1_score(labels, preds_topk, zero_division=0)
        prec_topk = precision_score(labels, preds_topk, zero_division=0)
        rec_topk = recall_score(labels, preds_topk, zero_division=0)

        # Oracle
        f1_oracle, best_t = best_f1_oracle(probs, labels)
        preds_oracle = (probs >= best_t).astype(int)
        prec_oracle = precision_score(labels, preds_oracle, zero_division=0)
        rec_oracle = recall_score(labels, preds_oracle, zero_division=0)


        print(f"\nFixed (0.5)  -> F1: {f1_fixed:.4f} | P: {prec_fixed:.4f} | R: {rec_fixed:.4f}")
        print(f"TopK ({TARGET_RATIO}) -> F1: {f1_topk:.4f} | P: {prec_topk:.4f} | R: {rec_topk:.4f}")
        print(f"Oracle       -> F1: {f1_oracle:.4f} | P: {prec_oracle:.4f} | R: {rec_oracle:.4f} @ t={best_t:.2f}")

        results.append({
            "graph": gml_path,
            "f1_fixed": f1_fixed,
            "prec_fixed": prec_fixed,
            "rec_fixed": rec_fixed,
            "f1_topk": f1_topk,
            "prec_topk": prec_topk,
            "rec_topk": rec_topk,
            "f1_oracle": f1_oracle,
            "prec_oracle": prec_oracle,
            "rec_oracle": rec_oracle,
        })

    # Save CSV
    import csv
    with open("ranking_eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved ranking_eval_results.csv")