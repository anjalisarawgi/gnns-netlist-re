import argparse
import torch
import os
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from preprocessing import normalize_features
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
from gnn.graphTransformer import GraphTransformer
from torch_geometric.data import Data



# ---------------------------------------------------------------------
# Load GML → Data object (same logic as your training script)
# ---------------------------------------------------------------------
def load_aisec_single_gml(gml_path, binary_label=True, label_mode="subcircuit_name"):
    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node]

        feat = attr.get("features", [])
        if isinstance(feat, (list, tuple, np.ndarray)):
            features.append(feat)
        else:
            features.append([])

        if binary_label:
            if label_mode == "boundary":
                # numeric boundary 0/1
                boundary = attr.get("boundary", 0)
                try:
                    labels.append(int(boundary))
                except:
                    labels.append(0)
            else:
                # sbox vs not_sbox
                label_name = attr.get("subcircuit_name", "unknown")
                labels.append(1 if label_name == "sbox" else 0)
        else:
            raise ValueError("Only binary mode supported in evaluation script.")

    features = normalize_features(np.array(features, dtype=np.float32))
    labels = torch.tensor(labels, dtype=torch.long)

    edges = []
    node_map = {node: i for i, node in enumerate(nodes)}
    for u, v in G.edges():
        edges.append([node_map[u], node_map[v]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # masks = all nodes are test nodes
    num_nodes = len(nodes)
    mask = torch.ones(num_nodes, dtype=torch.bool)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=labels,
        train_mask=mask,
        val_mask=mask,
        test_mask=mask
    )

    if label_mode == "boundary":
        id2label = {0: "not_boundary", 1: "boundary"}
    else:
        id2label = {0: "not_sbox", 1: "sbox"}

    return data, id2label



# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------
def load_model(model_type, in_dim, out_dim, model_path):
    if model_type == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_type == "gcn":
        model = GCN(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_type == "gat":
        model = gat(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_type == "graphTransformer":
        model = GraphTransformer(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    else:
        raise ValueError("Unknown model type")

    state = torch.load(model_path, map_location="cpu")

    fixed_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            fixed_state[k.replace("_orig_mod.", "")] = v
        else:
            fixed_state[k] = v

    model.load_state_dict(fixed_state)
    model.eval()

    return model
# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data):
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()
    y_true = data.y.cpu().numpy()

    f1 = f1_score(y_true, pred, zero_division=0)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    acc = (pred == y_true).sum() / len(y_true)

    mask1 = y_true == 1
    mask0 = y_true == 0

    acc1 = (pred[mask1] == y_true[mask1]).sum() / max(mask1.sum(), 1)
    acc0 = (pred[mask0] == y_true[mask0]).sum() / max(mask0.sum(), 1)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "total_acc": acc,
        "class1_acc": acc1,
        "class0_acc": acc0,
        "y_pred": pred
    }



# ---------------------------------------------------------------------
# Save predictions back into GML
# ---------------------------------------------------------------------
def save_predictions_to_gml(original_gml_path, id2name, preds, out_path):
    G = nx.read_gml(original_gml_path, label="id")
    nodes = list(G.nodes())

    for i, node in enumerate(nodes):
        G.nodes[node]["pred_label"] = id2name[int(preds[i])]
        G.nodes[node]["pred_label_id"] = int(preds[i])

    nx.write_gml(G, out_path)
    print(f"[INFO] Saved prediction GML → {out_path}")



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True,
                        choices=["graphsage", "gcn", "gat", "graphTransformer"])

    parser.add_argument("--test_gml", nargs="+", required=True)
    parser.add_argument("--label_mode", default="subcircuit_name",
                        choices=["subcircuit_name", "boundary"])

    args = parser.parse_args()

    for gml_path in args.test_gml:
        print(f"\n=== Evaluating {gml_path} ===")

        # Load graph
        data, id2label = load_aisec_single_gml(
            gml_path, binary_label=True, label_mode=args.label_mode
        )

        in_dim = data.num_features
        out_dim = 2

        # Load trained model
        model = load_model(
            model_type=args.model_type,
            in_dim=in_dim,
            out_dim=out_dim,
            model_path=args.model_path
        )

        # Evaluate
        metrics = evaluate(model, data)
        print(metrics)

        # Save GML
        base = os.path.splitext(os.path.basename(gml_path))[0]
        out_file = f"{base}_predictions.gml"

        save_predictions_to_gml(
            original_gml_path=gml_path,
            id2name=id2label,
            preds=metrics["y_pred"],
            out_path=out_file
        )