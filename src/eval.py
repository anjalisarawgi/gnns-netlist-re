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
import argparse
import json
import torch
import os


# same function as in train
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
                boundary = attr.get("boundary", 0)
                try:
                    labels.append(int(boundary))
                except:
                    labels.append(0)
            else:
                label_name = attr.get("subcircuit_name", "unknown")
                labels.append(1 if label_name == "sbox" else 0)

    features = normalize_features(np.array(features, dtype=np.float32))
    labels = torch.tensor(labels, dtype=torch.long)

    edges = []
    node_map = {node: i for i, node in enumerate(nodes)}
    for u, v in G.edges():
        edges.append([node_map[u], node_map[v]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

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

    return data, id2label, len(nodes), len(edges)




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



#saving
def save_predictions_to_gml(original_gml_path, id2name, preds, out_path):
    G = nx.read_gml(original_gml_path, label="id")
    nodes = list(G.nodes())

    for i, node in enumerate(nodes):
        G.nodes[node]["pred_label"] = id2name[int(preds[i])]
        G.nodes[node]["pred_label_id"] = int(preds[i])

    nx.write_gml(G, out_path)
    print(f"[INFO] Saved prediction GML → {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_type",
        default="gat",
        choices=["graphsage", "gcn", "gat", "graphTransformer"]
    )

    # Not needed anymore if using JSON, but keep it optional
    parser.add_argument("--test_gml", nargs="+", default=None)

    parser.add_argument(
        "--test_files",
        type=str,
        default=None,
        help="Path to JSON file containing { 'test_graphs': [...] }"
    )

    parser.add_argument(
        "--label_mode",
        default="boundary",
        choices=["subcircuit_name", "boundary"]
    )

    args = parser.parse_args()


    #  folder name = model_path
    model_name = args.model_path

    results_dir = os.path.join("results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "eval_results.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("graph,num_nodes,num_edges,f1,precision,recall,total_acc,class1_acc,class0_acc\n")

    # text graphs segment
    test_graphs = []
    if args.test_files:
        with open(args.test_files, "r") as f:
            config = json.load(f)
        if "test_graphs" not in config:
            raise ValueError("missing test_graphs")
        test_graphs = config["test_graphs"]

    if args.test_gml:
        test_graphs.extend(args.test_gml)

    if len(test_graphs) == 0:
        raise ValueError("missing test graphs")

    # evaluate one graph at a time
    for gml_path in test_graphs:
        print(f"Evaluating {gml_path}")

        data, id2label, num_nodes, num_edges = load_aisec_single_gml(
            gml_path,
            binary_label=True,
            label_mode=args.label_mode
        )

        in_dim = data.num_features
        out_dim = 2

        model = load_model(
            model_type=args.model_type,
            in_dim=in_dim,
            out_dim=out_dim,
            model_path=args.model_path
        )

        metrics = evaluate(model, data)
        print(metrics)

        with open(csv_path, "a") as f:
            f.write(
                f"{gml_path},"
                f"{num_nodes},"
                f"{num_edges},"
                f"{metrics['f1']},"
                f"{metrics['precision']},"
                f"{metrics['recall']},"
                f"{metrics['total_acc']},"
                f"{metrics['class1_acc']},"
                f"{metrics['class0_acc']}\n"
            )

        # saves to  results/model_name
        base = os.path.splitext(os.path.basename(gml_path))[0]
        out_file = os.path.join(results_dir, f"{base}_predictions.gml")

        save_predictions_to_gml(
            original_gml_path=gml_path,
            id2name=id2label,
            preds=metrics["y_pred"],
            out_path=out_file
        )

        # base = os.path.splitext(os.path.basename(gml_path))[0]
        # out_file = f"{base}_predictions.gml"

        # save_predictions_to_gml(
        #     original_gml_path=gml_path,
        #     id2name=id2label,
        #     preds=metrics["y_pred"],
        #     out_path=out_file
        # )