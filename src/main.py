import torch 
from preprocessing import normalize_features, load_GNNRE_full, load_GNNRE_gmls, load_GNN_aes_core_gmls, load_aisec_single_gml, load_aisec_multiple_gmls
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler    
import torch
from torch_geometric.data import Data
import numpy as np
import os 
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Batch
import random
import matplotlib.pyplot as plt
import argparse
import json 
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.loader import GraphSAINTEdgeSampler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


subcircuit_map = defaultdict(set)

import sys
from datetime import datetime

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def isatty(self):
            return False

    logfile = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    print(f"Logging to: {log_path}")


def train(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss = 0
    total_valid_nodes = 0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        valid_mask = (batch.y != -1) & (batch.train_mask)
        if valid_mask.sum() == 0:
            continue

        if class_weights is not None:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight=class_weights)
        else:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * valid_mask.sum().item()
        total_valid_nodes += valid_mask.sum().item()

    if total_valid_nodes == 0:
        return 0.0
    return total_loss / total_valid_nodes


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    valid_mask = mask & (data.y != -1)
    if valid_mask.sum().item() == 0:
        return float('nan')

    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()
    return correct / valid_mask.sum().item()


@torch.no_grad()
def evaluate_binary(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)
    pred = (probs[:, 1] > 0.5).long()

    valid_mask = mask & (data.y != -1)
    if valid_mask.sum().item() == 0:
        return float('nan'), float('nan'), float('nan')

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall


def classwise_accuracy(model, data, mask, id2name=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    valid_mask = mask & (data.y != -1)
    if valid_mask.sum().item() == 0:
        return {}

    y_true = data.y[valid_mask]
    y_pred = pred[valid_mask]

    acc_per_class = {}
    for c in torch.unique(y_true).tolist():
        mask_c   = y_true == c
        correct_c = (y_pred[mask_c] == c).sum().item()
        acc = correct_c / mask_c.sum().item()
        label_name = id2name.get(c, f"Class {c}") if id2name else f"Class {c}"
        acc_per_class[label_name] = acc

    return acc_per_class


def log_class_distribution(y, name=""):
    valid = y[y != -1]
    if valid.numel() == 0:
        print(f"Class distribution in {name}: (empty)")
        return
    values, counts = torch.unique(valid, return_counts=True)
    print(f"Class distribution in {name}:")
    for v, c in zip(values.tolist(), counts.tolist()):
        print(f"  Class {v}: {c} nodes")


def run_training(train_data, val_data, test_data, train_loader,
                 in_dim, out_dim, id2name=None,
                 model_name="graphsage", use_weighted_loss=False):
    """
    Three fully separate graphs:
      train_data  — train_mask is all-True  (training circuit)
      val_data    — val_mask   is all-True  (validation circuit)
      test_data   — test_mask  is all-True  (test circuit)
    """
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "gcn":
        model = GCN(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "gat":
        model = gat(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if use_weighted_loss:
        train_labels = train_data.y[train_data.train_mask].cpu().numpy()
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float, device=train_data.x.device)
        print("Using weighted cross entropy loss.")
    else:
        class_weights = None
        print("Using standard cross entropy loss.")

    epochs = 250
    for epoch in range(1, epochs + 1):
        loss      = train(model, train_loader, optimizer, class_weights)
        train_acc = evaluate(model, train_data, train_data.train_mask)
        val_acc   = evaluate(model, val_data,   val_data.val_mask)      # <-- separate val graph

        if out_dim == 2:
            f1, precision, recall = evaluate_binary(model, val_data, val_data.val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, '
                  f'F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}')
        else:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}')

        if epoch % 100 == 0 or epoch == 1:
            classwise_acc = classwise_accuracy(model, val_data, val_data.val_mask, id2name)
            if classwise_acc:
                print("  Val Class-wise Accuracy:")
                for cls, acc in classwise_acc.items():
                    print(f"    {cls}: {acc:.4f}")

    # --- Final test evaluation on held-out test circuit ---
    print("\n=== Final Test Evaluation ===")
    test_acc = evaluate(model, test_data, test_data.test_mask)
    print(f"Test accuracy: {test_acc:.4f}")

    if out_dim == 2:
        f1, precision, recall = evaluate_binary(model, test_data, test_data.test_mask)
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")

    classwise_acc = classwise_accuracy(model, test_data, test_data.test_mask, id2name)
    if classwise_acc:
        print("  Test Class-wise Accuracy:")
        for cls, acc in classwise_acc.items():
            print(f"    {cls}: {acc:.4f}")

    return model


def save_predictions_to_gml(original_gml_path, data, model, id2name, output_gml_path="gml_results.gml"):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    G = nx.read_gml(original_gml_path, label="id")
    node_map = {node: idx for idx, node in enumerate(G.nodes())}

    for node in G.nodes():
        idx = node_map[node]
        true_id = data.y[idx].item()
        pred_id = pred[idx].item()

        G.nodes[node]["true_label"]      = id2name.get(true_id, str(true_id))
        G.nodes[node]["predicted_label"] = id2name.get(pred_id, str(pred_id))
        G.nodes[node]["correct"]         = (true_id == pred_id)

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


def run_phase1(args):
    print("=== Phase 1: Fine-grained subcircuit classification ===")

    if not args.train_gmls or not args.val_gmls or not args.test_gmls:
        raise ValueError("You must specify --train_gmls, --val_gmls, and --test_gmls")

    # --- Load each split as its own independent graph ---
    train_data, id2name = load_aisec_multiple_gmls(
        gml_paths=args.train_gmls,
        label_type=args.label_type,
        binary_label=args.binary_label,
        positive_class=args.positive_class,
    )
    train_data.train_mask = torch.ones(train_data.num_nodes,  dtype=torch.bool)
    train_data.val_mask   = torch.zeros(train_data.num_nodes, dtype=torch.bool)
    train_data.test_mask  = torch.zeros(train_data.num_nodes, dtype=torch.bool)

    val_data, _ = load_aisec_multiple_gmls(
        gml_paths=args.val_gmls,
        label_type=args.label_type,
        binary_label=args.binary_label,
        positive_class=args.positive_class,
    )
    val_data.train_mask = torch.zeros(val_data.num_nodes, dtype=torch.bool)
    val_data.val_mask   = torch.ones(val_data.num_nodes,  dtype=torch.bool)
    val_data.test_mask  = torch.zeros(val_data.num_nodes, dtype=torch.bool)

    test_data, _ = load_aisec_multiple_gmls(
        gml_paths=args.test_gmls,
        label_type=args.label_type,
        binary_label=args.binary_label,
        positive_class=args.positive_class,
    )
    test_data.train_mask = torch.zeros(test_data.num_nodes, dtype=torch.bool)
    test_data.val_mask   = torch.zeros(test_data.num_nodes, dtype=torch.bool)
    test_data.test_mask  = torch.ones(test_data.num_nodes,  dtype=torch.bool)

    in_dim  = train_data.num_features
    out_dim = 2 if args.binary_label else len(torch.unique(train_data.y))
    print(f"in_dim: {in_dim}  |  out_dim: {out_dim}")

    log_class_distribution(train_data.y[train_data.train_mask], "train")
    log_class_distribution(val_data.y[val_data.val_mask],       "val")
    log_class_distribution(test_data.y[test_data.test_mask],    "test")

    if args.binary_label:
        n_pos = {
            "train": (train_data.y == 1).sum().item(),
            "val":   (val_data.y   == 1).sum().item(),
            "test":  (test_data.y  == 1).sum().item(),
        }
        print(f"Positive examples — {n_pos}")
        if n_pos["train"] == 0:
            print("[WARNING] No positive examples in training data!")
            print("  Tip: pass --positive_class as the subcircuit NAME, e.g. --positive_class aes_sbox")

    train_loader = [train_data]
    # batch_size = max(1, int(0.3 * train_data.num_nodes))
    # train_loader = GraphSAINTRandomWalkSampler(
    #     train_data,
    #     batch_size=batch_size,
    #     walk_length=2,
    #     shuffle=True,
    # )

    model = run_training(
        train_data,
        val_data,
        test_data,
        train_loader,
        in_dim,
        out_dim,
        id2name,
        model_name=args.model,
        use_weighted_loss=args.weighted_loss,
    )
    return train_data, model, id2name


def run_phase2(data, model, args, id2name):
    print("=== Phase 2: Zoom-in on target subcircuit class ===")
    with open("results/aes_cipher_top_gephi/subcircuit_map.json", "r") as f:
        subcircuit_map = json.load(f)
    print(f"Loaded subcircuit map from JSON.")
    print(json.dumps(subcircuit_map, indent=2))

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    target_subcircuit_label = 3
    PHASE2_CLASS = [i for i, name in id2name.items() if name == target_subcircuit_label][0]
    phase2_mask = pred == PHASE2_CLASS

    G = nx.read_gml(args.gml_path, label="id")
    node_list = list(G.nodes())

    def is_valid_subcircuit(sc_id):
        return sc_id.startswith("top+us") and "round2" not in sc_id

    subcircuit_ids = []
    phase2_indices = []
    for idx in torch.where(phase2_mask)[0]:
        node_idx  = idx.item()
        node_name = node_list[node_idx]
        sc_id     = G.nodes[node_name].get("subcircuit_id", "unknown")
        if is_valid_subcircuit(sc_id):
            subcircuit_ids.append(sc_id)
            phase2_indices.append(node_idx)

    k = 5
    scid_counts       = Counter(subcircuit_ids)
    top_k_subcircuits = set([s for s, _ in scid_counts.most_common(k)])

    filtered_indices = []
    filtered_labels  = []
    for idx, sid in zip(phase2_indices, subcircuit_ids):
        if sid in top_k_subcircuits:
            filtered_indices.append(idx)
            filtered_labels.append(sid)

    unique_ids     = sorted(set(filtered_labels))
    id_map         = {name: i for i, name in enumerate(unique_ids)}
    id2name_phase2 = {i: name for name, i in id_map.items()}

    new_y      = torch.full((data.num_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros_like(new_y, dtype=torch.bool)
    val_mask   = torch.zeros_like(new_y, dtype=torch.bool)
    test_mask  = torch.zeros_like(new_y, dtype=torch.bool)

    for idx, sid in zip(filtered_indices, filtered_labels):
        new_y[idx] = id_map[sid]
        if data.train_mask[idx]:
            train_mask[idx] = True
        elif data.val_mask[idx]:
            val_mask[idx] = True
        elif data.test_mask[idx]:
            test_mask[idx] = True

    phase2_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        y=new_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    log_class_distribution(phase2_data.y[phase2_data.train_mask], "PHASE 2 train")
    log_class_distribution(phase2_data.y[phase2_data.val_mask],   "PHASE 2 val")
    log_class_distribution(phase2_data.y[phase2_data.test_mask],  "PHASE 2 test")

    phase2_loader = GraphSAINTRandomWalkSampler(
        phase2_data,
        batch_size=int(0.3 * phase2_data.num_nodes),
        walk_length=2,
        shuffle=True,
    )

    print("Running Phase 2 training...")
    run_training(
        phase2_data,
        phase2_data,   # val portion via val_mask
        phase2_data,   # test portion via test_mask
        train_loader=phase2_loader,
        in_dim=phase2_data.num_node_features,
        out_dim=len(unique_ids),
        id2name=id2name_phase2,
        model_name=args.model,
        use_weighted_loss=args.weighted_loss,
    )


if __name__ == "__main__":
    setup_logging("logs")
    set_seed(42)

    parser = argparse.ArgumentParser(description="GNN for Subcircuit Detection")
    parser.add_argument("--gml_path",   type=str, default=None,
                        help="Single GML path (only used for save_predictions_to_gml)")
    parser.add_argument("--gml_paths",  nargs="+", help="List of GML paths to combine")
    parser.add_argument("--train_gmls", nargs="+", help="GML files used for training")
    parser.add_argument("--val_gmls",   nargs="+", help="GML files used for validation")  # NEW
    parser.add_argument("--test_gmls",  nargs="+", help="GML files used for testing")
    parser.add_argument("--model", type=str, default="graphsage",
                        choices=["graphsage", "gcn", "gat"])
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Use class-weighted cross entropy loss")
    parser.add_argument("--label_type",
                        choices=["subcircuit", "subcircuit_original"],
                        default="subcircuit")
    parser.add_argument("--class_reduce", action="store_true")
    parser.add_argument("--binary_label", action="store_true",
                        help="Enable binary classification")
    parser.add_argument("--positive_class", type=str, default=None,
                        help="Subcircuit NAME to treat as positive class, e.g. aes_sbox")
    parser.add_argument("--remove_edges", action="store_true")

    args = parser.parse_args()

    # Allow integer ID to be passed too
    if args.positive_class is not None:
        try:
            args.positive_class = int(args.positive_class)
        except ValueError:
            pass  # keep as string name

    data, model, id2name = run_phase1(args)

    if args.binary_label:
        print("Skipping Phase 2: Binary classification selected.")
    elif args.label_type == "subcircuit":
        run_phase2(data, model, args, id2name)
    else:
        print("Skipping Phase 2: subcircuit_original does not support it.")

    output_dir = os.path.join(
        "results",
        f"{args.model}_{os.path.basename(args.gml_path or 'multi').replace('.gml', '')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.gml_path and os.path.exists(args.gml_path):
        save_predictions_to_gml(
            args.gml_path, data, model, id2name,
            output_gml_path=os.path.join(output_dir, "predictions.gml"),
        )
    else:
        print("Skipping save_predictions_to_gml: not a single-GML run.")