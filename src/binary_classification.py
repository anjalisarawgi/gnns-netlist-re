import torch
import os
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from main import save_predictions_to_gml
from utils.set_seed import set_seed
import wandb
import random
import networkx as nx
from preprocessing import normalize_features
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
from gnn.graphTransformer import GraphTransformer 
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader
import argparse
import time
import json
from torch_geometric.loader import NeighborLoader
import yaml
from torch_geometric.utils import k_hop_subgraph


# could increase to 24 -- 
torch.set_num_threads(24)        # for math mult (pytorch)    
torch.set_num_interop_threads(2)     # pytorch - helper threads
os.environ["OMP_NUM_THREADS"] = "24" # max 20 cores (pytorch)
os.environ["MKL_NUM_THREADS"] = "24" # max 20 cores (intel math libr)
os.environ["NUMEXPR_NUM_THREADS"] = "24"    # 20 threads max
# --- optimization - 

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_method", type=str, choices=["graphsaint", "khop"], default="graphsaint",
                    help="Sampling method: 'graphsaint' or 'khop'")
parser.add_argument("--model", default="gat", choices=["graphsage", "gat", "gcn", "graphTransformer"])
parser.add_argument("--train_gml", type = str,  help="which graph (gml_path) do you want to train on?", nargs="+")
parser.add_argument("--test_gml", type = str, help="which graph (gml_path) do you want to test on?")
parser.add_argument("--epochs", type = int, default=250)

parser.add_argument("--label_mode", type=str, choices = ["subcircuit_name", "boundary"], default="subcircuit_name", help="for sbox and key expand, please use subcircuit")
# graphsaint
parser.add_argument("--walk_length", type=int, default=5, help="what is the walk length you want to set for graphsaint sampling method")

# khop
parser.add_argument("--radius", type=int, default=3, help="what is the radius you want to set for khop sampling method")
parser.add_argument("--num_subgraphs", type=int, default=500, help="what is the number of subgraphs you want to set for khop sampling method")

# khop v2  - neighbourLoader ( + radius)
# parser.add_argument("--batch_size", type=int, default=2048, help="for NeighborLoader")
# parser.add_argument("--neighbors_per_hop", type=int, default=128, help="for NeighborLoader")
# ml args 
parser.add_argument("--set_gradient_clipping", action="store_true", help="do you want to enable gradient clipping (for potentially stable training)?")
parser.add_argument("--normalize_class_weights", action="store_true", help="kinda confused - but to stabalize training? (i think its just like scaling the weights to avoid exploding gradients)")

# config 
parser.add_argument("--config", type=str, help="Path to YAML config file")
args = parser.parse_args()


# yaml - file
config_tag = None
if args.config:
    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        setattr(args, key, value)

# wandb setup 
test_name = os.path.splitext(os.path.basename(args.test_gml))[0]
train_roots = [os.path.splitext(os.path.basename(p))[0] for p in args.train_gml]
train_name = "+".join(train_roots[:2]) + ("+test" if len(train_roots) > 2 else "")

if args.sampling_method == "graphsaint":
    sampling_suffix = f"graphsaint_walk{args.walk_length}"
elif args.sampling_method == "khop":
    sampling_suffix = f"khop_r{args.radius}_n{args.num_subgraphs}"
else:
    sampling_suffix = args.sampling_method

run_name = f"{config_tag}_{args.model}_{sampling_suffix}_{args.epochs}ep_{train_name}_for_{test_name}"


wandb.init(project="gnn-subcircuit-detection", name=run_name)
wandb.config.update(vars(args))

set_seed(42)


def ego_subgraphs_from_data(full_data, radius=2, num_subgraphs=10, seed=42):
    random.seed(seed)
    G = nx.Graph()
    edge_list = full_data.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    subgraph_data_list = []

    for _ in range(num_subgraphs):
        center = random.choice(range(full_data.num_nodes))
        nodes = nx.ego_graph(G, center, radius=radius).nodes()
        nodes = list(nodes)

        sub_nodes = torch.tensor(nodes, dtype=torch.long)
        sub_edge_index, _ = subgraph(sub_nodes, full_data.edge_index, relabel_nodes=True)

        sub_data = Data(
            x=full_data.x[sub_nodes],
            edge_index=sub_edge_index,
            y=full_data.y[sub_nodes],
            train_mask=full_data.train_mask[sub_nodes],
            val_mask=full_data.val_mask[sub_nodes],
            test_mask=full_data.test_mask[sub_nodes],
        )
        sub_data.global_node_id = sub_nodes
        subgraph_data_list.append(sub_data)

    return subgraph_data_list

    
def load_aisec_single_gml(gml_path, label_type="subcircuit", binary_label=False, positive_class=None, remove_edges=False,label_mode = "subcircuit_name"):
    print("calling gnn from path:", gml_path)
    random.seed(42)

    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node]

        # feat = list(map(int, attr.get("features", [])))
        # features.append(feat)

        feat = attr.get("features", [])
        if isinstance(feat, (list, tuple, np.ndarray)):
            features.append(feat)
        else:
            features.append([])

        # if binary_label:
        #     label_name = attr.get("subcircuit_name", "unknown")
        #     if label_name == "sbox":
        #         labels.append(1)   # positive
        #     else:
        #         labels.append(0)   # negative
        # elif label_type == "subcircuit":
        #     labels.append(attr.get("subcircuit", -1))
        # else:
        #     labels.append(int(attr.get("subcircuit_original", -1)))
        if binary_label:
            if label_mode == "boundary":
                # use numeric boundary field
                boundary_value = attr.get("boundary", 0)
                try:
                    label = int(boundary_value)
                except (ValueError, TypeError):
                    label = 0
                labels.append(label)
            else:
                # default: classify based on subcircuit_name == "sbox"
                label_name = attr.get("subcircuit_name", "unknown")
                labels.append(1 if label_name == "sbox" else 0)
        elif label_type == "subcircuit":
            labels.append(attr.get("subcircuit", -1))
        else:
            labels.append(int(attr.get("subcircuit_original", -1)))

    # features = normalize_features(np.array(features, dtype=np.float32))

    
    if binary_label:
        if label_mode == "boundary":
            id2label = {0: "not_boundary", 1: "boundary"}
        else:
            id2label = {0: "not_sbox", 1: "sbox"}
        # id2label = {0: "not_sbox", 1: "sbox"}
        labels = torch.tensor(labels, dtype=torch.long)
    elif label_type == "subcircuit":
        label_set = sorted(set(labels))
        label_map = {v: i for i, v in enumerate(label_set)}
        labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
        id2label = {i: l for l, i in label_map.items()}
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        id2label = {int(l): str(l) for l in sorted(set(labels.tolist()))}

    features = normalize_features(np.array(features, dtype=np.float32))
        
    ### DEBUG >>> Label distribution
    print("\n[DEBUG] Label summary for:", gml_path)
    unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({id2label.get(int(u), '?')}): {c} samples")
    print(f"  → Total: {len(labels)} nodes")
    print(f"  Labels tensor shape: {labels.shape}, dtype: {labels.dtype}\n")

    print(f"Total labels: {len(labels)}")   
    print(f"Unique labels: {sorted(set(labels.tolist()))}")
    label_counts = Counter(labels.tolist())
    print("Label counts:", label_counts)

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_cutoff = int(0.90 * num_nodes)
    val_cutoff = train_cutoff + int(0.05 * num_nodes)

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
        # x=torch.tensor(features, dtype=torch.float),
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, id2label


def train(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss = 0
    batch_count = 0  # new counter

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        # only train on valid, labeled nodes
        valid_mask = (batch.y != -1) & (batch.train_mask)
        if valid_mask.sum() == 0:
            continue

        if class_weights is not None:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight=class_weights)
        else:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])

        loss.backward()
        if args.set_gradient_clipping: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print("[INFO] Using gradient clipping")
        optimizer.step()

        total_loss += loss.item()   # no multiplication
        batch_count += 1            # count batches

    # return average loss per batch (easy to read)
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    valid_mask = mask & (data.y != -1) 
    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()

    accuracy = correct / valid_mask.sum().item() 
    return accuracy

@torch.no_grad()
def evaluate_binary(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=1) # agressive 
    # probs = torch.softmax(out, dim=1) # not so agressive (1) 
    # pred = (probs[:, 1] > 0.9).long() # not so agressive (2) 

    valid_mask = mask & (data.y != -1)

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall

@torch.no_grad()
def evaluate_on_dataset(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=1) # agressive 
    # probs = torch.softmax(out, dim=1)
    # pred = (probs[:, 1] > 0.9).long()

    y_true = data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    total_acc = (y_true == y_pred).sum() / len(y_true)
    sbox_mask = (y_true == 1)
    not_sbox_mask = (y_true == 0)
    sbox_acc = (y_pred[sbox_mask] == y_true[sbox_mask]).sum() / sbox_mask.sum()
    not_sbox_acc = (y_pred[not_sbox_mask] == y_true[not_sbox_mask]).sum() / not_sbox_mask.sum()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "total_acc": total_acc,
        "sbox_acc": sbox_acc,
        "not_sbox_acc": not_sbox_acc
    }


def classwise_accuracy(model, data, mask, id2name=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # y_true = data.y[mask]
    # y_pred = pred[mask]
    valid_mask = mask & (data.y != -1)
    y_true = data.y[valid_mask]
    y_pred = pred[valid_mask]

    unique_classes = torch.unique(y_true).tolist()
    acc_per_class = {}

    for c in unique_classes:
        mask_c = y_true == c 
        total_c = mask_c.sum().item()
        correct_c = (y_pred[mask_c] == c).sum().item()  
        acc = correct_c / total_c if total_c > 0 else 0
        # acc_per_class[c] = acc
        label_name = id2name.get(c, f"Class {c}") if id2name else f"Class {c}"
        acc_per_class[label_name] = acc

    return acc_per_class

# des_data, _ = load_aisec_single_gml(
#     gml_path="graphs/processed/aes_encryption_latest/osu035/aes_key_expand_128_gephi_test6.gml",
#     binary_label=True
# )
# des_data.train_mask[:] = False
# des_data.val_mask[:] = False
# des_data.test_mask[:] = False
# des_mask = torch.ones_like(des_data.y, dtype=torch.bool)

import sys

class Tee:
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal
    def write(self, data):
        self.file.write(data)
        self.terminal.write(data)
    def flush(self):
        self.file.flush()
        self.terminal.flush()

def run_training(data, train_loader, in_dim, out_dim, id2name=None, model_name="gat", use_weighted_loss=False, des_data = None):
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
        print("[INFO] using graphsage model")
    elif model_name == "gcn":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using gcn model")
    elif model_name == "gat":
        model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        model = torch.compile(model)
        print("[INFO] using gat model")
    elif model_name == "graphTransformer":
        model = GraphTransformer(in_channels=in_dim, hidden_channels=256, out_channels = out_dim)
        print("[INFO] using graphTransformer model")

    base_lr = 0.01 
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr ) # weight_decay=5e-4

    # warmup_epochs = 50 
    # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda = lambda epoch: min((epoch+1)/ warmup_epochs, 1.0)
    # )
    # warmup_scheduler.step()

    if use_weighted_loss:
        train_labels = data.y[data.train_mask].cpu().numpy()
        train_labels = train_labels[train_labels != -1]
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        print("Class weights:", weights)
        
        if args.normalize_class_weights:
            weights = weights / np.mean(weights)  
            print("[INFO]Normalized Class Weights:", weights)
        class_weights = torch.tensor(weights, dtype=torch.float, device=data.x.device)
        print("Using weighted cross entropy loss.")
    else:
        class_weights = None
        print("Using standard cross entropy loss.")
    
    ### DEBUG >>> Class weights
    # print("\n[DEBUG] Class weighting info:")
    # for cls, w in zip(classes, weights):
    #     label_name = id2name.get(int(cls), f"Class {cls}") if id2name else f"Class {cls}"
    #     print(f"  {label_name}: weight = {w:.4f}")
    # print("  → Higher weight means rarer class\n")

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        train_acc = None
        val_acc = None

        # Total epoch timer
        t_epoch_start = time.perf_counter()

        # -----------------------------
        # Sampling
        # -----------------------------
        t_sampling_start = time.perf_counter()
        batches = list(train_loader)
        t_sampling_end = time.perf_counter()
        sampling_time = t_sampling_end - t_sampling_start

        # -----------------------------
        # Training
        # -----------------------------
        t_training_start = time.perf_counter()
        loss = train(model, batches, optimizer, class_weights)
        t_training_end = time.perf_counter()
        training_time = t_training_end - t_training_start

        # -----------------------------
        # Periodic Evaluation
        # -----------------------------
        if epoch % 50 == 0 or epoch == epochs:
            train_acc = evaluate(model, data, data.train_mask)
            val_acc   = evaluate(model, data, data.val_mask)

            test_acc_tmp = evaluate(model, data, data.test_mask)
            f1_tmp, precision_tmp, recall_tmp = evaluate_binary(model, data, data.test_mask)

            print(
                f"[Periodic Eval] Epoch {epoch} | "
                f"Train={train_acc:.4f} | Val={val_acc:.4f} | "
                # f"Test={test_acc_tmp:.4f} | "
                f"F1={f1_tmp:.4f} | P={precision_tmp:.4f} | R={recall_tmp:.4f}"
            )

            wandb.log({
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                # "test_accuracy": test_acc_tmp,
                "test_f1": f1_tmp,
                "test_precision": precision_tmp,
                "test_recall": recall_tmp,
                "eval_epoch": epoch,
            })

            # ---------------------------------
            # Cross-graph DES evaluation
            # ---------------------------------
            if des_data is not None:
                # des_mask = torch.ones_like(des_data.y, dtype=torch.bool)
                # des_f1, des_precision, des_recall = evaluate_binary(model, des_data, des_mask)
                metrics_des = evaluate_on_dataset(model,des_data)

                # wandb.log({
                #     "des/f1": des_f1,
                #     "des/precision": des_precision,
                #     "des/recall": des_recall,
                #     "des/epoch": epoch,
                # })

                # print(
                #     f"[DES Eval] Epoch {epoch} | "
                #     f"F1={des_f1:.4f} | P={des_precision:.4f} | R={des_recall:.4f}"
                # )
                wandb.log({
                    "des/epoch": epoch,
                    "des/f1": metrics_des["f1"],
                    "des/precision": metrics_des["precision"],
                    "des/recall": metrics_des["recall"],
                    "des/acc_total": metrics_des["total_acc"],
                    "des/acc_not_sbox": metrics_des["not_sbox_acc"],
                    "des/acc_sbox": metrics_des["sbox_acc"],
                })

                print(
                    f"[DES Eval] Epoch {epoch} | "
                    f"F1={metrics_des['f1']:.4f} | "
                    f"P={metrics_des['precision']:.4f} | "
                    f"R={metrics_des['recall']:.4f} | "
                    f"Acc_total={metrics_des['total_acc']:.4f} | "
                    f"Acc_not_sbox={metrics_des['not_sbox_acc']:.4f} | "
                    f"Acc_sbox={metrics_des['sbox_acc']:.4f} " 
                )
                
        # -----------------------------
        # WandB logging
        # -----------------------------
        t_wandb_start = time.perf_counter()
        t_wandb_end = time.perf_counter()
        wandb_time = t_wandb_end - t_wandb_start

        # -----------------------------
        # Total and Misc Time
        # -----------------------------
        t_epoch_end = time.perf_counter()
        epoch_total_time = t_epoch_end - t_epoch_start
        misc_time = epoch_total_time - sampling_time - training_time - wandb_time

        # -----------------------------
        # PRINT EVERY EPOCH
        # -----------------------------
        train_acc_print = f"{train_acc:.4f}" if train_acc is not None else "----"
        val_acc_print   = f"{val_acc:.4f}"   if val_acc is not None else "----"

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={loss:.4f} | "
            f"Train Acc={train_acc_print} | "
            f"Val Acc={val_acc_print} | "
            f"sampling={sampling_time:.2f}s | "
            f"training={training_time:.2f}s | "
            f"wandb={wandb_time:.2f}s | "
            f"misc={misc_time:.2f}s | "
            f"total={epoch_total_time:.2f}s"
        )       
    ######### 

    test_acc = evaluate(model, data, data.test_mask)
    print(f"Final test accuracy: {test_acc:.4f}")

    if out_dim == 2:  # binary classification
        f1, precision, recall = evaluate_binary(model, data, data.test_mask)
        print(f"Binary classification metrics:")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")

        wandb.log({
            "final/test_accuracy": test_acc,
            "final/test_f1": f1,
            "final/test_precision": precision,
            "final/test_recall": recall,
        })
    
    classwise_acc = classwise_accuracy(model, data, data.test_mask, id2name)
    print("  Test Class-wise Accuracy:")
    for cls, acc in classwise_acc.items():
        print(f"    Class {cls}: {acc:.4f}")
    
    # Save model
    run_dir =  os.path.join("models", wandb.run.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = f"{run_dir}/model.pt"
    torch.save(model.state_dict(), model_path)
    print("[INFO] Saved model to:", model_path)

    # saving some meta data for logging
    metadata = {
        "model_file": model_path,
        "run_name": wandb.run.name,
        "num_training_graphs": len(args.train_gml),
        "training_graphs": args.train_gml,
        "test_graph": args.test_gml,
        "model_type": args.model,
        "sampling_method": args.sampling_method,
        "epochs": args.epochs,
        "label_mode": args.label_mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = f"{run_dir}/metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("[INFO] Saved metadata to:", json_path)
    return model



def merge_data(data1, data2):
    # Offset node indices in data2's edge_index
    offset = data1.num_nodes
    data2_edge_index = data2.edge_index + offset

    # Concatenate all fields
    x = torch.cat([data1.x, data2.x], dim=0)
    edge_index = torch.cat([data1.edge_index, data2_edge_index], dim=1)
    y = torch.cat([data1.y, data2.y], dim=0)

    train_mask = torch.cat([data1.train_mask, data2.train_mask], dim=0)
    val_mask = torch.cat([data1.val_mask, data2.val_mask], dim=0)
    test_mask = torch.cat([data1.test_mask, data2.test_mask], dim=0)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )


######################################
# gml_paths = [
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_1.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_2.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_3.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_4.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_5.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_6.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_7.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_8.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_9.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_10.gml",

# ]
########################################


from functools import reduce
if __name__ == "__main__":
    # aes_data, id2label = load_aisec_single_gml(
    #     # gml_path="graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi_test6.gml",
    #     gml_path=args.train_gml,
    #     binary_label=True
    # )

    print("[INFO] loading the following training graphs:")
    train_graphs = []
    for i, gml_path in enumerate(args.train_gml):
        print(f"[{i+1}] {gml_path}")
        graph_data, label_map = load_aisec_single_gml(gml_path=gml_path, binary_label=True, label_mode = args.label_mode)
        if i == 0:
            id2label = label_map
        train_graphs.append(graph_data)
    aes_data = reduce(merge_data, train_graphs)
    print("[INFO] training on:", args.train_gml)
    print("Number of features:", aes_data.num_features)
    print("Feature matrix shape:", aes_data.x.shape)
    print("Number of SBOX nodes :", (aes_data.y == 1).sum().item())
    print("Number of not-SBOX nodes :", (aes_data.y == 0).sum().item())
    print("Total training nodes:", aes_data.num_nodes)

    print("[DEBUG] Combined data split:")
    print("  Train nodes:", aes_data.train_mask.sum().item())
    print("  Val nodes  :", aes_data.val_mask.sum().item())
    print("  Test nodes :", aes_data.test_mask.sum().item())

    des_data, _ = load_aisec_single_gml(
        # gml_path="graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi_test6.gml",
        gml_path=args.test_gml,
        binary_label=True,
        label_mode = args.label_mode
    )
    des_data.train_mask[:] = False
    des_data.val_mask[:] = False
    des_data.test_mask[:] = False
    print("Total test nodes:", des_data.num_nodes)


    if args.sampling_method == "graphsaint":
        sample_start = time.perf_counter()
        aes_loader = GraphSAINTRandomWalkSampler(
            aes_data,
            batch_size=10000, #int(0.30 * aes_data.num_nodes),
            walk_length=args.walk_length,
            shuffle=True,
        )
        sample_time = time.perf_counter() - sample_start
        print(f"[TIME] GraphSAINT sampler setup took {sample_time:.2f} seconds.")

    elif args.sampling_method == "khop":
        print(f"[INFO] Using k-hop sampling...")
        subgraph_list = ego_subgraphs_from_data(
            aes_data,
            radius=args.radius,
            num_subgraphs=args.num_subgraphs
        )
        aes_loader = DataLoader(subgraph_list, batch_size=32768, shuffle=True)
        
    # elif args.sampling_method == "neighSampler": # ???
    #     print("[INFO] Using NeighborLoader for k-hop neighborhood sampling...")
    #     num_neighbors = [args.neighbors_per_hop] * args.radius
    #     aes_loader = NeighborLoader(
    #         aes_data,
    #         num_neighbors=num_neighbors,  
    #         batch_size=args.batch_size,   
    #         shuffle=True,                 
    #         num_workers=8,                
    #         persistent_workers=True,
    #         pin_memory=True
    #     )
    # logging 
    log_dir = os.path.join("logs", wandb.run.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    log_file = open(log_path, "w")
    sys.stdout = Tee(log_file, sys.__stdout__)
        
    model = run_training(
        data=aes_data,
        train_loader=aes_loader,
        in_dim=aes_data.num_features,
        out_dim=2,
        id2name=id2label,
        model_name=args.model, 
        use_weighted_loss=True,
        des_data=des_data 
    )

    print("\n[DEBUG] DES dataset:")
    print("Total nodes:", des_data.num_nodes)
    print("SBOX nodes:", (des_data.y == 1).sum().item())
    print("Not-SBOX nodes:", (des_data.y == 0).sum().item())

    mask = torch.ones_like(des_data.y, dtype=torch.bool)

    f1, precision, recall = evaluate_binary(model, des_data, mask)
    print(f"\n=== Cross-graph test (AES→DES) ===")
    # print(f"F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

    # model.eval()
    # out = model(des_data.x, des_data.edge_index)
    # pred = out.argmax(dim=1)

    # y_true = des_data.y.cpu().numpy()
    # y_pred = pred.cpu().numpy()

    # total_correct = (y_true == y_pred).sum()
    # total_acc = total_correct / len(y_true)

    # sbox_mask = (y_true == 1)
    # not_sbox_mask = (y_true == 0)

    # sbox_acc = (y_pred[sbox_mask] == y_true[sbox_mask]).sum() / sbox_mask.sum()
    # not_sbox_acc = (y_pred[not_sbox_mask] == y_true[not_sbox_mask]).sum() / not_sbox_mask.sum()

    # print("\n=== DES Accuracy Breakdown ===")
    # print(f"Total Accuracy   : {total_acc:.4f}")
    # print(f"SBOX Accuracy    : {sbox_acc:.4f}")
    # print(f"Not-SBOX Accuracy: {not_sbox_acc:.4f}")


    metrics = evaluate_on_dataset(model, des_data)
    print(f"F1 = {metrics['f1']:.4f}, Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
    print(f"Total Accuracy   : {metrics['total_acc']:.4f}")
    print(f"SBOX Accuracy    : {metrics['sbox_acc']:.4f}")
    print(f"Not-SBOX Accuracy: {metrics['not_sbox_acc']:.4f}")

    
    output_dir = "results/aes_to_des"
    os.makedirs(output_dir, exist_ok=True)
    test_graph_name = os.path.splitext(os.path.basename(args.test_gml))[0]
    output_path = os.path.join(output_dir, f"{test_graph_name}_predictions.gml")
    print("[DEBUG]  Saving predictions to:", output_path)

    if args.label_mode == "boundary":
        output_labels = {0: "not_boundary", 1: "boundary"}
    else:
        output_labels = {0: "not_sbox", 1: "sbox"}


    save_predictions_to_gml(
        original_gml_path = args.test_gml,
        data=des_data,
        model=model,
        id2name=output_labels,
        output_gml_path=output_path
    )

    # results = check_boundary_coverage(
    #     gml_path=output_path,
    #     model=model,
    #     threshold=0.6
    # )