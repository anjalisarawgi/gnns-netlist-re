import torch 
from preprocessing import normalize_features, load_GNNRE_full, load_GNNRE_gmls, load_GNN_aes_core_gmls,load_aisec_single_gml, load_aisec_multiple_gmls
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler    
import torch
from torch_geometric.data import Data
import numpy as np
import os 
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
# from gml_to_pyg import convert_gml_to_pyg
from utils.set_seed import set_seed
from utils.logging import setup_logging
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Batch
import random
import wandb
import matplotlib.pyplot as plt
import argparse
import json 
from collections import defaultdict, Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.loader import GraphSAINTEdgeSampler


subcircuit_map = defaultdict(set) 


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

        if binary_label:
            label = int(attr.get("subcircuit_original", -1))  # <-- Use fine-grained labels for binary
        elif label_type == "subcircuit":
            label = attr.get("subcircuit", "unknown")
        else:
            label = int(attr.get("subcircuit_original", -1))

        labels.append(label)

    features = normalize_features(np.array(features))
    print(f"Total labels: {len(labels)}")
    print(f"Unique labels before encoding: {sorted(set(labels))}")
    label_counts = Counter(labels)
    print("Label counts (raw):")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")

    # If you also want the dictionary explicitly:
    print("Label count dictionary:", dict(label_counts))
    
    

    if label_type == "subcircuit":
        label_set = sorted(set(labels))
        label_map = {v: i for i, v in enumerate(label_set)}
        labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
        id2label = {i: l for l, i in label_map.items()}
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        id2label = {int(l): str(l) for l in sorted(set(labels.tolist()))}


    if binary_label:
        if positive_class is None:
            raise ValueError("You must set --positive_class when using --binary_label")
        print(f"Binary classification: Positive class = {positive_class}")
        labels = torch.where(labels == positive_class, 1, 0)
        print("Label counts after binarization:")
        print("Positive (1):", (labels == 1).sum().item())
        print("Negative (0):", (labels == 0).sum().item())
        id2label = {0: "negative", 1: "positive"}

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


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

    
    # --- Remove cross-split edges (inductive setup) ---
    if remove_edges:
        print("removing edge information to prevent leakage")
        train_nodes = set(torch.where(train_mask)[0].tolist())
        val_nodes   = set(torch.where(val_mask)[0].tolist())
        test_nodes  = set(torch.where(test_mask)[0].tolist())

        new_edges = []
        for src, dst in edge_index.t().tolist():
            if (src in train_nodes and dst in train_nodes) \
            or (src in val_nodes and dst in val_nodes) \
            or (src in test_nodes and dst in test_nodes):
                new_edges.append([src, dst])

        edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    else:
        print("keeping all edges to retain all information")
        ####    ---------------------------------------------


    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, id2label


def train(model, loader, optimizer, class_weights = None):
    model.train()
    total_loss = 0
    

    # batch.x = node features
    # batch.edge_index = edge index of the subgraph
    # batch.y = true labels

    for batch in loader: # iterates over batches from the graph sampler ( each batch is a subgraph)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index) # (node features, edge index) and out is prediction logits of shape: [num_nodes_in_batch, num_classes]

        # we use -1 now to mark nodes we dont want in training
        # also input and ouput nodes are already labeled -1 
        valid_mask = (batch.y != -1) & (batch.train_mask)
        if valid_mask.sum() == 0:
            continue

        if class_weights is not None:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight=class_weights)
        else:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * valid_mask.sum().item()  # sum loss over valid nodes
        total_valid_nodes = valid_mask.sum().item()

    return total_loss / total_valid_nodes



@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    valid_mask = mask & (data.y != -1) 
    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()

    accuracy = correct / mask.sum().item() 
    return accuracy


from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_binary(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    
    # pred = out.argmax(dim=1) # agressive 
    probs = torch.softmax(out, dim=1) # not so agressive (1) 
    pred = (probs[:, 1] > 0.7).long() # not so agressive (2) 

    valid_mask = mask & (data.y != -1)

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall


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

def log_class_distribution(y, name=""):
    values, counts = torch.unique(y, return_counts=True)
    print(f"Class distribution in {name}:")
    for v, c in zip(values.tolist(), counts.tolist()):
        print(f"  Class {v}: {c} nodes")

def run_training(data, train_loader, in_dim, out_dim, id2name=None, model_name="gat", use_weighted_loss=False):
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "gcn":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    elif model_name == "gat":
        model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 ) # weight_decay=5e-4

    if use_weighted_loss:
        train_labels = data.y[data.train_mask].cpu().numpy()
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float, device=data.x.device)
        print("Using weighted cross entropy loss.")
    else:
        class_weights = None
        print("Using standard cross entropy loss.")
        

    epochs = 500
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer, class_weights)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)

        log_data = {
            "epoch": epoch,
            "loss": loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        }

        if out_dim == 2:
            f1, precision, recall = evaluate_binary(model, data, data.val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        else:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}')

        if epoch % 100 == 0 or epoch == 1:
            classwise_acc = classwise_accuracy(model, data, data.val_mask, id2name)
            wandb.log({
                "epoch": epoch,
                "val_classwise_accuracy": {
                    cls: acc for cls, acc in classwise_acc.items()
                }
            })
            print("  Val Class-wise Accuracy:")
            for cls, acc in classwise_acc.items():
                print(f"    Class {cls}: {acc:.4f}")
        
        wandb.log(log_data)



    test_acc = evaluate(model, data, data.test_mask)
    wandb.log({"final_test_accuracy": test_acc})
    print(f"Final test accuracy: {test_acc:.4f}")

    if out_dim == 2:  # binary classification
        f1, precision, recall = evaluate_binary(model, data, data.test_mask)
        print(f"Binary classification metrics:")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
    
    classwise_acc = classwise_accuracy(model, data, data.test_mask, id2name)
    print("  Test Class-wise Accuracy:")
    for cls, acc in classwise_acc.items():
        wandb.log({f"test_acc/{cls}": acc})
        print(f"    Class {cls}: {acc:.4f}")
        
    return model



def save_predictions_to_gml(original_gml_path, data, model, id2name, output_gml_path="gml_results.gml"):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

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
    
    # data, id2name = load_aisec_single_gml(args.gml_path, label_type=args.label_type,  binary_label=args.binary_label, positive_class=args.positive_class)
    # data, id2name = load_aisec_multiple_gmls(
    #     gml_paths=args.gml_paths,
    #     label_type=args.label_type,
    #     binary_label=args.binary_label,
    #     positive_class=args.positive_class
    # )

    if args.gml_paths:  # multiple GMLs
        data, id2name = load_aisec_multiple_gmls(
            gml_paths=args.gml_paths,
            label_type=args.label_type,
            binary_label=args.binary_label,
            positive_class=args.positive_class
        )
    elif args.gml_path:  # single GML
        data, id2name = load_aisec_single_gml(
            gml_path=args.gml_path,
            label_type=args.label_type,
            binary_label=args.binary_label,
            positive_class=args.positive_class,
            remove_edges=args.remove_edges
        )
    else:
        raise ValueError("You must specify either --gml_path or --gml_paths")



    in_dim = data.num_features
    # out_dim = len(torch.unique(data.y))
    out_dim = 2 if args.binary_label else len(torch.unique(data.y))
    print("in_dim", in_dim)
    print("out_dim", out_dim)

    log_class_distribution(data.y[data.train_mask], "train")
    log_class_distribution(data.y[data.val_mask], "val")
    log_class_distribution(data.y[data.test_mask], "test")

    batch_size = int(0.3 * data.num_nodes)
    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2, shuffle=True)
    # train_loader = GraphSAINTEdgeSampler(data,batch_size=batch_size, num_steps=2,shuffle=True)

    model = run_training(
        data,
        train_loader,
        in_dim,
        out_dim,
        id2name,
        model_name=args.model,
        use_weighted_loss=args.weighted_loss,
    )

    return data, model, id2name


def run_phase2(data, model, args, id2name):
    print("=== Phase 2: Zoom-in on target subcircuit class ===")
    with open("results/aes_cipher_top_gephi/subcircuit_map.json", "r") as f:
        subcircuit_map = json.load(f)
    print(f"Loaded subcircuit map from JSON.")
    print(json.dumps(subcircuit_map, indent=2))

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    target_subcircuit_label = 3  # hardcoded for now
    PHASE2_CLASS = [i for i, name in id2name.items() if name == target_subcircuit_label][0]
    phase2_mask = pred == PHASE2_CLASS

    G = nx.read_gml(args.gml_path, label="id")
    node_list = list(G.nodes())

    def is_valid_subcircuit(sc_id):
        return sc_id.startswith("top+us") and not "round2" in sc_id

    subcircuit_ids = []
    phase2_indices = []
    for idx in torch.where(phase2_mask)[0]:
        node_idx = idx.item()
        node_name = node_list[node_idx]
        sc_id = G.nodes[node_name].get("subcircuit_id", "unknown")
        if is_valid_subcircuit(sc_id):
            subcircuit_ids.append(sc_id)
            phase2_indices.append(node_idx)

    k = 5
    scid_counts = Counter(subcircuit_ids)
    top_k_subcircuits = set([s for s, _ in scid_counts.most_common(k)])

    filtered_indices = []
    filtered_labels = []
    for idx, sid in zip(phase2_indices, subcircuit_ids):
        if sid in top_k_subcircuits:
            filtered_indices.append(idx)
            filtered_labels.append(sid)

    unique_ids = sorted(set(filtered_labels))
    id_map = {name: i for i, name in enumerate(unique_ids)}
    id2name_phase2 = {i: name for name, i in id_map.items()}

    new_y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for idx, sid in zip(filtered_indices, filtered_labels):
        new_y[idx] = id_map[sid]

    train_mask = torch.zeros_like(new_y, dtype=torch.bool)
    val_mask = torch.zeros_like(new_y, dtype=torch.bool)
    test_mask = torch.zeros_like(new_y, dtype=torch.bool)

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
    log_class_distribution(phase2_data.y[phase2_data.val_mask], "PHASE 2 val")
    log_class_distribution(phase2_data.y[phase2_data.test_mask], "PHASE 2 test")

    phase2_loader = GraphSAINTRandomWalkSampler(
        phase2_data,
        batch_size=int(0.3 * phase2_data.num_nodes),
        walk_length=2,
        shuffle=True,
    )

    print("Running Phase 2 training...")
    run_training(
        phase2_data,
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
    parser.add_argument("--gml_path", type=str, default="aes_key_expand_features.gml", help="Path to the input GML file")
    parser.add_argument("--gml_paths", nargs="+",  help="List of GML paths to combine")
    parser.add_argument("--model", type=str, default="gat", choices=["graphsage", "gcn", "gat"], help="GNN model to use")
    # parser.add_argument("--class_reduce", action="store_true", help="Whether to reduce classes or not, ", default=False)
    parser.add_argument("--weighted_loss", action="store_true", help="Use class-weighted cross entropy loss")
    parser.add_argument(
        "--label_type",
        choices=["subcircuit", "subcircuit_original"],
        default="subcircuit",
        help="Label type to use for training (only 'subcircuit' supports phase 2)"
    )
    parser.add_argument(
        "--class_reduce",
        action="store_true",
        help="Enable class reduction in phase 2 (only used if label_type=subcircuit)"
    )

    parser.add_argument(
        "--binary_label",
        action="store_true",
        help="Enable binary classification (label is 1 if it matches --positive_class, else 0)"
    )

    parser.add_argument(
        "--positive_class",
        type=int,
        default=None,
        help="Label ID to treat as positive class when using binary classification"
    )
    parser.add_argument(
        "--remove_edges",
        action="store_true",
        help="If set, Remove edges between train/val/test splits to simulate inductive setup"
    )

    args = parser.parse_args()

    wandb.init(project="gnn-subcircuit-detection", name=f"{args.model}-{os.path.basename(args.gml_path).replace('.gml', '')}-class_reduce-{args.class_reduce}")

    data, model, id2name = run_phase1(args)

    if args.binary_label:
        print("Skipping Phase 2: Binary classification selected.")
    elif args.label_type == "subcircuit":
        run_phase2(data, model, args, id2name)
    else:
        print("Skipping Phase 2: subcircuit_original does not support it.")
            

    output_dir = os.path.join("results", f"{args.model}_{os.path.basename(args.gml_path).replace('.gml', '')}")
    os.makedirs(output_dir, exist_ok=True)
    # save_predictions_to_gml(args.gml_path, data, model, id2name, output_gml_path=os.path.join(output_dir, "predictions.gml"))

    # only save if its a single gml input (gml_path)
    if args.gml_path and os.path.exists(args.gml_path):
        save_predictions_to_gml(
            args.gml_path,
            data,
            model,
            id2name,
            output_gml_path=os.path.join(output_dir, "predictions.gml"),
        )
    else:
        print("Skipping save_predictions_to_gml since it is not single GML input.")
        
    run_phase2(data, model, args, id2name)

    # data = load_GNNRE_full('data/Interconnected-Modules/adj_full.npz', 'data/Interconnected-Modules/feats.npy', 'data/Interconnected-Modules/class_map.json', 'data/Interconnected-Modules/role.json')
    # train_loader = GraphSAINTRandomWalkSampler(data, batch_size=3000, walk_length=3, shuffle=True, sample_coverage=50)
    # val_data = data 
    # test_data = data

    # in_dim = data.num_features
    # out_dim = len(torch.unique(data.y))
    # print("Input dimension:", in_dim)
    # print("Output dimension:", out_dim)
    # run_training(data, train_loader, in_dim, out_dim)
    # print("Training complete.")


    # # gml (GNNRE)
    # data =load_GNNRE_gmls("data/Interconnected-Modules/")
    # full_data = Batch.from_data_list(data)
    # train_loader = GraphSAINTRandomWalkSampler(full_data, batch_size=3000, walk_length=2, shuffle=True, sample_coverage=50)
    # val_data = full_data 
    # test_data = full_data
    # in_dim = full_data.num_features
    # out_dim = len(torch.unique(full_data.y))
    # print("Input dimension:", in_dim)
    # print("Output dimension:", out_dim)
    # run_training(full_data, train_loader, in_dim, out_dim)
    # print("Training complete.")
