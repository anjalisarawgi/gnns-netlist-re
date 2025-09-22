import torch 
from preprocessing import normalize_features, load_GNNRE_full, load_GNNRE_gmls, load_GNN_aes_core_gmls,load_aisec_single_gml
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


subcircuit_map = defaultdict(set) 


def load_aisec_single_gml(gml_path, class_reduce):
    print("calling gnn from path:", gml_path)
    random.seed(42)
    
    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    subcircuit_ids = []
    for node in nodes:
        attr = G.nodes[node]

        # making features as a list
        node_feats = attr['features']
        if isinstance(node_feats, list):
            feat = list(map(int, node_feats))
        else:
            feat = [int(v) for v in G.nodes[node].get("features", [])]
        features.append(feat)

        #### here we choose if we want to reduce to lesser classes 
        if class_reduce == True:
            subcircuit = attr.get('subcircuit')
        else:
            subcircuit = attr.get('subcircuit_id') or 'unknown' 
        
        subcircuit_ids.append(subcircuit)

    # map subcircuit -> integer labels 
    unique_subcircuits = sorted(set(subcircuit_ids))
    subcircuit2id = {name: idx for idx, name in enumerate(unique_subcircuits)}
    labels = torch.tensor([subcircuit2id[s] for s in subcircuit_ids], dtype=torch.long)

    features = normalize_features(np.array(features)) # normalize
    # edges = list(G.edges())
    # edge_index = torch.tensor(edges, dtype = torch.long).t().contiguous()
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_ratio, test_ratio, val_ratio = 0.6, 0.2, 0.2
    

    train_cutoff = int(train_ratio*num_nodes )
    val_cutoff = train_cutoff + int(val_ratio* num_nodes)

    train_idx = indices[:train_cutoff]
    val_idx = indices[train_cutoff:val_cutoff]
    test_idx = indices[val_cutoff:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # creating pyG object 
    data = Data(x=features, edge_index=edge_index, y = labels)
    data.train_mask = train_mask
    data.val_mask = val_mask 
    data.test_mask = test_mask 
    
    id2subcircuit = {idx: name for name, idx in subcircuit2id.items()}
    
    return data, id2subcircuit

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

def run_training(data, train_loader, in_dim, out_dim, id2name=None, model_name="graphsage", use_weighted_loss=False):
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "GCN":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    elif model_name == "GAT":
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

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy : {train_acc:.4f}, val accuracy {val_acc:.4f}')

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
    data, id2name = load_aisec_single_gml(args.gml_path, class_reduce=args.class_reduce)

    in_dim = data.num_features
    out_dim = len(torch.unique(data.y))
    print("in_dim", in_dim)
    print("out_dim", out_dim)

    log_class_distribution(data.y[data.train_mask], "train")
    log_class_distribution(data.y[data.val_mask], "val")
    log_class_distribution(data.y[data.test_mask], "test")

    batch_size = int(0.3 * data.num_nodes)
    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2, shuffle=True)

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
    parser.add_argument("--model", type=str, default="graphsage", choices=["graphsage", "GCN", "GAT"], help="GNN model to use")
    parser.add_argument("--class_reduce", action="store_true", help="Whether to reduce classes or not", default=False)
    parser.add_argument("--weighted_loss", action="store_true", help="Use class-weighted cross entropy loss")
    args = parser.parse_args()

    wandb.init(project="gnn-subcircuit-detection", name=f"{args.model}-{os.path.basename(args.gml_path).replace('.gml', '')}-class_reduce-{args.class_reduce}")

    data, model, id2name = run_phase1(args)

    output_dir = os.path.join("results", f"{args.model}_{os.path.basename(args.gml_path).replace('.gml', '')}")
    os.makedirs(output_dir, exist_ok=True)
    save_predictions_to_gml(args.gml_path, data, model, id2name, output_gml_path=os.path.join(output_dir, "predictions.gml"))

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
