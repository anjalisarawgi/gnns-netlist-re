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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import json 
from collections import defaultdict, Counter

subcircuit_map = defaultdict(set) ### ???


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
    

def plot_tsne(data, labels, id2name, title="t-SNE of Node Features", save_path="tsne.png"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z = tsne.fit_transform(data)

    label_names = [id2name[label] for label in labels]
    unique_labels = sorted(set(label_names))

    plt.figure(figsize=(10, 8))
    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(label_names) if l == lbl]
        plt.scatter(z[idxs, 0], z[idxs, 1], label=lbl, s=10, alpha=0.6)

    plt.title(title)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    print(f"Saved t-SNE plot to: {save_path}")

def train(model, loader, optimizer):
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

def run_training(data, train_loader, in_dim, out_dim, id2name=None, model_name="graphsage"):
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "GCN":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    elif model_name == "GAT":
        model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 ) # weight_decay=5e-4

    epochs = 500
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer)
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



if __name__ == "__main__":
    setup_logging("logs")
    set_seed(42)
    parser = argparse.ArgumentParser(description="GNN for Subcircuit Detection")
    parser.add_argument("--gml_path", type=str, default="aes_key_expand_features.gml", help="Path to the input GML file")
    parser.add_argument("--model", type=str, default="graphsage", choices=["graphsage", "GCN", "GAT"], help="GNN model to use")
    parser.add_argument("--class_reduce", action="store_true", help="Whether to reduce classes or not", default = False)
    args = parser.parse_args()

    gml_path = args.gml_path
    model_name = args.model
    
    wandb.init(
        project="gnn-subcircuit-detection",
        name=f"{model_name}-{os.path.basename(gml_path).replace('.gml', '')}-class_reduce-{args.class_reduce}",  
        # config={
        #     "model": model_name,
        #     # "hidden_channels": 256,
        #     # "lr": 0.01,
        #     # "epochs": 1000,
        #     # "batch_size": 5000,
        #     # "walk_length": 3,
        #     # "dataset": os.path.basename(gml_path)
        # }
    )

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

    # # gml (AES)
    # data = load_GNN_aes_core_gmls("data/aes_core/")
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




    ##### gml (AES LOAD SINGLE FILE)
    data, id2name = load_aisec_single_gml(gml_path, class_reduce=args.class_reduce)

    in_dim = data.num_features
    out_dim = len(torch.unique(data.y))
    print("in_dim", in_dim)
    print("out_dim", out_dim)

    log_class_distribution(data.y[data.train_mask], "train")
    log_class_distribution(data.y[data.val_mask], "val")
    log_class_distribution(data.y[data.test_mask], "test")

    random.seed(42)
    num_nodes = data.num_nodes
    batch_size = int(0.3 * num_nodes)
    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2, shuffle=True)




    model = run_training(data, train_loader, in_dim, out_dim, id2name, model_name)    
    
    # os.makedirs("results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(gml_path))[0]
    output_dir = os.path.join("results", f"{model_name}_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_gml_path = os.path.join(output_dir, "predictions.gml")
    tsne_path = os.path.join(output_dir, "tsne.png")

    # Save predictions to a new GML for Gephi analysis
    save_predictions_to_gml(
        original_gml_path=gml_path,
        data=data,
        model=model,
        id2name=id2name,
        output_gml_path=  output_gml_path
        # output_gml_path=gml_path.replace(".gml", "_results.gml")
    )

    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()

    # plot_tsne(features, labels, id2name, title=f"t-SNE: {base_name} ({model_name})", save_path=tsne_path)
    # print(f"DONE! GML + t-SNE saved to: {output_dir}")
    print("DONE with phase1!!!")

    

    ####################################################
    # Phase 2
    print("Starting phase 2...")
    subcircuit_json_path = "results/aes_cipher_top_gephi/subcircuit_map.json"
    with open(subcircuit_json_path, "r") as f:
        subcircuit_map = json.load(f)
    print(f"Loaded subcircuit map from: {subcircuit_json_path}")
    print(json.dumps(subcircuit_map, indent=2))

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    ##### ???
    target_subcircuit_label = 3# or "top+us_..." if class_reduce=True
    PHASE2_CLASS = [i for i, name in id2name.items() if name == target_subcircuit_label][0]
    print(f"Target class index: {PHASE2_CLASS} ({id2name[PHASE2_CLASS]})")
    phase2_mask = pred == PHASE2_CLASS
    
    G = nx.read_gml(gml_path, label="id")
    node_list = list(G.nodes())
    def is_valid_subcircuit(sc_id):
        return sc_id.startswith("top+us") and not "round2" in sc_id
        # return sc_id.startswith("top+u0+u") 
        # return sc_id == "top+u0"
        # return True

    subcircuit_ids = []
    phase2_indices = []
    for idx in torch.where(phase2_mask)[0]:
        node_idx = idx.item()
        node_name = node_list[node_idx]
        sc_id = G.nodes[node_name].get("subcircuit_id", "unknown")
        if is_valid_subcircuit(sc_id):
            subcircuit_ids.append(sc_id)
            phase2_indices.append(node_idx)

    ### 
    k = 5
    scid_counts = Counter(subcircuit_ids)
    # top_k_subcircuits = set([s for s, _ in scid_counts.most_common(k)])
    top_k_subcircuits = set(scid_counts.keys())  # keep all

    filtered_indices = []
    filtered_labels = []
    for idx, sid in zip(phase2_indices, subcircuit_ids):
        if sid in top_k_subcircuits:
            filtered_indices.append(idx)
            filtered_labels.append(sid)
    phase2_indices = filtered_indices
    subcircuit_ids = filtered_labels

    # Map subcircuit_id -> integer class
    unique_ids = sorted(set(subcircuit_ids))
    id_map = {name: i for i, name in enumerate(unique_ids)}
    id2name_phase2 = {i: name for name, i in id_map.items()}

    # Build label vector for the entire graph (others get -1)
    new_y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for idx, scid in zip(phase2_indices, subcircuit_ids):
        new_y[idx] = id_map[scid]

    # # Create train/val/test masks
    # phase2_train_mask = torch.zeros_like(new_y, dtype=torch.bool)
    # phase2_val_mask = torch.zeros_like(new_y, dtype=torch.bool)
    # phase2_test_mask = torch.zeros_like(new_y, dtype=torch.bool)

    # indices = phase2_indices
    # random.shuffle(indices)
    # n = len(indices)
    # t_split = int(0.6 * n)
    # v_split = int(0.8 * n)
    # phase2_train_mask[indices[:t_split]] = True
    # phase2_val_mask[indices[t_split:v_split]] = True
    # phase2_test_mask[indices[v_split:]] = True

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

    # Create PyG data object
    phase2_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        y=new_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print("Total nodes in phase 2:", len(phase2_indices))
    print("Label counts:", torch.unique(new_y[new_y != -1], return_counts=True))
    print("Train size:", train_mask.sum().item())
    print("Train size:", val_mask.sum().item())
    print("Test size:", test_mask.sum().item())

    log_class_distribution(phase2_data.y[phase2_data.train_mask], "PHASE 2 train")
    log_class_distribution(phase2_data.y[phase2_data.val_mask], "PHASE 2 val")
    log_class_distribution(phase2_data.y[phase2_data.test_mask], "PHASE 2 test")

    # phase2_loader = [phase2_data] ### ???
    phase2_loader = GraphSAINTRandomWalkSampler(
        phase2_data,
        batch_size=int(0.3 * phase2_data.num_nodes),
        walk_length=2,
        shuffle=True,
        # sample_coverage=50
    )


    print("Running phase 2 training...")
    phase2_model = run_training(
        phase2_data,
        train_loader=phase2_loader,
        in_dim=in_dim,
        out_dim=len(unique_ids),
        id2name=id2name_phase2,
        model_name=model_name
    )


    # print("\n==========================")
    # print("🚀 Starting Phase 2")
    # print("==========================")

    # target_class = 1  # the predicted class from Phase 1
    # G = nx.read_gml(gml_path, label="id")
    # node_list = list(G.nodes())

    # # Get all nodes where Phase 1 model predicted class 3
    # model.eval()
    # out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)
    # phase2_mask = pred == target_class
    # phase2_indices = torch.where(phase2_mask)[0]

    # print(f"Total nodes predicted as class {target_class}: {len(phase2_indices)}")

    # # Get true subcircuit_ids for these nodes
    # subcircuit_ids = []
    # for idx in phase2_indices:
    #     node_idx = idx.item()
    #     node_name = node_list[node_idx]
    #     sc_id = G.nodes[node_name].get("subcircuit_id", "unknown")
    #     subcircuit_ids.append(sc_id)

    # # Debug print a few examples
    # print("Sample of nodes predicted as class 3:")
    # for i in range(min(10, len(phase2_indices))):
    #     node_idx = phase2_indices[i].item()
    #     node_name = node_list[node_idx]
    #     sc_id = subcircuit_ids[i]
    #     true_label = data.y[node_idx].item()
    #     print(f"  Node {node_idx} ({node_name}): true_label={true_label}, subcircuit_id={sc_id}")

    # # Map subcircuit_ids to class labels for Phase 2
    # unique_ids = sorted(set(subcircuit_ids))
    # id_map = {name: i for i, name in enumerate(unique_ids)}
    # id2name_phase2 = {i: name for name, i in id_map.items()}

    # print("Unique subcircuit_ids found:")
    # for i, sid in enumerate(unique_ids):
    #     print(f"  Class {i}: {sid}")

    # # Assign new labels
    # new_y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    # for idx, scid in zip(phase2_indices, subcircuit_ids):
    #     new_y[idx.item()] = id_map[scid]



    # # Split into train/val/test
    # valid_indices = [i.item() for i in phase2_indices]
    # random.shuffle(valid_indices)

    # n = len(valid_indices)
    # t_split = int(0.6 * n)
    # v_split = int(0.8 * n)

    # train_mask = torch.zeros_like(new_y, dtype=torch.bool)
    # val_mask = torch.zeros_like(new_y, dtype=torch.bool)
    # test_mask = torch.zeros_like(new_y, dtype=torch.bool)

    # train_mask[valid_indices[:t_split]] = True
    # val_mask[valid_indices[t_split:v_split]] = True
    # test_mask[valid_indices[v_split:]] = True

    # # Quick Sanity Check: Unique subcircuit_ids used in Phase 2 training
    # train_label_ids = new_y[train_mask]
    # train_subcircuits = [id2name_phase2[label.item()] for label in train_label_ids]
    # counts = Counter(train_subcircuits)

    # print("Phase 2 - UNIQUE subcircuit_ids used in TRAINING:")
    # for subcircuit, count in counts.items():
    #     label_id = id_map[subcircuit]
    #     print(f"  Label ID: {label_id:2d} | Subcircuit ID: {subcircuit:30s} | Nodes: {count}")


    # print(f"Phase 2 label distribution:")
    # vals, counts = torch.unique(new_y[new_y != -1], return_counts=True)
    # for v, c in zip(vals.tolist(), counts.tolist()):
    #     print(f"  Class {v} ({id2name_phase2[v]}): {c} nodes")

    # print(f"Split sizes:")
    # print(f"  Train: {train_mask.sum().item()} nodes")
    # print(f"  Val:   {val_mask.sum().item()} nodes")
    # print(f"  Test:  {test_mask.sum().item()} nodes")

    # # Create PyG Data object
    # phase2_data = Data(
    #     x=data.x,
    #     edge_index=data.edge_index,
    #     y=new_y,
    #     train_mask=train_mask,
    #     val_mask=val_mask,
    #     test_mask=test_mask
    # )

    # # Log class distribution
    # log_class_distribution(phase2_data.y[phase2_data.train_mask], "PHASE 2 train")
    # log_class_distribution(phase2_data.y[phase2_data.val_mask], "PHASE 2 val")
    # log_class_distribution(phase2_data.y[phase2_data.test_mask], "PHASE 2 test")

    # # Create loader
    # phase2_loader = GraphSAINTRandomWalkSampler(
    #     phase2_data,
    #     batch_size=int(0.3 * phase2_data.num_nodes),
    #     walk_length=2,
    #     shuffle=True,
    # )

    # # Train model
    # print("Running Phase 2 training...")
    # phase2_model = run_training(
    #     phase2_data,
    #     train_loader=phase2_loader,
    #     in_dim=in_dim,
    #     out_dim=len(unique_ids),
    #     id2name=id2name_phase2,
    #     model_name=model_name
    # )
