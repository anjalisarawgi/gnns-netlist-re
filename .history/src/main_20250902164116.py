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
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Batch
import random

def load_aisec_single_gml(gml_path):
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

    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1
    

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
    
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes

    return total_loss / len(loader.dataset)  # avg loss per sample


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    correct = (pred[mask] == data.y[mask]).sum().item()
    accuracy = correct / mask.sum().item() 
    return accuracy
    # correct = 0
    # total = 0 
    # for data in data_loader:
    #     out = model(data.x, data.edge_index)
    #     pred = out.argmax(dim=1)
    #     correct += (pred == data.y).sum().item()
    #     total += data.y.size(0)
    # return correct / total if total > 0 else 0

def classwise_accuracy(model, data, mask, id2name=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    y_true = data.y[mask]
    y_pred = pred[mask]
    num_classes = len(torch.unique(data.y))
    acc_per_class = {}

    for c in range(num_classes):
        mask_c = y_true == c 
        total_c = mask_c.sum().item()
        correct_c = (y_pred[mask_c] == c).sum().item()  
        acc = correct_c / total_c if total_c > 0 else 0
        # acc_per_class[c] = acc
        label_name = id2name[c] if id2name else f"Class {c}"
        acc_per_class[label_name] = acc

    return acc_per_class

def log_class_distribution(y, name=""):
    values, counts = torch.unique(y, return_counts=True)
    print(f"Class distribution in {name}:")
    for v, c in zip(values.tolist(), counts.tolist()):
        print(f"  Class {v}: {c} nodes")

def run_training(data, train_loader, in_dim, out_dim, id2name=None):
    model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    # model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 ) # weight_decay=5e-4

    epochs = 100
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy : {train_acc:.4f}, val accuracy {val_acc:.4f}')

        if epoch % 100 == 0 or epoch == 1:
            classwise_acc = classwise_accuracy(model, data, data.val_mask, id2name)
            print("  Val Class-wise Accuracy:")
            for cls, acc in classwise_acc.items():
                print(f"    Class {cls}: {acc:.4f}")

    test_acc = evaluate(model, data, data.test_mask)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    classwise_acc = classwise_accuracy(model, data, data.test_mask, id2name)
    print("  Test Class-wise Accuracy:")
    for cls, acc in classwise_acc.items():
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

    nx.write_gml(G, output_gml_path)
    print(f"Saved GML with predictions to: {output_gml_path}")

if __name__ == "__main__":
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
    data, id2name = load_aisec_single_gml("aes_cipher_top_gephi_features.gml")

    in_dim = data.num_features
    out_dim = len(torch.unique(data.y))
    print("in_dim", in_dim)
    print("out_dim", out_dim)

    log_class_distribution(data.y[data.train_mask], "train")
    log_class_distribution(data.y[data.val_mask], "val")
    log_class_distribution(data.y[data.test_mask], "test")
    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=5000, walk_length=5, shuffle=True)
    model = run_training(data, train_loader, in_dim, out_dim, id2name)    

    # Save predictions to a new GML for Gephi analysis
    save_predictions_to_gml(
        original_gml_path="aes_cipher_top_gephi_features.gml",
        data=data,
        model=model,
        id2name=id2name,
        output_gml_path="aes_cipher_top_gephi_features_results.gml"
    )