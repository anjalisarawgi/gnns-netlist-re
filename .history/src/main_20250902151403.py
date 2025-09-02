import torch 
from preprocessing import normalize_features, load_GNNRE_full, load_GNNRE_gmls,load_aisec_single_gml
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import random

def load_aisec_single_gml(gml_path):
    print("calling gnn from path:", gml_path)
    random.seed(42)
    
    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    subcircuit_ids = []
    valid_nodes = []
    for node in nodes:
        attr = G.nodes[node]
        label = attr.get('label', '').strip("'")
        if label.startswith("INPUT") or label.startswith("OUTPUT"):
            continue 

        # making features as a list
        node_feats = attr['features']
        feat = list(map(int, node_feats)) if isinstance(node_feats, list) else [int(v) for v in attr.get("features", [])]
        features.append(feat)

        subcircuit = attr.get('subcircuit_id') or 'unknown'
        subcircuit_ids.append(subcircuit)

        valid_nodes.append(node)

    # map subcircuit -> integer labels 
    unique_subcircuits = sorted(set(subcircuit_ids))
    subcircuit2id = {name: idx for idx, name in enumerate(unique_subcircuits)}
    labels = torch.tensor([subcircuit2id[s] for s in subcircuit_ids], dtype=torch.long)

    id2subcircuit = {idx: name for name, idx in subcircuit2id.items()}
    

    features = normalize_features(np.array(features)) # normalize
    # Build edge_index from valid nodes only
    node_map = {node: idx for idx, node in enumerate(valid_nodes)}
    edges = [
        (node_map[src], node_map[dst])
        for src, dst in G.edges()
        if src in node_map and dst in node_map
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(valid_nodes)
    print("num_nodes:", num_nodes)
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

    #  pyG object   
    data = Data(x=features, edge_index=edge_index, y = labels)
    data.train_mask = train_mask
    data.val_mask = val_mask 
    data.test_mask = test_mask 
    data.subcircuit_id_map = id2subcircuit 
    return data


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



def run_training(data, train_loader, in_dim, out_dim):
    model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    # model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 ) # weight_decay=5e-4

    epochs = 500
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train accuracy : {train_acc:.4f}, val accuracy {val_acc:.4f}')

    test_acc = evaluate(model, data, data.test_mask)
    print(f"Final test accuracy: {test_acc:.4f}")
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', choices=['aisec_subcircuit_classf', 'aisec_bianry_classf_us00_round2'], default='aisec_subcircuit_classf')
    parser.add_argument("--gml_path", type=str, default="aes_key_expand_128_modified_wfeatures.gml")

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



    ##### gml (AES LOAD SINGLE FILE)
    args = parser.parse_args()
    data = load_aisec_single_gml(gml_path, task_type= "aisec_bianry_classf_us00_round2")

    in_dim = data.num_features
    out_dim = len(torch.unique(data.y))
    print("in_dim", in_dim)
    print("out_dim", out_dim)

    train_loader = GraphSAINTRandomWalkSampler(data, batch_size=1400, walk_length=3, shuffle=True)
    # run_training(data, train_loader, in_dim, out_dim)    
    model = run_training(data, train_loader, in_dim, out_dim)  # ⬅️ capture model

    ###tsne 
    print("🔍 Visualizing node embeddings with t-SNE...")
    label_names = [data.subcircuit_id_map[int(label)] for label in data.y.cpu().numpy()]

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    z_tsne = tsne.fit_transform(embeddings)
    unique_names = sorted(set(label_names))
    name_to_color_idx = {name: i for i, name in enumerate(unique_names)}
    colors = [name_to_color_idx[name] for name in label_names]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors, cmap='tab20', s=5)
    handles = [
        plt.Line2D([], [], marker='o', color='w',
                markerfacecolor=cm.tab20(i / len(unique_names)),
                label=name, markersize=8)
        for i, name in enumerate(unique_names)
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE of GNN Node Embeddings (Colored by Subcircuit ID)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig("tsne_with_subcircuit_names.png")