import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


############ normalizing 
from sklearn.preprocessing import StandardScaler
def normalize_features(features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.tensor(features, dtype=torch.float)


from torch_geometric.data import Data
def load_adj(path):
    adj = sp.load_npz(path)
    adj = adj + adj.T
    adj[adj>1]=1
    return adj 

def adj_to_edge_index(adj):
    coo = adj.tocoo()
    edge_index = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
    return edge_index

def load_labels(path):
    with open(path, 'r') as f:
        class_map = json.load(f)
    labels = [class_map[str(i)] for i in range(len(class_map))]
    return torch.tensor(labels, dtype=torch.long)

def load_role_splits(path):
    with open(path, 'r') as f:
        role_splits = json.load(f)
    test_idx = torch.tensor(role_splits['tr'], dtype=torch.long)
    train_idx = torch.tensor(role_splits['te'], dtype=torch.long)
    val_idx = torch.tensor(role_splits['va'], dtype=torch.long)
    return test_idx, train_idx, val_idx

def load_GNNRE_full(adj_path, feat_path, label_path, role_path):
    adj = load_adj(adj_path)
    edge_index = adj_to_edge_index(adj)
    features = np.load(feat_path)
    labels = load_labels(label_path)
    train_idx, test_idx, val_idx = load_role_splits(role_path)

    features = normalize_features(features)
    
    data = Data(x=features, edge_index=edge_index, y=labels)
    data.train_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    data.test_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    data.val_mask = torch.zeros(labels.size(0), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def all_gml_files(path):
    gml_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.gml'):
                gml_files.append(os.path.join(root, file))
    return gml_files


def load_GNNRE_gmls(gml_path):
    gml_files = all_gml_files(gml_path)
    print("number of gml files", len(gml_files))

    data_list = []

    for gml_file in gml_files:
        G = nx.read_gml(gml_file, label='id')

        nodes = sorted(G.nodes())
        features = []
        labels = []
        for node in nodes:
            attr = G.nodes[node]
            feat = list(map(int, attr['features'].strip('[]').split(',')))
            features.append(feat)
            labels.append(int(attr['class_label']))

        keep_cols = list(range(11)) + [20, 21]
        features = np.array(features)
        features = features[:, keep_cols]
        features = normalize_features(features)
        labels = torch.tensor(labels, dtype=torch.long)

        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        num_nodes = len(nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        gml_lower = gml_file.lower()
        if "train" in gml_lower:
            train_mask[:] = True
        elif "validate" in gml_lower:
            val_mask[:] = True
        elif "test" in gml_lower:
            test_mask[:] = True
            
        data = Data(x=features, edge_index=edge_index, y=labels)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data_list.append(data)
    
    return data_list


import random
def load_GNN_aes_core_gmls(gml_path):
    gml_files = all_gml_files(gml_path)
    print("number of AES core gmls", len(gml_files))

    gml_files_parsed = [f for f in gml_files if '_parsed.gml' in f]
    print("number of AES core gmls after filtering", len(gml_files_parsed))

    random.seed(42)
    random.shuffle(gml_files_parsed)

    test_files = gml_files[:3]
    val_files = gml_files[3:6]
    train_files = gml_files[6:]
    print("train files", len(train_files), "val files", len(val_files), "test files", len(test_files))

    data_list = []
    for gml_file in gml_files_parsed:
        G = nx.read_gml(gml_file, label='id')

        nodes = sorted(G.nodes())
        features = []
        labels = []
        for node in nodes:
            attr = G.nodes[node]
            feat = list(map(int, attr['features']))
            features.append(feat)
            labels.append(int(attr['class_label']))

        features = normalize_features(features)
        labels = torch.tensor(labels, dtype=torch.long)

        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        num_nodes = len(nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if gml_file in train_files:
            train_mask[:] = True
        elif gml_file in val_files:
            val_mask[:] = True
        elif gml_file in test_files:
            test_mask[:] = True

        data = Data(x=features, edge_index=edge_index, y=labels)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data_list.append(data)


def load_aisec_single_gml(gml_path):
    print("calling gnn from path:", gml_path)
    random.seed(42)
    
    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    subcircuit_ids = []
    for node in nodes:
        attr = G.nodes[node]

        node_feats = attr['features']
        if isinstance(node_feats, list):
            feat = list(map(int, node_feats))
        else:
            feat = [int(v) for v in G.nodes[node].get("features", [])]
        features.append(feat)

        subcircuit = attr.get('subcircuit_id') or 'unknown'
        subcircuit_ids.append(subcircuit)

    unique_subcircuits = sorted(set(subcircuit_ids))
    subcircuit2id = {name: idx for idx, name in enumerate(unique_subcircuits)}
    labels = torch.tensor([subcircuit2id[s] for s in subcircuit_ids], dtype=torch.long)

    features = normalize_features(np.array(features))
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1

    train_cutoff = int(train_ratio * num_nodes)
    val_cutoff = train_cutoff + int(val_ratio * num_nodes)

    train_idx = indices[:train_cutoff]
    val_idx = indices[train_cutoff:val_cutoff]
    test_idx = indices[val_cutoff:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.train_mask = train_mask
    data.val_mask = val_mask 
    data.test_mask = test_mask 

    return data


def load_aisec_multiple_gmls(gml_paths, label_type="subcircuit", binary_label=False, positive_class=None):
    """
    Load and combine multiple GML files into a single PyG Data object.
    No internal random split is done here — all nodes get train_mask=True.
    The caller (run_phase1) is responsible for setting the correct mask
    (train / val / test) on the returned object.
    """
    from collections import Counter

    print(f"Loading {len(gml_paths)} GMLs and combining...")

    all_features = []
    all_raw_labels = []
    all_edges = []
    offset = 0

    for path in gml_paths:
        G = nx.read_gml(path, label="id")
        nodes = list(G.nodes())
        node_map = {node: idx + offset for idx, node in enumerate(nodes)}

        for node in nodes:
            attr = G.nodes[node]

            feat = list(map(int, attr.get("features", [])))
            all_features.append(feat)

            # Read subcircuit_name (actual GML attribute), fall back gracefully
            sc_name = str(attr.get("subcircuit_name", "")).strip().strip("'")
            if not sc_name:
                sc_name = str(attr.get("subcircuit_original", "unknown")).strip()
            if not sc_name:
                sc_name = "unknown"
            all_raw_labels.append(sc_name)

        for src, dst in G.edges():
            all_edges.append((node_map[src], node_map[dst]))

        offset += len(nodes)

    all_features = normalize_features(np.array(all_features))

    # Build label map from raw string names
    label_counts_raw = Counter(all_raw_labels)
    print("Global label counts:", dict(label_counts_raw))

    label_set = sorted(set(all_raw_labels))
    label_map = {v: i for i, v in enumerate(label_set)}
    id2label_global = {i: v for i, v in enumerate(label_set)}

    encoded_labels = [label_map[l] for l in all_raw_labels]
    labels = torch.tensor(encoded_labels, dtype=torch.long)

    label_counts_enc = Counter(encoded_labels)
    print("\n=== Global Label Mapping ===")
    for i, lname in id2label_global.items():
        count = label_counts_enc.get(i, 0)
        print(f"{i}: {lname} — {count} nodes")

    # Binary classification
    if binary_label:
        if positive_class is None:
            raise ValueError("You must provide --positive_class in binary mode.")

        positive_class_id = None
        if isinstance(positive_class, str):
            if positive_class in label_map:
                positive_class_id = label_map[positive_class]
            else:
                print(f"[INFO] Positive class '{positive_class}' not found in this graph — all nodes are negative.")
        else:
            if positive_class in id2label_global:
                positive_class_id = positive_class
            else:
                print(f"[INFO] Positive class ID {positive_class} not found in this graph — all nodes are negative.")

        if positive_class_id is not None:
            print(f"Binary classification: Positive class = '{id2label_global[positive_class_id]}' (label ID: {positive_class_id})")
            labels = torch.where(
                labels == positive_class_id,
                torch.tensor(1, dtype=torch.long),
                torch.tensor(0, dtype=torch.long)
            )
        else:
            labels = torch.zeros(len(encoded_labels), dtype=torch.long)

        id2label_global = {0: "negative", 1: "positive"}
        print("After binarization:", Counter(labels.tolist()))
        
    if all_edges:
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    num_nodes = len(all_features)

    # All nodes active — caller sets the appropriate mask
    full_mask = torch.ones(num_nodes, dtype=torch.bool)
    zero_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data = Data(
        x=all_features.clone().detach().float(),
        edge_index=edge_index,
        y=labels,
        train_mask=full_mask,
        val_mask=zero_mask,
        test_mask=zero_mask,
    )

    return data, id2label_global


if __name__ == "__main__":
    load_GNN_aes_core_gmls("data/aes_core")