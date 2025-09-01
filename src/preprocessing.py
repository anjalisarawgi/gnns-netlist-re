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



# elif dataset == "GNNRE_full":
    # data - full graph 
from torch_geometric.data import Data
def load_adj(path):
    adj = sp.load_npz(path)
    adj = adj + adj.T # make it symmetric
    adj[adj>1]=1 # make it binary 
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



    # keep_cols = keep_cols = list(range(11))+ [20, 21] # trying to see if i can drop it 
    # features = features[:, keep_cols]
    # print("one sample of the features", features[0])

    # normalize 
    features = normalize_features(features)
    
    data = Data(x=features, edge_index = edge_index, y = labels)
    data.train_mask = torch.zeros(labels.size(0), dtype=torch.bool) # creating masks as long as the number of nodes
    data.test_mask = torch.zeros(labels.size(0), dtype=torch.bool) # creating masks as long as the number of nodes
    data.val_mask = torch.zeros(labels.size(0), dtype=torch.bool) # creating masks as long as the number of nodes

    data.train_mask[train_idx]  = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data




# ### if by gml

# all gml files
def all_gml_files(path):
    gml_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.gml'):
                gml_files.append(os.path.join(root, file))
    return gml_files

# print(all_gml_files("data/Interconnected-Modules/"))
# we have 37 gmls 

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

        # drop features
        keep_cols = list(range(11))+ [20, 21] # trying to see if i can drop it 
        features = np.array(features)
        features = features[:, keep_cols]
        # print("one sample of the features", features[0])
        features = normalize_features(features)
        labels = torch.tensor(labels, dtype=torch.long)

        # edge
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # role based on file names 
        num_nodes = len(nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        gml_lower = gml_file.lower()
        if "train" in gml_lower:
            train_mask[:]   = True
        elif "validate" in gml_lower:
            val_mask[:] = True
        elif "test" in gml_lower:
            test_mask[:] = True
            
        # pyg
        data = Data(x=features, edge_index=edge_index, y=labels)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data_list.append(data)
    
    return data_list


#### aes core files
import random
def load_GNN_aes_core_gmls(gml_path):
    gml_files = all_gml_files(gml_path)
    print("number of AES core gmls", len(gml_files))

    # consider gml with only _parsed.gml
    gml_files_parsed = [f for f in gml_files if '_parsed.gml' in f]
    print("number of AES core gmls after filtering", len(gml_files_parsed))


    # we want to split the files into train, val, test - based on the file numbers
    random.seed(42)  # for reproducibility
    random.shuffle(gml_files_parsed)  # shuffle the files

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
            # feat = list(map(int, attr['features'].strip('[]').split(','))) 
            feat = list(map(int, attr['features']))
            features.append(feat)
            labels.append(int(attr['class_label']))


        # print("one sample of the features", features[0])
        features = normalize_features(features)
        labels = torch.tensor(labels, dtype=torch.long)

        # edge
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

        # pyg
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


    return data







if __name__ == "__main__":
    load_GNN_aes_core_gmls("data/aes_core")