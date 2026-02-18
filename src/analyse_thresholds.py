import torch
import joblib
import numpy as np
import networkx as nx
import json
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from gnn.gat import gat
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.graphTransformer import GraphTransformer
from collections import Counter
import random

def load_single_gml(
    gml_path,
    remove_edges=False,
    use_partition_features=False,
    use_graph_features=False
):
    print("[INFO] Calling gml from path:", gml_path)
    
    G = nx.read_gml(gml_path, label = "id") 
    nodes = list(G.nodes()) # list of node ids

    features = []
    labels = []

    base_feat_dim = None
    after_partition_dim = None
    after_graph_dim = None
    for node in nodes:
        attr = G.nodes[node] # attr?
        feat = attr.get("features", [])
        if base_feat_dim is None:
            base_feat_dim = len(feat)

        if use_partition_features:     
            partition_feat = attr.get("partition_features", [0.0, 0.0])
            # partition_feat_twoHop = partition_feat[1]
            feat = list(feat) +list(partition_feat)
            # feat = list(feat) + [float(partition_feat_twoHop)]
            if after_partition_dim is None:
                after_partition_dim = len(feat)

        if use_graph_features:
            graph_feat = attr.get("graph_features", [0.0, 0.0])
            graph_feat_subset = graph_feat[:2]
            feat = list(feat) +list(graph_feat_subset)
            if after_graph_dim is None:
                after_graph_dim = len(feat)


        if not isinstance(feat, (list, tuple, np.ndarray)):
            raise ValueError(f"Node {node} has invalid features")

        features.append(feat)
    
        boundary_value = attr.get("boundary", 0) # a boundary with no label for boundary gets boundary = 0 (note: essentially this is simply input output node and we want to use it as a no boundary node)
        try:
            label = int(boundary_value)
        except (ValueError, TypeError): ######??? - we wanna change this ***s
            label= 0
        labels.append(label)

    id2label = {0: "not_boundary", 1:"boundary"}
    labels = torch.tensor(labels, dtype = torch.long)
    labels[labels == -1] = 0     # treating -1 as label boundary =  0 i.e. not treaitng this as a boundary node


    unique_classes, class_counts = np.unique(labels.cpu().numpy(), return_counts = True)
    for u,c in zip(unique_classes, class_counts):
        print(f"Class {u} ({id2label.get(int(u), '?')}): {c} samples")
    print("[INFO] Total nodes:", len(nodes))
    print("[INFO] Label tensor shape:", labels.shape, "Data Type:", labels.dtype)
    print("[INFO] Total labels:", len(labels))
    print("[INFO] Unique labels:", sorted(set(labels.tolist())))
    print("[INFO] Label Counts:", Counter(labels.tolist()) )

    # node map is like a lookup (between NetworkX and pyG) ???
    # and we want edge_index to be of shape for pyg: [2, num_edges]
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 

    # edge features
    edge_attrs = []
    for src, dst in G.edges():
        attr = G[src][dst]
        e_feat = attr.get("edge_features", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(e_feat, (list, tuple, np.ndarray)):
            raise ValueError(f"Edge ({src},{dst}) has invalid edge_features")
        edge_attrs.append(e_feat)

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    # print("\n[DEBUG] FIRST EDGE TENSOR VERSION")
    # print("edge_attr[0]:", edge_attr[0])
    # print("edge_attr shape:", edge_attr.shape)
    # print("=================================\n")

    # splits --- train / test / val s
    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    #### this can be optional but we did thsi to remove the edges if it is connecting to for example a test node
    ### i.e. removing the edges that connects to diffefernt splits 
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


    features = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)

    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,  
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, id2label


def load_model(model_path, metadata_path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    in_dim = meta["final_feature_dim"]

    if model_type == "gat":
        model = gat(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "gcn":
        model = GCN(in_channels=in_dim, hidden_channels=256, out_channels=2)
    elif model_type == "graphTransformer":
        model = GraphTransformer(in_channels=in_dim, hidden_channels=256, out_channels=2)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

@torch.no_grad()
def get_probs(model, data):
    out = model(data.x, data.edge_index, data.edge_attr)
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    labels = data.y.cpu().numpy()
    return probs, labels

def sweep_thresholds(probs, labels):
    thresholds = np.linspace(0.01, 0.99, 99)

    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        predicted_ratio = preds.mean()
        true_ratio = labels.mean()

        results.append({
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "predicted_ratio": float(predicted_ratio),
            "true_ratio": float(true_ratio),
        })

    return results



if __name__ == "__main__":

    model_dir = "models/wPartitions/fullgraph_per_design_ce_soft_500ep_for_aes_128_combined_m1+aes_inv_cipher_top_combined_m1+more"
    model_path = os.path.join(model_dir, "model.pt")
    metadata_path = os.path.join(model_dir, "metadata.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    test_graphs = [
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/tiny_aes_latest/osu035/aes_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_inv_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/nangate/aes_key_expand_128_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/sha1-master/osu035/sha1_core_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/des_latest/gscl45nm/des_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_key_expand_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/trigonometric_functions_in_double_fpu_latest/gscl45nm/top_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/mips32r1_latest/gscl45nm/ALU_combined_m1.gml",
    ]

    os.makedirs("thresholds", exist_ok=True)

    # loading model and scaler to normalize
    model = load_model(model_path, metadata_path)
    scaler = joblib.load(scaler_path)
    all_best_thresholds = {}
    for gml_path in test_graphs:
        print("Processing:", gml_path)
        data, _ = load_single_gml(gml_path,remove_edges=False, use_partition_features=True,  use_graph_features=False)
        data.x = torch.tensor(scaler.transform(data.x.cpu().numpy()),dtype=torch.float32)

        probs, labels = get_probs(model, data)
        results = sweep_thresholds(probs, labels)
        parts = os.path.splitext(gml_path)[0].split(os.sep)
        graph_name = "__".join(parts[-3:])  

        # beest results
        best_result = max(results, key=lambda x: x["f1"])
        all_best_thresholds[graph_name] = {
            "gml_path": gml_path,
            "threshold": best_result["threshold"],
            "f1": best_result["f1"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "predicted_ratio": best_result["predicted_ratio"],
            "true_ratio": best_result["true_ratio"],
        }
                
        save_path = os.path.join("thresholds", f"{graph_name}.json")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        print("Saved:", save_path)

    best_summary_path = os.path.join("thresholds", "best_thresholds_summary.json")
    with open(best_summary_path, "w") as f:
        json.dump(all_best_thresholds, f, indent=2)

    print("Saved global best-threshold summary:", best_summary_path)