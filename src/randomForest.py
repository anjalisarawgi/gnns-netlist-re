import os
import argparse
import yaml
import joblib
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict
import networkx as nx
import random
from torch_geometric.data import Data
from collections import Counter

# IMPORTANT: adjust these imports to your repo
from gnn.gat import gat


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
        # features.append(feat[:-1])
        # feat = feat[:15] + feat[16:] # skip feature at index 15
        # feat = feat[:-3] # skip last 3 features
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


    ## normalizing features *** ???
    # features = normalize_features(np.array(features, dtype = np.float32)) ### - this becomes one scaler for each graph

    # debugging for checking if everything is okay
    unique_classes, class_counts = np.unique(labels.cpu().numpy(), return_counts = True)
    # unique_classes, class_counts = torch.unique(labels, return_counts=True)
    # unique_classes = unique_classes.tolist()
    # class_counts = class_counts.tolist()

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

    # ---- EDGE FEATURES ----
    edge_attrs = []
    for src, dst in G.edges():
        attr = G[src][dst]
        e_feat = attr.get("edge_features", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(e_feat, (list, tuple, np.ndarray)):
            raise ValueError(f"Edge ({src},{dst}) has invalid edge_features")
        edge_attrs.append(e_feat)

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    print("\n[DEBUG] FIRST EDGE TENSOR VERSION")
    print("edge_attr[0]:", edge_attr[0])
    print("edge_attr shape:", edge_attr.shape)
    print("=================================\n")

    # splits --- train / test / val s
    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    # train_cutoff = int(0.80 * num_nodes) # ***
    # val_cutoff = train_cutoff + int(0.10 * num_nodes)
    
    # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # train_mask[indices[:train_cutoff]] = True
    # val_mask[indices[train_cutoff:val_cutoff]] = True
    # test_mask[indices[val_cutoff:]] = True
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
    ########################################

    # data = Data(
    #     # x = torch.tensor(featues, dtype = torch.float), 
    #     x = torch.as_tensor(features, dtype = torch.float32), 
    #     edge_index = edge_index,
    #     y = labels, 
    #     train_mask = train_mask, 
    #     val_mask = val_mask, 
    #     test_mask = test_mask
    # )

    features = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
    # data = Data(
    #     x = features,
    #     edge_index = edge_index,
    #     y = labels,
    #     train_mask = train_mask,
    #     val_mask = val_mask,
    #     test_mask = test_mask
    # )
    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,   # NEW
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print("[FEATURE DIM CHECK]")
    print("  base features dim           :", base_feat_dim)

    if use_partition_features:
        print("  after partition features   :", after_partition_dim)

    if use_graph_features:
        print("  after graph features       :", after_graph_dim)

    print("  final feature dim (tensor) :", data.x.shape[1])

    print("edge_index shape:", data.edge_index.shape)
    print("edge_attr  shape:", data.edge_attr.shape)

    assert data.edge_index.shape[1] == data.edge_attr.shape[0], \
        f"Mismatch: {data.edge_index.shape[1]} edges but {data.edge_attr.shape[0]} edge features"

    assert data.edge_index.max().item() < data.num_nodes, \
        "Edge index contains node id >= num_nodes"

    assert data.edge_index.min().item() >= 0, \
        "Edge index contains negative node ids"

    print("[EDGE CHECK] Passed ✔")

    return data, id2label



@torch.no_grad()
def extract_embeddings(model, data):
    model.eval()
    logits, emb = model(data.x, data.edge_index, data.edge_attr, return_embeddings=True)
    return emb.cpu().numpy(), data.y.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config used in training (contains train_gml/test_gml)")
    parser.add_argument("--run_dir", required=True, help="Path to run dir that contains model.pt + scaler.pkl")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    use_partition_features = cfg.get("use_partition_features", False)
    use_graph_features = cfg.get("use_graph_features", False)

    train_gml = cfg["train_gml"]
    test_gml = cfg["test_gml"]

    model_path = os.path.join(args.run_dir, "model.pt")
    scaler_path = os.path.join(args.run_dir, "scaler.pkl")

    assert os.path.exists(model_path), f"Missing: {model_path}"
    assert os.path.exists(scaler_path), f"Missing: {scaler_path}"

    print("[INFO] Loading scaler:", scaler_path)
    scaler = joblib.load(scaler_path)

    # figure out in_dim from first train graph
    tmp_data, _ = load_single_gml(
                    train_gml[0],
                    remove_edges=False,
                    use_partition_features=use_partition_features,
                    use_graph_features=use_graph_features
                )
    in_dim = tmp_data.num_features

    print("[INFO] Loading GAT model:", model_path)
    model = gat(in_channels=in_dim, hidden_channels=256, out_channels=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # --------- Build RF train set from TRAIN GRAPHS ----------
    X_train_list, y_train_list = [], []

    print("\n[INFO] Extracting TRAIN embeddings...")
    for p in train_gml:
        print("  -", p)
        data, _ = load_single_gml(
                    p,
                    remove_edges=False,
                    use_partition_features=use_partition_features,
                    use_graph_features=use_graph_features
                )

        # apply TRAIN scaler (the one saved during training)
        data.x = torch.tensor(
            scaler.transform(data.x.cpu().numpy()),
            dtype=torch.float32
        )

        X, y = extract_embeddings(model, data)
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    print("[INFO] RF train matrix:", X_train.shape, "labels:", y_train.shape)

    # --------- Train RF ----------
    print("\n[INFO] Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=args.seed,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    # save RF
    rf_path = os.path.join(args.run_dir, "rf_on_gat_embeddings.joblib")
    joblib.dump(rf, rf_path)
    print("[INFO] Saved RF to:", rf_path)

    # --------- Evaluate on TEST GRAPHS ----------
    print("\n[INFO] Evaluating on TEST graphs...")
    macro = defaultdict(list)

    for p in test_gml:
        name = os.path.splitext(os.path.basename(p))[0]
        data, _ = load_single_gml(
                p,
                remove_edges=False,
                use_partition_features=use_partition_features,
                use_graph_features=use_graph_features
            )

        data.x = torch.tensor(
            scaler.transform(data.x.cpu().numpy()),
            dtype=torch.float32
        )

        X_test, y_test = extract_embeddings(model, data)
        y_pred = rf.predict(X_test)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        macro["f1"].append(f1)
        macro["precision"].append(prec)
        macro["recall"].append(rec)
        macro["accuracy"].append(acc)

        print(f"[TEST] {name} | F1={f1:.4f} P={prec:.4f} R={rec:.4f} Acc={acc:.4f}")

    if len(test_gml) > 0:
        print("\n[TEST] MACRO | "
              f"F1={np.mean(macro['f1']):.4f} "
              f"P={np.mean(macro['precision']):.4f} "
              f"R={np.mean(macro['recall']):.4f} "
              f"Acc={np.mean(macro['accuracy']):.4f}")


if __name__ == "__main__":
    main()