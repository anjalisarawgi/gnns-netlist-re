import os
import sys
import json
import time
import random
import argparse
import csv
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import yaml
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

### all the model s
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        head_dim = 512 // 8  # 512 // 8 = 64 per head dimensions 

        self.convs.append(GATConv(in_channels, head_dim, heads=8, concat=True, dropout=0.1)) # layer 0 
        for _ in range(5 - 2):
            self.convs.append(GATConv(hidden_dim, head_dim, heads=8, concat=True,dropout=0.1)) # layer 1 and ahead
        self.convs.append(GATConv(hidden_dim, out_channels, heads=1, concat=False, dropout=0.1)) # last layer

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs   = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, 512))
        for _ in range(5 - 2):
            self.convs.append(SAGEConv(512, 512))
        self.convs.append(SAGEConv(512, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        return self.convs[-1](x, edge_index)


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.convs   = torch.nn.ModuleList()
#         self.convs.append(GCNConv(in_channels, 512))
#         for _ in range(5 - 2):
#             self.convs.append(GCNConv(512, 512))
#         self.convs.append(GCNConv(512, out_channels))

#     def forward(self, x, edge_index):
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=0.1, training=self.training)
#         return self.convs[-1](x, edge_index)




def normalize_features(arr: np.ndarray) -> torch.Tensor:
    scaler = StandardScaler()
    arr = scaler.fit_transform(arr)
    return torch.tensor(arr, dtype=torch.float32)

def load_gml(gml_path, mode, positive_class, label_attr, max_features, name2id):
    print(f"[LOAD] {gml_path}")
    with open(gml_path, "r", encoding="utf-8") as f:
        G = nx.parse_gml(f.read(), label="id")
    nodes = list(G.nodes())
    N = len(nodes)

    # features
    raw_feats = [G.nodes[n].get("features", []) for n in nodes]
    feat_arr = np.array(raw_feats, dtype=np.float32)[:, :max_features]
    features = torch.tensor(StandardScaler().fit_transform(feat_arr), dtype=torch.float32)

    # labels
    if mode == "binary":
        labels = torch.tensor([1 if G.nodes[n].get(label_attr, "") == positive_class else 0 for n in nodes], dtype=torch.long)
        id2name = {0: f"not_{positive_class}", 1: positive_class}
    else:
        raw_labels = [G.nodes[n].get(label_attr, "unknown") for n in nodes]
        if name2id is None:
            name2id = {name: i for i, name in enumerate(sorted(set(raw_labels)))}
        labels = torch.tensor([name2id.get(l, -1) for l in raw_labels], dtype=torch.long)
        id2name = {i: name for name, i in name2id.items()}

    # edges (undirected)
    node_map = {n: i for i, n in enumerate(nodes)}
    edges = [(node_map[s], node_map[d]) for s, d in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # splits 80/10/10
    idxs = random.sample(range(N), N)
    t1, t2 = int(0.8 * N), int(0.9 * N)
    train_mask, val_mask, test_mask = [torch.zeros(N, dtype=torch.bool) for _ in range(3)]
    train_mask[idxs[:t1]] = True
    val_mask[idxs[t1:t2]] = True
    test_mask[idxs[t2:]]  = True
	
	# data object for each graph
    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    #print(f"  nodes={N} | edges={edge_index.shape[1]} | features={features.shape[1]}")
    #for u, c in zip(*torch.unique(labels, return_counts=True)):
        #print(f"  class {u.item()} ({id2name.get(u.item(), '?')}): {c.item()} nodes")

    return data, id2name


def merge_graphs(graphs):
    offset, xs, eis, ys, tr, va, te = 0, [], [], [], [], [], []
    for g in graphs:
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        ys.append(g.y)
        tr.append(g.train_mask)
        va.append(g.val_mask)
        te.append(g.test_mask)
        offset += g.num_nodes
    return Data(x=torch.cat(xs), edge_index=torch.cat(eis, dim=1), y=torch.cat(ys), train_mask=torch.cat(tr), val_mask=torch.cat(va), test_mask=torch.cat(te)) # for the combined graph
                

#3###
def train_epoch(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index)
        mask = batch.train_mask & (batch.y != -1)
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(out[mask], batch.y[mask], weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def evaluate_full(model, data, mask) -> dict:
    model.eval()
    out   = model(data.x, data.edge_index)
    pred  = out.argmax(dim=1)
    valid = mask & (data.y != -1)
    y_true = data.y[valid].cpu().numpy()
    y_pred = pred[valid].cpu().numpy()

    return {
        "acc"       : float((y_true == y_pred).mean()),
        # Paper reports both Micro-F1 and Macro-F1 (Sec. IV-B, Fig. 13)
        # "micro_f1"  : float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
        # "macro_f1"  : float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        # "precision" : float(precision_score(y_true, y_pred, average=avg_bi, zero_division=0)),
        # "recall"    : float(recall_score(y_true, y_pred,    average=avg_bi, zero_division=0)),
    }




@torch.no_grad()
def classwise_metrics(model, data, mask, id2name):
    model.eval()
    out = model(data.x, data.edge_index) # run 
    pred   = out.argmax(dim=1) # the predictions 
    valid = mask & (data.y != -1) # ignore all -1 and only look at label nodes -1 in this is case is the default value for non labeled nodes 
    y_true = data.y[valid].cpu().numpy()
    y_pred = pred[valid].cpu().numpy()

    probs = F.softmax(out[valid], dim=1).cpu().numpy() # softmax
    all_classes = sorted(np.unique(y_true).tolist())
    n_cls = len(all_classes)

    # One-vs-rest binarised labels — needed for average_precision_score
    y_bin = label_binarize(y_true, classes=list(range(probs.shape[1])))

    result = {}
    for c in all_classes:
        name = id2name.get(int(c), f"class_{c}")
        
        if n_cls >= 2 and probs.shape[1] > c: # this is simply a safety check to make sure the eval / test cases have atleast 2 classes to evaluate the pr-auc
            pr_auc = float(average_precision_score(y_bin[:, c], probs[:, c]))
                
        result[name] = {
            "acc"    : float((y_pred[y_true == c] == c).mean()),
            "f1"     : float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)),
            "pr_auc" : pr_auc,
        }
    return result


def print_metrics(metrics, classwise):
    print("acc", metrics['acc']:.4f)
        #f"micro_f1={metrics['micro_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        #f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}"
    if classwise:
        for cls, m in classwise.items(): # cls = class name eg sbox of the ditionary and m is one of the three prauc look at the function above 
			print("acc", m['acc']:.4f,  "f1"=m['f1']:.4f, "pr_auc", m['pr_auc']:.4f)

# main training loop 
def run(args):
    set_seed(42)

    # finding all the class names across the files (only for multiclass calssification)
    all_paths = args.train_gmls + [args.val_gml, args.test_gml]
    all_names = set()
    if args.mode == "multiclass":
        for path in all_paths:
            # G = nx.read_gml(path, label="id")
            with open(path, "r", encoding="utf-8") as f:
                G = nx.parse_gml(f.read(), label="id")
            for node in G.nodes():
                all_names.add(G.nodes[node].get(args.label_attr, "unknown"))
        name2id = {name: i for i, name in enumerate(sorted(all_names))}
        id2name = {i: name for name, i in name2id.items()}
        print("[CLASSES count]", len(name2id), "[CLASSES types]: "sorted(name2id)")
    else:
        name2id = None  # binary mode doesn't need this

	# trainsetup where we load and perge the graphs
    train_graphs = []
    for path in args.train_gmls:
        g, id2name = load_gml(path, mode=args.mode, positive_class=args.positive_class, label_attr=args.label_attr, max_features=args.max_features, name2id=name2id)
        train_graphs.append(g)

    train_data = merge_graphs(train_graphs) if len(train_graphs) > 1 else train_graphs[0]
    train_data.train_mask[:] = True
    train_data.val_mask[:]   = False
    train_data.test_mask[:]  = False

	# val data setup remains as it is just masking
    val_data, _ = load_gml(args.val_gml, mode=args.mode, positive_class=args.positive_class, label_attr=args.label_attr, max_features=args.max_features, name2id=name2id)
    val_data.train_mask[:] = False
    val_data.val_mask[:]   = True
    val_data.test_mask[:]  = False
	
	
	# test data setup same like val 
    test_data, _ = load_gml( args.test_gml, mode=args.mode, positive_class=args.positive_class, label_attr=args.label_attr, max_features=args.max_features, name2id=name2id)
    test_data.train_mask[:] = False
    test_data.val_mask[:]   = False
    test_data.test_mask[:]  = True

    in_dim  = train_data.num_features
    out_dim = 2 if args.mode == "binary" else len(id2name)

	# setting up th emodel trianing wegiths for loss ce 
    train_labels = train_data.y[train_data.train_mask & (train_data.y != -1)].cpu().numpy() # get the trianing labels  (excluding -1 which means unknown)
    present_classes = np.unique(train_labels)
    present_weights = compute_class_weight("balanced", classes=present_classes, y=train_labels) # weights based on fequency 

    weights = np.ones(out_dim, dtype=np.float32) # build a weight tensor by starting with 1.0 for all classes
    weights[present_classes] = present_weights # then we update based on the classes
    class_weights = torch.tensor(weights, dtype=torch.float32) # if the class is not seen during training - we set the weight for the class to 1

    # graphsaint
    loader = GraphSAINTRandomWalkSampler( data, batch_size= 3000, walk_length= 2, sample_coverage = 50, shuffle= True)

    # models and params
     if model_name == "gat":
        return GAT(in_dim, out_dim)
    elif model_name == "graphsage":
        return GraphSAGE(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown model:", model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_macro_f1 = -1.0
    best_state        = None

    # main trianing loop 
    for epoch in range(1, args.epochs + 1):

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs: # 5 because we evaluate every 5 epochs
            train_m     = evaluate_full(model, train_data, train_data.train_mask)
        
            ####
            val_m = evaluate_full(model, val_data,   val_data.val_mask)   
            with torch.no_grad():
                val_pred_counts = Counter(model(val_data.x, val_data.edge_index).argmax(dim=1).cpu().tolist())
            print("[VAL PRED DIST]", {id2name.get(k, str(k)): v for k, v in val_pred_counts.most_common()})
            
#            #####
#            test_m_live = evaluate_full(model, test_data,  test_data.test_mask)
#            with torch.no_grad():
#                test_pred_counts = Counter(model(test_data.x, test_data.edge_index).argmax(dim=1).cpu().tolist())
#            print("[TEST PRED DIST]", {id2name.get(k, str(k)): v for k, v in test_pred_counts.most_common()})
#
            print(f"Epoch {epoch:04d}  loss={loss:.4f}  time={dt:.1f}s")
            print("[TRAIN]")
            print_metrics(train_m, classwise_metrics(model, train_data, train_data.train_mask, id2name))
            print("[VAL]")
            print_metrics(val_m, classwise_metrics(model, val_data, val_data.val_mask, id2name))
            #print_metrics("TEST", test_m_live,
            #    classwise_metrics(model, test_data, test_data.test_mask, id2name))

            # checkpoint on best validation Macro-F1
            if val_m["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_m["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"the New best val Macro-F1={best_val_macro_f1:.4f}")

	
	
    print("[TEST]")
    test_m = evaluate_full(model, test_data, test_data.test_mask)
    print_metrics(test_m, classwise_metrics(model, test_data, test_data.test_mask, id2name))
    save_outputs(args, model, test_data, id2name, test_m, circuit_label)

    return model, id2name


# saving
@torch.no_grad()
def save_predictions_gml(gml_path, data, model, id2name, out_path):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)

    # G = nx.read_gml(gml_path, label="id")
    with open(gml_path, "r", encoding="utf-8") as f:
        G = nx.parse_gml(f.read(), label="id")
    for idx, node in enumerate(G.nodes()):
        true_id = data.y[idx].item()
        pred_id = pred[idx].item()
        G.nodes[node]["true_label"]      = id2name.get(true_id, str(true_id))
        G.nodes[node]["predicted_label"] = id2name.get(pred_id, str(pred_id))
        G.nodes[node]["correct"]         = int(true_id == pred_id)

    nx.write_gml(G, out_path)
    print(f"[GML] Predictions saved → {out_path}")


def save_outputs(args, model, test_data, id2name, test_m, circuit_label):
    out_dir = os.path.join("results", f"{args.model}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    # GML predictions on test graph
    if args.test_gml:
        gml_out = os.path.join(out_dir, "predictions.gml")
        save_predictions_gml(args.test_gml, test_data, model, id2name, gml_out)


def parse_args():
    p = argparse.ArgumentParser(description="GNN-RE (Alrahis et al., TCAD 2021)")
    p.add_argument("--config", default=None)
    p.add_argument("--train_gmls",nargs="+", default=None)
    p.add_argument("--val_gml",default=None)
    p.add_argument("--test_gml", default=None)
    p.add_argument("--mode", choices=["binary", "multiclass"], default="multiclass")
    p.add_argument("--positive_class", default="sbox")
    p.add_argument("--label_attr",default="subcircuit_name")
    p.add_argument("--max_features", type=int, default=33) # features also made based on the gnn-re paper 
    p.add_argument("--model", choices=["gat", "graphsage"], default="gat")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=50)
    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            setattr(args, key, value)
    ## debug
    missing = [k for k in ("train_gmls", "val_gml", "test_gml") if not getattr(args, k, None)]
    if missing:
        p.error(f"Missing required fields: {missing}")

    return args
    

if __name__ == "__main__":
    args = parse_args()
    run(args)

