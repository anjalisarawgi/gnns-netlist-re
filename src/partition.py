import torch
import os
from torch_geometric.loader import GraphSAINTSampler, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler,GraphSAINTEdgeSampler
from main import save_predictions_to_gml
from utils.set_seed import set_seed
import wandb
import random
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
from gnn.graphTransformer import GraphTransformer 
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader
import argparse
import time
import json
from torch_geometric.loader import NeighborLoader
import yaml
from sklearn.preprocessing import StandardScaler
import csv
from functools import reduce


import sys
import os
from datetime import datetime

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, 'isatty', lambda: False)() for f in self.files)

# Create logs directory
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"logs/run_{timestamp}.log", "w")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)


# could increase to 24 -- 
torch.set_num_threads(20)        # for math mult (pytorch)    
torch.set_num_interop_threads(2)     # pytorch - helper threads
os.environ["OMP_NUM_THREADS"] = "20" # max 20 cores (pytorch)
os.environ["MKL_NUM_THREADS"] = "20" # max 20 cores (intel math libr)
os.environ["NUMEXPR_NUM_THREADS"] = "20"    # 20 threads max
# --- optimization - 

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_method", type=str, choices=["graphsaint","graphsaint_rw", "graphsaint_node", "graphsaint_edge", "khop"], default="graphsaint",
                    help="Sampling method: 'graphsaint' or 'khop'")
parser.add_argument("--model", default="gat", choices=["graphsage", "gat", "gcn", "graphTransformer"])
parser.add_argument("--train_gml", type = str,  help="which graph (gml_path) do you want to train on?", nargs="+")
parser.add_argument("--val_gml", type = str, help="which graph (gml_path) do you want to evluate (validation) on?", nargs="+")
parser.add_argument("--test_gml", type = str, help="which graph (gml_path) do you want to test on?", nargs="+")
parser.add_argument("--epochs", type = int, default=250)
parser.add_argument("--lr", type = float, default=0.01, help = "learaning rate for main trianing")
# parser.add_argument("--label_mode", type=str, choices = ["subcircuit_name", "boundary"], default="subcircuit_name", help="for sbox and key expand, please use subcircuit")
# graphsaint
parser.add_argument("--sample_coverage", type=int, default=50, help="how many times a node can be seen (sampled as a subgraph/node) for each epoch?") # for others
parser.add_argument("--walk_length", type=int, default=5, help="what is the walk length you want to set for graphsaint sampling method") # for random walk sampling only
parser.add_argument("--num_steps", type=int, default=5, help="how many iterations per epoch do you want?") 
parser.add_argument("--perc_batchsize", type=float, default = 0.01, help="this is for the size of the batch size")

# khop
parser.add_argument("--radius", type=int, default=3, help="what is the radius you want to set for khop sampling method")
parser.add_argument("--num_subgraphs", type=int, default=500, help="what is the number of subgraphs you want to set for khop sampling method")

# # khop v2  - neighbourLoader ( + radius)
# parser.add_argument("--batch_size", type=int, default=2048, help="for NeighborLoader")
# parser.add_argument("--neighbors_per_hop", type=int, default=128, help="for NeighborLoader")
# ml args 
parser.add_argument("--set_gradient_clipping", action="store_true", help="do you want to enable gradient clipping (for potentially stable training)?")
# parser.add_argument("--decision_threshold", type=float, default=0.5, help="Probability threshold for boundary=1 at evaluation time")
# parser.add_argument("--normalize_class_weights", action="store_true", help="kinda confused - but to stabalize training? (i think its just like scaling the weights to avoid exploding gradients)")
parser.add_argument("--reduction_method_cel", type = str, choices=["sum", "mean"])
# cofnig 
parser.add_argument("--config", type=str, help="Path to YAML config file")
args = parser.parse_args()



# yaml 
config_tag = None
if args.config:
    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        setattr(args, key, value)

# wandb setup 
# test_name = os.path.splitext(os.path.basename(args.test_gml))[0]
test_roots = [os.path.splitext(os.path.basename(p))[0] for p in args.test_gml]
test_name = "+".join(test_roots[:2]) + ("+more" if len(test_roots) > 2 else "")

train_roots = [os.path.splitext(os.path.basename(p))[0] for p in args.train_gml]
train_name = "+".join(train_roots[:2]) + ("+test" if len(train_roots) > 2 else "")

if args.sampling_method == "graphsaint_rw":
    sampling_suffix = f"graphsaint_walk{args.walk_length}"
elif args.sampling_method == "khop":
    sampling_suffix = f"khop_r{args.radius}_n{args.num_subgraphs}"
else:
    sampling_suffix = args.sampling_method

run_name = f"{args.perc_batchsize}perc_{config_tag}_{args.model}_{sampling_suffix}_{args.epochs}ep_for_{test_name}"


wandb.init(project="gnn-parition-detection", name=run_name)
wandb.config.update(vars(args))


# # setting label names 
# if args.label_mode == "subcircuit_name":
#     pos_label = "is_sbox"
#     neg_label = "is_not_sbox"
# elif args.label_mode == "boundary":
#     pos_label = "is_boundary"
#     neg_label = "is_not_boundary"

# set seed
set_seed(42)


############################################################

################
# all functions 
################
# possible probelms 

# each graph is getting its own scaler - dangerus
# normalize (mean = 0, sd = 1)
def normalize_features(features):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_tensor = torch.tensor(features, dtype=torch.float)
    return features_tensor

    
#### merge_data
def merge_data(gml_1, gml_2):
    # note here we make offsets so we dont have overlapping edge indexes 
    offset = gml_1.num_nodes
    gml_2_edgeIndex = gml_2.edge_index + offset

    # concat 
    x = torch.cat([gml_1.x, gml_2.x], dim=0)
    edge_index = torch.cat([gml_1.edge_index, gml_2_edgeIndex], dim=1)
    y = torch.cat([gml_1.y, gml_2.y], dim=0)

    train_mask = torch.cat([gml_1.train_mask, gml_2.train_mask], dim = 0)
    val_mask = torch.cat([gml_1.val_mask, gml_2.val_mask], dim = 0)
    test_mask = torch.cat([gml_1.test_mask, gml_2.test_mask], dim = 0)

    # data obj
    merged_data = Data(
        x = x, 
        edge_index=edge_index,
        y = y, 
        train_mask = train_mask, 
        val_mask = val_mask, 
        test_mask = test_mask
    )

    return merged_data

## this function takes a .gml graph --> changes to PyTorch Geometric Dataset
## note:
# a) x = node features (matrix)
# b) y = node labels 
# c) edge_index = edges (2xE tensor)
def load_single_gml(gml_path, remove_edges = False):
    print("[INFO] Calling gml from path:", gml_path)
    
    G = nx.read_gml(gml_path, label = "id") 
    nodes = list(G.nodes()) # list of node ids

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node] # attr?
        feat = attr.get("features", [])

        if not isinstance(feat, (list, tuple, np.ndarray)):
            raise ValueError(f"Node {node} has invalid features")
        features.append(feat[:-1])
        # features.append(feat)
    
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

    # splits --- train / test / val s
    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    # train_cutoff = int(0.90 * num_nodes) # ***
    # val_cutoff = train_cutoff + int(0.05 * num_nodes)
    
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
    data = Data(
        x = features,
        edge_index = edge_index,
        y = labels,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask
    )
    return data, id2label

def focal_loss(logits, targets, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce)
    

def train(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss = 0 
    batch_count = 0 
    total_nodes = 0 

    epoch_nodes = set() # for coverage and debugging and analysis

    for batch in loader: # here, batch is is not the full graph but the sampled subgraph by graphSAINT
        ###### ?????? - i think this logs the node indexes covered in eahc epoch
        if hasattr(batch, "global_id"):
            epoch_nodes.update(batch.global_id.cpu().tolist())
        elif hasattr(batch, "global_node_id"):
            epoch_nodes.update(batch.global_node_id.cpu().tolist())
  
        ###

        ### note:
        # a) batch = subgraph
        # b) batch.x = node features
        # c) batch.edge_index = edges between those nodes
        # d) batch.y = node labels
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index) # here the out.shape = [Num_nodes_in_batch, num_classes]

        # loss_per_node = F.cross_entropy(out, batch.y, reduction="sum")
        loss_per_node = focal_loss(out, batch.y, gamma = 2.0)

        if hasattr(batch, "node_norm"):
            # print("[INFO] using node_norm for loss calculation")
            loss = (loss_per_node * batch.node_norm).sum()
        else:
            loss = loss_per_node.mean()

        # valid_mask = batch.train_mask # disable this later
        # if class_weights is not None:
        #     # loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight = class_weights, reduction = args.reduction_method_cel)
        #     loss = F.cross_entropy(out, batch.y, weight=class_weights)
        # else:
        #     # loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])
        #     loss = F.cross_entropy(out, batch.y)
        
        loss.backward()
        
        if args.set_gradient_clipping: 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print("[INFO] Using gradient clipping")

        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
        # total_nodes += valid_mask.sum().item()
        total_nodes += batch.num_nodes


    if args.reduction_method_cel == "mean": ### ??? not sure if this is the right way to calculate the average loss
        average_loss = total_loss / batch_count if batch_count > 0 else 0  # average for batch
    elif  args.reduction_method_cel == "sum":
        average_loss = total_loss / total_nodes if total_nodes > 0 else 0  # average for nodes

    # average_loss = total_loss

    return average_loss, epoch_nodes




@torch.no_grad()
def evaluate_train_acc(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    # pred = predict_with_threshold(out, args.decision_threshold)

    valid_mask = mask 
    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()

    accuracy = correct / valid_mask.sum().item() 
    return accuracy


@torch.no_grad()
def evaluate_train_fpr(data, model, mask):
    model.eval()
    out = model(data.x, data.edge_index)

    # another moving part: ???
    pred = out.argmax(dim=1) 
    # probs = torch.softmax(out, dim=1) # not so agressive (1) 
    # pred = (probs[:, 1] > 0.9).long() # not so agressive (2) 
    # pred = predict_with_threshold(out, args.decision_threshold)


    valid_mask = mask 

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall



@torch.no_grad()
def eval_class_acc(data, model, mask, id2name=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    # pred = predict_with_threshold(out, args.decision_threshold)


    valid_mask = mask 
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



# test file acc only
@torch.no_grad()
def evaluate_test(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    
    # another moving part: ???
    pred = out.argmax(dim=1) 
    # probs = torch.softmax(out, dim=1) # not so agressive (1) 
    # pred = (probs[:, 1] > 0.9).long() # not so agressive (2) 
    # pred = predict_with_threshold(out, args.decision_threshold)

    
    # valid_mask = (data.y != -1)
    y_true = data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()
    # y_true = data.y.cpu().numpy() ???
    # y_pred = pred.cpu().numpy() ???

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    

    total_acc = (y_true == y_pred).sum() / len(y_true)  # calculating acc
    positive_mask = (y_true == 1)
    negative_mask = (y_true == 0)
    positive_acc = (y_pred[positive_mask] == y_true[positive_mask]).sum() / positive_mask.sum() if positive_mask.sum() > 0.0 else 0.0
    negative_acc = (y_pred[negative_mask] == y_true[negative_mask]).sum() / negative_mask.sum() if negative_mask.sum() > 0.0 else 0.0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "total_acc": total_acc,
        "boundary_1_acc": positive_acc,
        "boundary_0_acc": negative_acc
    }


def predict_with_threshold(out, threshold):
    probs = torch.softmax(out, dim=1)
    return (probs[:, 1] >= threshold).long()

@torch.no_grad()
def evaluate_loss(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    # loss = F.cross_entropy(out, data.y, reduction="mean")
    loss = focal_loss(out, data.y, gamma=2.0).mean()
    return loss.item()
    
##### training  and eval functions:

def run_training(train_data, train_loader, in_dim, out_dim, id2name=None, model_name = "gat", use_weighted_loss = False, val_graphs=None, test_graphs=None):
    class_weights = None

    # setting the model
    if model_name == "graphsage":
        model = graphSAGE(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using graphsage model")
    elif model_name == "gat":
        model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim) 
        # model = torch.compile(model) # ???
        print("[INFO] using gat model")
    elif model_name == "gcn":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using GCN model")
    elif model_name == "graphTransformer":
        model = GraphTransformer(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using GraphTransformer model")


    ##### training parameters 
    # base_lr = 0.01 
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


    # warmup_epochs = 50 
    # warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda = lambda epoch: min((epoch+1)/ warmup_epochs, 1.0)
    # )
    # warmup_scheduler.step()


    ### weighted loss 
    if use_weighted_loss:
        print("[INFO] Using weighted losses")
        # train_labels = train_data.y[train_data.train_mask].cpu().numpy()
        train_labels = train_data.y.cpu().numpy()
        classes = np.unique(train_labels) # can be ignored ???

        weights= compute_class_weight('balanced', classes = classes, y = train_labels)
        print("Class weights:", weights)

        # normlaizing class weights: ??? can be disabled
        print("[INFO] Normalizing class weights (weighted loss)")
        weights = weights / np.mean(weights)

        # final torch classes format 
        class_weights = torch.tensor(weights, dtype = torch.float)


    ##### main training loop now 
    for epoch in range (1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()
        loss, epoch_nodes = train(model, train_loader, optimizer, class_weights)
        train_time = time.perf_counter() - train_start

        # eval_start = time.perf_counter()
        # eval_time = time.perf_counter() - eval_start
        
        epoch_time = time.perf_counter() - epoch_start
        # print("  Test Class-wise Accuracy:")
        # for cls, acc in classwise_acc.items():
        #     print(f"    Class {cls}: {acc:.4f}")

        # class_acc_str = " | ".join(
        #     [f"{cls}:{acc:.3f}" for cls, acc in classwise_acc.items()]
        # )


        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
            f"time: train={train_time:.2f}s, total={epoch_time:.2f}s"
        )

        log_dict = {
            "epoch": epoch,
            "train/loss": loss,
            "time/train_sec": train_time,
            "time/epoch_sec": epoch_time,
        }
        wandb.log(log_dict)

        if epoch % 100 == 0 :
            # ###########
            # ## train side of eval
            # ###########
            # f1, precision, recall = evaluate_train_fpr(train_data, model, train_data.val_mask)
            # train_acc = evaluate_train_acc(model, train_data, train_data.train_mask)
            # val_acc   = evaluate_train_acc(model, train_dfata, train_data.val_mask)
            # classwise_acc = eval_class_acc( train_data, model, train_data.val_mask, id2name)
            # class_acc_str = " | ".join(
            #     [f"{cls}:{acc:.3f}" for cls, acc in classwise_acc.items()]
            # )

            # print(
            #     f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
            #     f"TrainAcc_trainset: {train_acc:.4f}, ValAcc_trainset: {val_acc:.4f}, "
            #     f"F1_trainset: {f1:.4f}, P_trainset: {precision:.4f}, R_trainset: {recall:.4f}, "
            #     f"ClassAcc [{class_acc_str}],"
            # )

            # wandb_log = {
            #     "epoch": epoch,
            #     "train_evaluate/train_acc": train_acc,
            #     "train_evaluate/val_acc": val_acc,
            #     "train_evaluate/f1": f1,
            #     "train_evaluate/precision": precision,
            #     "train_evaluate/recall": recall,
            # }

            # classwise metrics under the same namespace
            # for cls, acc in classwise_acc.items():
            #     wandb_log[f"train_evaluate/class_acc/{cls}"] = acc

            # wandb.log(wandb_log)

            # ---- VAL GRAPHS ----
            eval_start = time.perf_counter()

            val_metrics = defaultdict(list)  # collects lists of per-graph metrics
            val_losses = []
            for path, g in val_graphs:
                m = evaluate_test(g, model)
                val_loss = evaluate_loss(g, model)
                name = os.path.splitext(os.path.basename(path))[0]

                print(
                    f"[VAL][Epoch {epoch:03d}] {name} | "
                    f"Val Loss={val_loss:.4f}, "
                    f"F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, "
                    f"Acc={m['total_acc']:.4f}, b1={m['boundary_1_acc']:.4f}, b0={m['boundary_0_acc']:.4f}"
                )

                # store for macro avg
                for k, v in m.items():
                    val_metrics[k].append(float(v))

                # per-graph wandb
                wandb.log({
                    "epoch": epoch,
                    f"val_pgraph/{name}/f1": m["f1"],
                    f"val_pgraph/{name}/loss": val_loss,
                    f"val_pgraph/{name}/precision": m["precision"],
                    f"val_pgraph/{name}/recall": m["recall"],
                    f"val_pgraph/{name}/accuracy": m["total_acc"],
                    f"val_pgraph/{name}/boundary_acc": m["boundary_1_acc"],
                    f"val_pgraph/{name}/not_boundary_acc": m["boundary_0_acc"],
                })

                val_losses.append(val_loss)

            # macro avg over val designs
            val_macro = {k: float(np.mean(v)) for k, v in val_metrics.items()} if len(val_graphs) else {}
            print(
                f"[VAL][Epoch {epoch:03d}] MACRO | "
                f"Val Macro Loss={float(np.mean(val_losses))}, "
                f"F1={val_macro.get('f1', 0.0):.4f}, "
                f"P={val_macro.get('precision', 0.0):.4f}, "
                f"R={val_macro.get('recall', 0.0):.4f}, "
                f"Acc={val_macro.get('total_acc', 0.0):.4f}, "
                f"b1={val_macro.get('boundary_1_acc', 0.0):.4f}, "
                f"b0={val_macro.get('boundary_0_acc', 0.0):.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "val_macro/f1": val_macro.get("f1", 0.0),
                "val_macro/loss": float(np.mean(val_losses)),
                "val_macro/precision": val_macro.get("precision", 0.0),
                "val_macro/recall": val_macro.get("recall", 0.0),
                "val_macro/accuracy": val_macro.get("total_acc", 0.0),
                "val_macro/boundary_acc": val_macro.get("boundary_1_acc", 0.0),
                "val_macro/not_boundary_acc": val_macro.get("boundary_0_acc", 0.0),
                "time/val_eval_sec": time.perf_counter() - eval_start,
            })


    ### Save model
    run_dir =  os.path.join("models", wandb.run.name)
    os.makedirs(run_dir, exist_ok=True)
    model_path = f"{run_dir}/model.pt"
    torch.save(model.state_dict(), model_path)
    print("[INFO] Saved model to:", model_path)

    # saving some meta data for logging
    metadata = {
        "model_file": model_path,
        "run_name": wandb.run.name,
        "num_training_graphs": len(args.train_gml),
        "training_graphs": args.train_gml,
        "test_graph": args.test_gml,
        "model_type": args.model,
        "sampling_method": args.sampling_method,
        "epochs": args.epochs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = f"{run_dir}/metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("[INFO] Saved metadata to:", json_path)

    return model







        






########
# main #
########
if __name__ == "__main__":
    print("[INFO] Loading training graphs:")
    train_graphs = []
    for i, gml_path in enumerate(args.train_gml):
        print(f"[{i+1}] {gml_path}")
        graph_data, label_map = load_single_gml(gml_path = gml_path, remove_edges=False)

        print("Label counts:", Counter(graph_data.y.tolist())) # debug for -1  label
        assert (graph_data.y < 0).sum().item() == 0, "Still have negative labels!"

        if i ==0: ###???
            id2label = label_map

        train_graphs.append(graph_data)
    
    # debug statements 
    print("----- [DEBUG] -----")
    print(graph_data)
    print("x:", graph_data.x.shape, graph_data.x.dtype)
    print("y:", graph_data.y.shape, graph_data.y.dtype)
    print("edge_index:", graph_data.edge_index.shape, graph_data.edge_index.dtype)
    print("id2label:", label_map)

    #### *** check feature matrix and problem with the length idk 

    ### test gml 
    # testgml_data, _ = load_single_gml(gml_path = args.test_gml, remove_edges=True)
    # testgml_data.x= normalize_features(testgml_data.x.cpu().numpy()) # normalize

    print("[INFO] Loading validation graphs:")
    val_graphs = []
    for i, gml_path in enumerate(args.val_gml):
        print(f"[VAL {i+1}] {gml_path}")
        g, _ = load_single_gml(gml_path=gml_path, remove_edges=False)

        print("[INFO] val graphs -- Label counts:", Counter(g.y.tolist())) # debug
        assert (g.y < 0).sum().item() == 0

        g.train_mask[:] = False
        g.val_mask[:] = False
        g.test_mask[:] = False
        val_graphs.append((gml_path, g))

    print("[INFO] Loading test graphs:")
    test_graphs = []
    for i, gml_path in enumerate(args.test_gml):
        print(f"[TEST {i+1}] {gml_path}")
        g, _ = load_single_gml(gml_path=gml_path, remove_edges=False)

        print("[INFO] test graphs -- Label counts:", Counter(g.y.tolist())) # debug
        assert (g.y < 0).sum().item() == 0
        g.train_mask[:] = False
        g.val_mask[:] = False
        g.test_mask[:] = False
        test_graphs.append((gml_path, g))



    ### merges / combines -- using reduce 
    combined_data = reduce(merge_data, train_graphs)

    # train has the scaler and its fit here
    scaler = StandardScaler()
    combined_data.x = torch.tensor(
        scaler.fit_transform(combined_data.x.cpu().numpy()),
        dtype=torch.float32
    )

    # we use the train sclaer to transform val and test
    for path, g in val_graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)

    for path, g in test_graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)

    print("Train mean/std:", combined_data.x.mean().item(), combined_data.x.std().item())
    print("Val[0] mean/std:", val_graphs[0][1].x.mean().item(), val_graphs[0][1].x.std().item() if len(val_graphs) else ("NA", "NA"))
    print("Test[0] mean/std:", test_graphs[0][1].x.mean().item(), test_graphs[0][1].x.std().item() if len(test_graphs) else ("NA", "NA"))

    # scaler = StandardScaler()
    # combined_data.x = torch.tensor(scaler.fit_transform(combined_data.x.cpu().numpy()), dtype=torch.float32)
    # testgml_data.x  = torch.tensor(scaler.transform(testgml_data.x.cpu().numpy()), dtype=torch.float32)

    # combined_data.x = normalize_features(combined_data.x.cpu().numpy()) # normalize
    # print("Train mean/std:", combined_data.x.mean().item(), combined_data.x.std().item())
    # print("Test  mean/std:", testgml_data.x.mean().item(), testgml_data.x.std().item())

    combined_data.global_id = torch.arange(combined_data.num_nodes)  # setting global ids now which is permanent 
    print("[INFO] training on:", args.train_gml)
    print("[INFO] Number of features:", combined_data.num_features)
    print("[INFO] Feature matrix shape:", combined_data.x.shape)
    print(f"[INFO] Number of nodes for boundary = 1:", (combined_data.y == 1).sum().item())
    print(f"[INFO] Number of nodes for boundary = 0:", (combined_data.y == 0).sum().item())
    print("[INFO] Total training nodes:", combined_data.num_nodes)
    print("[INFO] (a) Train nodes:", combined_data.train_mask.sum().item())
    print("[INFO] (b) Val nodes  :", combined_data.val_mask.sum().item())
    print("[INFO] (c) Test nodes :", combined_data.test_mask.sum().item())

    ### samplers 
    # (a) graph saint
    if args.sampling_method ==  "graphsaint_rw":
        training_data_loader = GraphSAINTRandomWalkSampler(
            combined_data, 
            batch_size = int((args.perc_batchsize)*combined_data.num_nodes ), 
            walk_length = args.walk_length, 
            sample_coverage = args.sample_coverage
        )
    elif args.sampling_method ==  "graphsaint_edge":
        training_data_loader = GraphSAINTEdgeSampler(
            combined_data, 
            batch_size = int((args.perc_batchsize)*combined_data.num_nodes ), 
            num_steps = args.num_steps, 
            sample_coverage = args.sample_coverage
        )
    elif args.sampling_method ==  "graphsaint_node":
        training_data_loader = GraphSAINTNodeSampler(
            combined_data, 
            batch_size = int((args.perc_batchsize)*combined_data.num_nodes ), 
            num_steps = args.num_steps, 
            sample_coverage = args.sample_coverage
        )
    # elif args.sampling_method ==  "graphsaint":
    #     data_loader = GraphSAINTSampler(
    #         combined_data, 
    #         batch_size = int((args.perc_batchsize)*combined_data.num_nodes ), 
    #         num_steps = args.num_steps, 
    #         sample_coverage = args.sample_coverage
    #     )

    # (b) khop sampler (to do )


    ## calling the model 
    model = run_training(
        train_data=combined_data,
        train_loader=training_data_loader,
        in_dim=combined_data.num_features,
        out_dim=2,
        id2name=id2label,
        model_name=args.model,
        use_weighted_loss=False,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
    )
    # metrics = evaluate_test( testgml_data, model)

    # test gml prints
    # print("[INFO] Test GML results:")
    # print("Total nodes:", testgml_data.num_nodes)
    # print(f"Boundary = 1 nodes (+ve):", (testgml_data.y == 1).sum().item())
    # print(f"Boundary = 0 nodes (-ve):", (testgml_data.y == 0).sum().item())

    # print(f"F1 = {metrics['f1']:.4f}, Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}")
    # print(f"Total Accuracy   : {metrics['total_acc']:.4f}")
    # print(f"Boundary = 1 Accuracy    : {metrics[f'boundary_1_acc']:.4f}")
    # print(f"Boundary = 0 Accuracy: {metrics[f'boundary_0_acc']:.4f}")

    print("[INFO] Final evaluation on TEST graphs:")

    test_metrics = defaultdict(list)

    for path, g in test_graphs:
        n = evaluate_test(g, model)
        name = os.path.splitext(os.path.basename(path))[0]

        print(
            f"[TEST] {name} | "
            f"F1={n['f1']:.4f}, P={n['precision']:.4f}, R={n['recall']:.4f}, "
            f"Acc={n['total_acc']:.4f}, b1={n['boundary_1_acc']:.4f}, b0={n['boundary_0_acc']:.4f}"
        )

        for k, v in n.items():
            test_metrics[k].append(float(v))

        wandb.log({
            f"test/{name}/f1": n["f1"],
            f"test/{name}/precision": n["precision"],
            f"test/{name}/recall": n["recall"],
            f"test/{name}/accuracy": n["total_acc"],
            f"test/{name}/boundary_acc": n["boundary_1_acc"],
            f"test/{name}/not_boundary_acc": n["boundary_0_acc"],
        })

    test_macro = {k: float(np.mean(v)) for k, v in test_metrics.items()} if len(test_graphs) else {}
    print(
        f"[TEST] MACRO | "
        f"F1={test_macro.get('f1', 0.0):.4f}, "
        f"P={test_macro.get('precision', 0.0):.4f}, "
        f"R={test_macro.get('recall', 0.0):.4f}, "
        f"Acc={test_macro.get('total_acc', 0.0):.4f}, "
        f"b1={test_macro.get('boundary_1_acc', 0.0):.4f}, "
        f"b0={test_macro.get('boundary_0_acc', 0.0):.4f}"
    )

    wandb.log({
        "test_macro/f1": test_macro.get("f1", 0.0),
        "test_macro/precision": test_macro.get("precision", 0.0),
        "test_macro/recall": test_macro.get("recall", 0.0),
        "test_macro/accuracy": test_macro.get("total_acc", 0.0),
        "test_macro/boundary_acc": test_macro.get("boundary_1_acc", 0.0),
        "test_macro/not_boundary_acc": test_macro.get("boundary_0_acc", 0.0),
    })

    # saving the results in gml 
    output_dir = "results/aes_to_testml"
    os.makedirs(output_dir, exist_ok=True)

    output_labels = {0: "not_boundary", 1: "boundary"}
    print("[INFO] Output Labels:", output_labels)

    for path, g in test_graphs:
        test_graph_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, f"{test_graph_name}_predictions.gml")
        print("[INFO] Saving predictions to:", output_path)

        save_predictions_to_gml(
            original_gml_path=path,
            data=g,
            model=model,
            id2name=output_labels,
            output_gml_path=output_path
        )