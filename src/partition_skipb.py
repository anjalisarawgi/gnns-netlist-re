import torch
import os
from torch_geometric.loader import GraphSAINTSampler, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler,GraphSAINTEdgeSampler
# from main import save_predictions_to_gml
# from utils.set_seed import set_seed
import wandb
import random
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from gnn.graphSAGE import graphSAGE
from gnn.sage_bidirected import BiDirectedGraphSAGE
from gnn.sage_jk import JK_GraphSAGE
from gnn.sage_jk_bi import BiDirectedJK_GraphSAGE
from gnn.gcn import GCN
from gnn.gat import gat, MLP, gatv2, GAAN
from gnn.gin import GIN
from gnn.bi_and_hi_GAT import DirectedOnlyGAT, HierarchicalOnlyGAT4,  HierarchicalOnlyGAT6, HierarchicalDirectedGAT_v2, DirectedOnlyGATWithGlobal
from gnn.graphTransformer import GraphTransformer 
from gnn.new_gnn import DirectedGAT, HierarchicalGAT, HierarchicalDirectedGAT
from sklearn.utils.class_weight import compute_class_weight
# from gnn.abgnn import AsyncDirectedGAT
# from gnn.daggnn import BIGAT
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
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import sys
import os
from datetime import datetime
import joblib
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



@torch.no_grad()
def save_predictions_to_gml(original_gml_path, data, model, id2name, output_gml_path):
    import networkx as nx
    import torch
    import numpy as np

    model.eval()
    data = data.to(next(model.parameters()).device)

    # --- forward pass ---
    out = model(data.x, data.edge_index)
    probs_all = torch.softmax(out, dim=1)[:, 1].cpu().numpy()  # all nodes, for GML writing
    labels_all = data.y.cpu().numpy()                           # all nodes, includes -1

    # filter to labeled nodes only for all sklearn calls
    labeled_mask = data.label_mask.cpu().numpy() if hasattr(data, 'label_mask') else (labels_all >= 0)
    probs  = probs_all[labeled_mask]
    labels = labels_all[labeled_mask]

    # --- default prediction (labeled only) ---
    pred_default = (probs >= 0.5).astype(int)

    # --- BEST threshold (F1-based, labeled only) ---
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx   = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    pred_best  = (probs >= best_thresh).astype(int)

    # --- RATIO-BASED prediction (top-k, labeled only) ---
    N          = len(probs)
    true_ratio = float((labels == 1).mean())
    k          = max(1, int(np.ceil(true_ratio * N)))
    idx_sorted = np.argsort(-probs)
    topk_idx   = idx_sorted[:k]
    pred_ratio = np.zeros(N, dtype=int)
    pred_ratio[topk_idx] = 1

    # --- TP / FP / FN / TN (labeled only) ---
    def compute_classes(y_true, y_pred):
        classes = []
        for yt, yp in zip(y_true, y_pred):
            if   yt == 1 and yp == 1: classes.append("TP")
            elif yt == 0 and yp == 1: classes.append("FP")
            elif yt == 1 and yp == 0: classes.append("FN")
            else:                     classes.append("TN")
        return classes

    classes_default = compute_classes(labels, pred_default)
    classes_best    = compute_classes(labels, pred_best)
    classes_ratio   = compute_classes(labels, pred_ratio)

    # --- metrics printout ---
    from sklearn.metrics import f1_score, precision_score, recall_score

    def compute_metrics(y_true, y_pred, name):
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p  = precision_score(y_true, y_pred, zero_division=0)
        r  = recall_score(y_true, y_pred, zero_division=0)
        print(f"[{name}] F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        return f1, p, r

    print(f"[INFO] Best threshold: {best_thresh:.4f}")
    print("\n[THRESHOLD COMPARISON]")
    compute_metrics(labels, pred_default, "Default@0.5")
    compute_metrics(labels, pred_best,    f"Best@{best_thresh:.3f}")
    compute_metrics(labels, pred_ratio,   "Top-K (ratio)")

    print("\n[PREDICTED POSITIVE COUNTS]")
    print(f"Default@0.5        → {pred_default.sum()} nodes")
    print(f"Best@{best_thresh:.3f} → {pred_best.sum()} nodes")
    print(f"Top-K (ratio)      → {pred_ratio.sum()} nodes (target={k})")

    # --- expand predictions back to ALL nodes for GML writing ---
    labeled_indices = np.where(labeled_mask)[0]

    pred_default_all   = np.full(len(labels_all), -1, dtype=int)
    pred_best_all      = np.full(len(labels_all), -1, dtype=int)
    pred_ratio_all     = np.full(len(labels_all), -1, dtype=int)
    classes_default_all = ["UNLABELED"] * len(labels_all)
    classes_best_all    = ["UNLABELED"] * len(labels_all)
    classes_ratio_all   = ["UNLABELED"] * len(labels_all)

    for j, i in enumerate(labeled_indices):
        pred_default_all[i]    = pred_default[j]
        pred_best_all[i]       = pred_best[j]
        pred_ratio_all[i]      = pred_ratio[j]
        classes_default_all[i] = classes_default[j]
        classes_best_all[i]    = classes_best[j]
        classes_ratio_all[i]   = classes_ratio[j]

    # --- write to GML ---
    G     = nx.read_gml(original_gml_path)
    nodes = list(G.nodes())

    for i, node in enumerate(nodes):
        G.nodes[node]["prob"]               = float(probs_all[i])
        G.nodes[node]["pred_default"]       = int(pred_default_all[i])
        G.nodes[node]["pred_best"]          = int(pred_best_all[i])
        G.nodes[node]["pred_ratio"]         = int(pred_ratio_all[i])
        G.nodes[node]["pred_class_default"] = classes_default_all[i]
        G.nodes[node]["pred_class_best"]    = classes_best_all[i]
        G.nodes[node]["pred_class_ratio"]   = classes_ratio_all[i]
        G.nodes[node]["is_correct_default"] = int(pred_default_all[i] == labels_all[i]) if labeled_mask[i] else -1
        G.nodes[node]["is_correct_best"]    = int(pred_best_all[i]    == labels_all[i]) if labeled_mask[i] else -1
        G.nodes[node]["is_correct_ratio"]   = int(pred_ratio_all[i]   == labels_all[i]) if labeled_mask[i] else -1

    nx.write_gml(G, output_gml_path)
    print(f"[INFO] Saved GML with predictions → {output_gml_path}")


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
torch.set_num_threads(10)        # for math mult (pytorch)    
torch.set_num_interop_threads(2)     # pytorch - helper threads
os.environ["OMP_NUM_THREADS"] = "10" # max 20 cores (pytorch)
os.environ["MKL_NUM_THREADS"] = "10" # max 20 cores (intel math libr)
os.environ["NUMEXPR_NUM_THREADS"] = "10"    # 20 threads max
# --- optimization - 

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_method", type=str, choices=["graphsaint","graphsaint_rw", "graphsaint_node", "graphsaint_edge", "khop"], default="graphsaint",
                    help="Sampling method: 'graphsaint' or 'khop'")
parser.add_argument("--model", default="gat", choices=["graphsage", "gat", "gcn", "graphTransformer", "gin", "gatv2", "dGNN", 
                    "hGNN", "hdGNN",  "FlatDirectedGAT", "DirectedOnlyGAT", "HierarchicalOnlyGAT4",  "HierarchicalOnlyGAT6", "HierarchicalDirectedGAT_v2", "DirectedOnlyGATWithGlobal",
                     "GAAN", "GraphSAGE_ResNorm", "BiDirectedGraphSAGE","JK_GraphSAGE", "BiDirectedJK_GraphSAGE"])
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
parser.add_argument("--reduction_method_cel", type = str, choices=["sum", "mean"])
parser.add_argument( "--loss_type", type=str, choices=["ce", "ce_weighted", "ce_soft", "focal"], default="focal" )
parser.add_argument(
    "--training_mode",type=str, choices=["fullgraph", "graphsaint"],  default="graphsaint",    help="Train on full graph or sampled subgraphs")
parser.add_argument("--use_scheduler", action="store_true", help="do you want to use the learning rate scheduler?")
# features 
parser.add_argument("--use_partition_features", action="store_true", help="if you want to concatenate partition_features to node features")
parser.add_argument("--use_graph_features", action="store_true", help="if you want to concatenate partition_features to node features")
parser.add_argument(
    "--fullgraph_mode",
    type=str,
    choices=["merged", "per_design"],
    default="merged",
    help="How to train in fullgraph mode"
)
parser.add_argument("--use_unsupervised_features", action="store_true")
parser.add_argument("--use_unsupervised_features_louvian", action="store_true")

## test block
parser.add_argument("--use_lib_id", action="store_true", help="append library one-hot to node features")
parser.add_argument("--use_design_id", action="store_true", help="append design one-hot to node features (leaky if testing unseen designs!)")
##### test block end
parser.add_argument("--use_augmentation",      action="store_true")
parser.add_argument("--aug_gate_flip_frac",    type=float, default=0.10)
parser.add_argument("--aug_edge_corrupt_frac", type=float, default=0.10)
# cofnig 
parser.add_argument("--config", type=str, help="Path to YAML config file")
args = parser.parse_args()


GATE_TYPES = ["INPUT", "OUTPUT", "AND", "OR", "NAND", "NOR", "XOR", "XNOR", "INV", "AOI", "OAI", "MUX", "DFF", "UNKNOWN"]
GATE2ID = {g: i for i, g in enumerate(GATE_TYPES)}
NUM_GATE_TYPES = len(GATE_TYPES)  # 14


# yaml 
config_tag = None
if args.config:
    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():
        setattr(args, key, value)

def get_config_prefix():
    return config_tag if config_tag is not None else "no_config"

def make_result_dir(model_family: str):
    prefix = get_config_prefix()
    out_dir = os.path.join("results", model_family, wandb.run.name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

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

config_prefix = config_tag if config_tag is not None else "no_config"

if args.training_mode == "fullgraph":
    run_name = (
        f"{args.model}_{args.loss_type}_"
        f"fullgraph_{args.fullgraph_mode}_"
        f"{config_prefix}"
    )
else:
    run_name = "gnnsampling"

wandb.init(project="gnn-parition-detection", name=run_name)
wandb.config.update(vars(args))
wandb.config.update({"loss_type": args.loss_type})

# # setting label names 
# if args.label_mode == "subcircuit_name":
#     pos_label = "is_sbox"
#     neg_label = "is_not_sbox"
# elif args.label_mode == "boundary":
#     pos_label = "is_boundary"
#     neg_label = "is_not_boundary"

# set seed
set_seed(42)


##### test block for graph encoding:
from pathlib import Path

def parse_lib_design(gml_path: str):
    p = Path(gml_path)
    design_family = p.parts[-3]
    lib = p.parts[-2]
    return design_family, lib

train_families, train_libs = [], []
for p in args.train_gml:
    fam, lib = parse_lib_design(p)
    train_families.append(fam)
    train_libs.append(lib)

family2id = {f: i for i, f in enumerate(sorted(set(train_families)))}
lib2id    = {l: i for i, l in enumerate(sorted(set(train_libs)))}

print("[DOMAIN] lib2id:", lib2id)
print("[DOMAIN] family2id size:", len(family2id))
##########
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

def pr_auc_from_probs(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_prob))

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

    label_mask = torch.cat([gml_1.label_mask, gml_2.label_mask], dim=0)

    # data obj
    merged_data = Data(
        x = x, 
        edge_index=edge_index,
        y = y, 
        label_mask = label_mask,
        train_mask = train_mask, 
        val_mask = val_mask, 
        test_mask = test_mask
    )

    return merged_data


def train_family_weighted(model, graphs, graph_paths, optimizer, class_weights=None, soft_class_weights=None):
    model.train()
    optimizer.zero_grad()

    # group graph indices by family
    family_to_indices = defaultdict(list)
    for i, path in enumerate(graph_paths):
        fam, _ = parse_lib_design(path)
        family_to_indices[fam].append(i)

    num_families = len(family_to_indices)
    total_loss = 0.0

    for fam, indices in family_to_indices.items():
        num_families = len(family_to_indices)
        
        for i in indices:
            g = graphs[i].to(device)
            out = model(g.x, g.edge_index)
            labeled_mask = g.label_mask if hasattr(g, 'label_mask') else (g.y >= 0)

            if args.loss_type == "focal":
                loss_per_node = focal_loss(out[labeled_mask], g.y[labeled_mask])
            elif args.loss_type == "ce_weighted":
                loss_per_node = F.cross_entropy(out[labeled_mask], g.y[labeled_mask], weight=class_weights.to(out.device), reduction="none")
            elif args.loss_type == "ce_soft":
                loss_per_node = F.cross_entropy(out[labeled_mask], g.y[labeled_mask], weight=soft_class_weights.to(out.device), reduction="none")
            else:
                loss_per_node = F.cross_entropy(out[labeled_mask], g.y[labeled_mask], reduction="none")

            # divide by both len(indices) and num_families so each family contributes equally
            loss = loss_per_node.mean() / (len(indices) * num_families)
            loss.backward()          # ← backward while computation graph still alive
            total_loss += loss.item()

            del out, loss_per_node, loss
            g = g.to("cpu")

    if args.set_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    return total_loss


## this function takes a .gml graph --> changes to PyTorch Geometric Dataset
## note:
# a) x = node features (matrix)
# b) y = node labels 
# c) edge_index = edges (2xE tensor)
def augment_graph_noise(data: Data,
                        n_categorical: int = 14,
                        gate_flip_frac: float = 0.10,
                        edge_corrupt_frac: float = 0.10) -> Data:
    """
    Simulates real-world circuit noise:
      1) Gate mislabelling  — randomly swap one-hot gate type for gate_flip_frac of nodes
      2) Wrong wires        — randomly drop + add edges for edge_corrupt_frac of edges
    Continuous features (indeg, outdeg, ratio) are left alone — in a real
    corrupted netlist these would change too, but recomputing them is expensive.
    """
    data = data.clone()
    N = data.x.size(0)
    E = data.edge_index.size(1)

    # --- 1. gate label corruption (one-hot swap) ---
    n_flip = max(1, int(gate_flip_frac * N))
    flip_idx = torch.randperm(N)[:n_flip]
    # pick a random different gate type for each flipped node
    random_gates = torch.randint(0, n_categorical, (n_flip,))
    new_onehot = torch.zeros(n_flip, n_categorical)
    new_onehot[torch.arange(n_flip), random_gates] = 1.0
    data.x[flip_idx, :n_categorical] = new_onehot

    # --- 2. edge corruption (drop + add random edges) ---
    n_corrupt = max(1, int(edge_corrupt_frac * E))

    # drop n_corrupt random existing edges
    keep_mask = torch.ones(E, dtype=torch.bool)
    drop_idx = torch.randperm(E)[:n_corrupt]
    keep_mask[drop_idx] = False
    kept_edges = data.edge_index[:, keep_mask]

    # add n_corrupt random new edges to replace them
    rand_src = torch.randint(0, N, (n_corrupt,))
    rand_dst = torch.randint(0, N, (n_corrupt,))
    new_edges = torch.stack([rand_src, rand_dst], dim=0)

    data.edge_index = torch.cat([kept_edges, new_edges], dim=1)

    return data
def load_single_gml(gml_path, remove_edges = False):
    print("[INFO] Calling gml from path:", gml_path)
    
    G = nx.read_gml(gml_path) 
    nodes = list(G.nodes()) # list of node ids

    features = []
    labels = []
    label_mask = [] 
    base_feat_dim = None

    
    id2label = {0: "not_boundary", 1: "boundary", -1: "unlabeled"}  # Define early
    nan_boundary_count = 0
    nan_is_io_count = 0 
    for node in nodes:
        attr = G.nodes[node] # attr?
        feat = attr.get("features", [])
        # feat = feat[0:31] #### (14ohe) + (14ohe) + indeg, outdeg, ratio  -- basic dimensions
        # feat = feat[28:31] #### indeg, outdeg, fanin
        # feat = feat[0:28] ####  (14ohe) + (14ohe) 
        # feat = feat[:-2]
        feat = feat[:42]
        if base_feat_dim is None:
            base_feat_dim = len(feat)

        if args.use_partition_features:     
            partition_feat = attr.get("partition_features", [0.0, 0.0, 0.0])
            partition_feat = partition_feat[2:3]
            # partition_feat_twoHop = partition_feat[1]
            feat = list(feat) +list(partition_feat)
            # feat = list(feat) + [float(partition_feat_twoHop)]

        if args.use_graph_features:
            graph_feat = attr.get("graph_features", [0.0, 0.0])
            graph_feat_subset = graph_feat[:2]
            feat = list(feat) +list(graph_feat_subset)

        if args.use_unsupervised_features:
            f1 = float(attr.get("unsup_louvain_1hop", 0.0))
            f2 = float(attr.get("unsup_louvain_2hop", 0.0))
            feat = list(feat) + [f1, f2]

        if args.use_unsupervised_features_louvian:
            f1 = float(attr.get("unsup_leiden_1hop", 0.0))
            f2 = float(attr.get("unsup_leiden_2hop", 0.0))
            feat = list(feat) + [f1, f2]


        if not isinstance(feat, (list, tuple, np.ndarray)):
            raise ValueError(f"Node {node} has invalid features")
        # features.append(feat[:-1])
        # feat = feat[:15] + feat[16:] # skip feature at index 15
        # feat = feat[:-3] # skip last 3 features

        #### text block
        fam, lib = parse_lib_design(gml_path)

        if args.use_lib_id:
            lib_oh = np.zeros(len(lib2id), dtype=np.float32)
            if lib in lib2id:
                lib_oh[lib2id[lib]] = 1.0
            feat = list(feat) + lib_oh.tolist()

        if args.use_design_id:
            # I’d recommend using design_family here, NOT instance, to reduce leakage.
            fam_oh = np.zeros(len(family2id) + 1, dtype=np.float32)  # +1 unknown
            idx = family2id.get(fam, len(family2id))
            fam_oh[idx] = 1.0
            feat = list(feat) + fam_oh.tolist()
        #### text block end

        features.append(feat)

        if "boundary" in attr:
            boundary_value = attr["boundary"]
            try:
                label = int(boundary_value)
            except (ValueError, TypeError):
                label = 0
            labels.append(label)
            label_mask.append(True)
        else:
            labels.append(-1)  # Placeholder
            label_mask.append(False)
            nan_boundary_count += 1
            
            label_copy = str(attr.get("label_copy", "")).strip("'\"").upper()
            if "INPUT" in label_copy or "OUTPUT" in label_copy:
                nan_is_io_count += 1

    print(f"[TOCHECK] {gml_path}: missing boundary: {nan_boundary_count} "
          f"(of which INPUT/OUTPUT: {nan_is_io_count}, other: {nan_boundary_count - nan_is_io_count})")

    labels = torch.tensor(labels, dtype=torch.long)
    label_mask = torch.tensor(label_mask, dtype=torch.bool)

    # Debug: only count labeled nodes
    labeled_labels = labels[label_mask]
    unique_classes, class_counts = np.unique(labeled_labels.cpu().numpy(), return_counts=True)
    for u, c in zip(unique_classes, class_counts):
        print(f"Class {u} ({id2label.get(int(u), '?')}): {c} samples")
    
    print(f"[INFO] Total nodes: {len(nodes)} (labeled: {label_mask.sum().item()}, unlabeled: {(~label_mask).sum().item()})")


    # node map is like a lookup (between NetworkX and pyG) ???
    # and we want edge_index to be of shape for pyg: [2, num_edges]
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 

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
  
    features = torch.as_tensor(np.array(features, dtype=np.float32), dtype=torch.float32)

    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        label_mask=label_mask, 
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print("[FEATURE DIM CHECK]")
    print("  base features dim           :", base_feat_dim)

    if args.use_partition_features:
        print("  after partition features   :", after_partition_dim)

    if args.use_graph_features:
        print("  after graph features       :", after_graph_dim)

    print("  final feature dim (tensor) :", data.x.shape[1])
    return data, id2label

def focal_loss(logits, targets, gamma=2.0, alpha = 0.25): # high alpha, for   more imbalnace, higher gamma = focus on mistakes (0.65 , 0.75, 0.85) (1.0, 2.0, 3.0) so higher gamma downweights easy samples and says it to focus on harder samples
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    at = torch.where(targets ==1, alpha, 1 - alpha)
    return at * ((1 - pt) ** gamma) * ce
    

def train_equal_design_weight(model, graphs, optimizer, class_weights=None, soft_class_weights=None):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    num_graphs = len(graphs)

    for g in graphs:
        g = g.to(device)
        out = model(g.x, g.edge_index)

        labeled_mask = g.label_mask if hasattr(g, 'label_mask') else (g.y >= 0) 
        if args.loss_type == "focal":
            loss = focal_loss(out[labeled_mask], g.y[labeled_mask]).mean() / num_graphs
        elif args.loss_type == "ce_weighted":
            loss = F.cross_entropy(out[labeled_mask], g.y[labeled_mask], weight=class_weights.to(out.device)) / num_graphs
        elif args.loss_type == "ce_soft":
            loss = F.cross_entropy(out[labeled_mask], g.y[labeled_mask], weight=soft_class_weights.to(out.device)) / num_graphs
        else:
            loss = F.cross_entropy(out[labeled_mask], g.y[labeled_mask]) / num_graphs
        
        # divide by num_graphs here so gradients are equivalent to the mean
        # loss = loss_per_node.mean() / num_graphs
        loss.backward()  # frees this graph's computation graph immediately

        total_loss += loss.item()

        # ADD THESE TWO LINES:
        del out, loss
        torch.cuda.empty_cache()
        # optionally move graph back to CPU to free VRAM between designs
        g = g.to("cpu")

    if args.set_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    return float(total_loss)
    
    
def train_fullgraph(model, data, optimizer, class_weights=None, soft_class_weights=None):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x, data.edge_index)

     # Compute loss only on labeled nodes
    labeled_mask = data.label_mask if hasattr(data, 'label_mask') else (data.y >= 0)
    effective_mask = labeled_mask & data.train_mask

    if args.loss_type == "focal":
        loss_per_node = focal_loss(out[effective_mask], data.y[effective_mask])
    elif args.loss_type == "ce_weighted":
        loss_per_node = F.cross_entropy(
            out[effective_mask], data.y[effective_mask],
            weight=class_weights.to(out.device),
            reduction="none"
        )
    elif args.loss_type == "ce_soft":
        loss_per_node = F.cross_entropy(
            out[effective_mask], data.y[effective_mask],
            weight=soft_class_weights.to(out.device),
            reduction="none"
        )
    else:
        loss_per_node = F.cross_entropy(out[effective_mask], data.y[effective_mask], reduction="none")

    # train_mask lets you keep future flexibility
    loss = loss_per_node.mean()

    loss.backward()


    if args.set_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    return loss.item()

def train(model, loader, optimizer, class_weights=None, soft_class_weights=None):
    model.train()
    total_loss = 0 
    batch_count = 0 
    total_nodes = 0 

    epoch_nodes = set() # for coverage and debugging and analysis

    for batch in loader: # here, batch is is not the full graph but the sampled subgraph by graphSAINT
        batch = batch.to(device)
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
        # loss_per_node = focal_loss(out, batch.y, gamma = 2.0)
        labeled_mask = batch.label_mask if hasattr(batch, 'label_mask') else (batch.y >= 0)

        if labeled_mask.sum() == 0:
            continue

        if args.loss_type == "focal":
            loss_per_node = focal_loss(out[labeled_mask], batch.y[labeled_mask], gamma=2.0)
        elif args.loss_type == "ce_weighted":
            loss_per_node = F.cross_entropy(
                out[labeled_mask],
                batch.y[labeled_mask],
                weight=class_weights.to(out.device),
                reduction="none"
            )

        elif args.loss_type == "ce_soft":
            loss_per_node = F.cross_entropy(
                out[labeled_mask],
                batch.y[labeled_mask],
                weight=soft_class_weights.to(out.device),
                reduction="none"
            )
        elif args.loss_type == "ce":
            loss_per_node = F.cross_entropy(
                out[labeled_mask],
                batch.y[labeled_mask],
                reduction="none"
            )
        else:
            raise ValueError(f"Unknown loss_type: {args.loss_type}")

        if hasattr(batch, "node_norm"):
            # print("[INFO] using node_norm for loss calculation")
            loss = (loss_per_node * batch.node_norm[labeled_mask]).sum()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    label_mask = data.label_mask.to(device) if hasattr(data, 'label_mask') else (data.y >= 0)
    valid_mask = mask.to(device) & label_mask
    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()
    accuracy = correct / valid_mask.sum().item() if valid_mask.sum() > 0 else 0.0
    return accuracy


@torch.no_grad()
def evaluate_train_fpr(data, model, mask):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)

    # another moving part: ???
    # pred = out.argmax(dim=1) 
    # probs = torch.softmax(out, dim=1) # not so agressive (1) 
    # pred = (probs[:, 1] > 0.9).long() # not so agressive (2) 
    # pred = predict_with_threshold(out, args.decision_threshold)

    pred = out.argmax(dim=1)


    label_mask = data.label_mask.to(device) if hasattr(data, 'label_mask') else (data.y >= 0)
    valid_mask = mask.to(device) & label_mask

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall



@torch.no_grad()
def eval_class_acc(data, model, mask, id2name=None):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    # pred = out.argmax(dim=1)
    pred = out.argmax(dim=1)

    valid_mask = mask.to(device)
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

from collections import deque

##### [analysis block  ] #####
# creating function to find - nearest boundary nodes: i.e.
# eg how many steps / hops away is the nearest boudnary from a node
# we can consider shortest path to reach = number of hops
def multi_source_bfs(G, sources):
    visited ={} # storing distances
    queue = deque() # the nodes we still need to explore 

    for s in sources:
        visited[s] = 0 
        queue.append(s)

    while queue:
        node = queue.popleft() # if starting from node C , we remvoe c
        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited[nbr] = visited[node] + 1
                queue.append(nbr)
        
    return visited


def compute_region_iou(G, true_boundary_set, pred_boundary_set, k_hop):
        if len(pred_boundary_set) == 0:
            return 0.0

        # distance from all nodes to nearest TRUE boundary
        dist_true = multi_source_bfs(G, list(true_boundary_set))
        true_region = set(n for n, d in dist_true.items() if d <= k_hop)

        # distance from all nodes to nearest PREDICTED boundary
        dist_pred = multi_source_bfs(G, list(pred_boundary_set))
        pred_region = set(n for n, d in dist_pred.items() if d <= k_hop)

        intersection = len(true_region & pred_region)
        union = len(true_region | pred_region)

        if union == 0:
            return 0.0

        return intersection / union


# test file acc only
@torch.no_grad()
def evaluate_test(data, model):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    
    # another moving part: ???
    # pred = out.argmax(dim=1) 
    # probs = torch.softmax(out, dim=1) # not so agressive (1) 
    # pred = (probs[:, 1] > 0.9).long() # not so agressive (2) 
    # pred = predict_with_threshold(out, args.decision_threshold)

    pred = out.argmax(dim=1)

    labeled_mask = data.label_mask if hasattr(data, 'label_mask') else (data.y >= 0)

    # valid_mask = (data.y != -1)
    y_true = data.y[labeled_mask].cpu().numpy()
    y_pred = pred[labeled_mask].cpu().numpy()
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


@torch.no_grad()
def evaluate_region_metrics(data, model, k_percent=2.0, r=2, threshold=None, prob_mass=0.5):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy() # callingsoftmax predictions

    # labels
    labeled_mask = data.label_mask.cpu().numpy() if hasattr(data, 'label_mask') else (data.y.cpu().numpy() >= 0)
    y_true = data.y.cpu().numpy()
    true_boundary = set(np.where((y_true == 1) & labeled_mask)[0].tolist())

    if len(true_boundary) == 0: # gives indices
        return {}

    N = len(probs)

    # considering top highest prob nodes (eg 2% most confident)
    k = max(1, int(np.ceil((k_percent / 100.0) * N)))
    idx_sorted = np.argsort(-probs)
    topk_nodes = idx_sorted[:k]
    topk_set = set(topk_nodes.tolist())

    # building undirectred graph to convert format to network x for multi_source_bfs
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_nodes_from(range(data.num_nodes)) # not sure
    G.add_edges_from(edges)

    # a) Q1 - distance to the nearest boundary
    ## this part does this: for every node in the graph, it searches for the nearest TRUE boundary node  --> and then the mean of this
    dist_to_true = multi_source_bfs(G, list(true_boundary))
    dists_topk = np.array([dist_to_true.get(n, np.inf) for n in topk_nodes])
    mean_dist = float(np.mean(dists_topk[np.isfinite(dists_topk)])) if np.any(np.isfinite(dists_topk)) else float("inf")
    pct_within_2 = float(np.mean(dists_topk <= 2)) # the fraction of the nodes that is <= 2 hops away from the boundayr nodes

    # b) Q2: are our predicted ndoes atleast in the boundary region?
    ## so for eveyr node - we predict the distance to the nearest predicted node
    def boundary_covered(pred_set, k_hop):
        if len(pred_set) == 0:
            return 0.0
        dist_to_pred = multi_source_bfs(G, list(pred_set))
        covered = sum(1 for b in true_boundary if dist_to_pred.get(b, np.inf) <= k_hop) # count it if its within the hops
        return covered / len(true_boundary)

    boundary_coverage_topk_2hop = boundary_covered(topk_set, 2)
    boundary_coverage_topk_1hop = boundary_covered(topk_set, 1)


    # in this part i want to try for the normal threshold we are using for our trianing now: i.e. argmax 0.5
    threshold = 0.5

    thresh_set = set(np.where(probs >= threshold)[0].tolist())
    boundary_coverage_thresh_2 = boundary_covered(thresh_set, 2) # q2 - 2 hops
    boundary_coverage_thresh_1 = boundary_covered(thresh_set, 1) # q2 - 1 hop
    boundary_coverage_thresh_0 = boundary_covered(thresh_set, 0)

    # iou 
    region_iou_k1 = compute_region_iou(G, true_boundary, thresh_set, k_hop=1)
    region_iou_k2 = compute_region_iou(G, true_boundary, thresh_set, k_hop=2)


    #a) Q1 - distance to the nearest boundary
    # mean dist and p
    if len(thresh_set) > 0:
        dists_thresh = np.array([dist_to_true.get(n, np.inf) for n in thresh_set])

        finite_mask = np.isfinite(dists_thresh)
        if np.any(finite_mask):
            mean_dist_thresh = float(np.mean(dists_thresh[finite_mask]))
        else:
            mean_dist_thresh = float("inf")

        pct_thresh_within_2 = float(np.mean(dists_thresh <= 2))
        pct_thresh_within_1 = float(np.mean(dists_thresh <= 1))
    else:
        mean_dist_thresh = float("inf")
        pct_thresh_within_2 = 0.0
        pct_thresh_within_1 = 0.0

    return {
        "mean_dist_topk": mean_dist,
        "pct_within_2_topk": pct_within_2,
        "boundary_coverage_topk_1hop": boundary_coverage_topk_1hop,
        "boundary_coverage_topk_2hop": boundary_coverage_topk_2hop,
        "mean_dist_thresh": mean_dist_thresh,
        "pct_thresh_within_1": pct_thresh_within_1,
        "pct_thresh_within_2": pct_thresh_within_2,
        "boundary_coverage_thresh_0": boundary_coverage_thresh_0,
        "boundary_coverage_thresh_1": boundary_coverage_thresh_1,
        "boundary_coverage_thresh_2": boundary_coverage_thresh_2,
        "region_iou_k1": region_iou_k1,
        "region_iou_k2": region_iou_k2,

    }



def predict_with_threshold(out, threshold):
    probs = torch.softmax(out, dim=1)
    return (probs[:, 1] >= threshold).long()

@torch.no_grad()
def evaluate_loss(data, model, class_weights=None, soft_class_weights=None):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)

    labeled_mask = data.label_mask if hasattr(data, 'label_mask') else (data.y >= 0)

    # loss = F.cross_entropy(out, data.y, reduction="mean")
    # loss = focal_loss(out, data.y, gamma=2.0).mean()
    if args.loss_type == "focal":
        loss = focal_loss(out[labeled_mask], data.y[labeled_mask], gamma=2.0).mean()
    elif args.loss_type == "ce_weighted":
        loss = F.cross_entropy(
            out[labeled_mask],
            data.y[labeled_mask],
            weight=class_weights.to(out.device)
        )

    elif args.loss_type == "ce_soft":
        loss = F.cross_entropy(
            out[labeled_mask],
            data.y[labeled_mask],
            weight=soft_class_weights.to(out.device)
        )
    elif args.loss_type == "ce":
        loss = F.cross_entropy(out[labeled_mask], data.y[labeled_mask])


    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
        
    return loss.item()


def sanity_check_masks(data, name="graph"):
    total = data.num_nodes
    labeled = data.label_mask.sum().item()
    unlabeled = (~data.label_mask).sum().item()
    n_boundary = (data.y[data.label_mask] == 1).sum().item()
    n_not_boundary = (data.y[data.label_mask] == 0).sum().item()

    print(f"[SANITY] {name} | total={total}, labeled={labeled}, unlabeled={unlabeled}, boundary=1: {n_boundary}, boundary=0: {n_not_boundary}")

@torch.no_grad()
def compute_density_stats(model, data):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    
    labeled_mask = data.label_mask.cpu().numpy() if hasattr(data, 'label_mask') else (data.y.cpu().numpy() >= 0)
    y_true = data.y.cpu().numpy()
    true_ratio = float((y_true[labeled_mask] == 1).mean())
    mean_prob = float(probs[labeled_mask].mean())
    argmax_ratio = float((probs[labeled_mask] >= 0.5).mean())

    return {
        "true_ratio": true_ratio,
        "mean_pred_prob": mean_prob,
        "argmax_ratio_0.5": argmax_ratio
    }

### threholding start
@torch.no_grad()
def get_probs_and_labels(model, data):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    labels = data.y.cpu().numpy()
    labeled_mask = data.label_mask.cpu().numpy() if hasattr(data, 'label_mask') else (labels >= 0)
    return probs[labeled_mask], labels[labeled_mask]
    


### threholding end
NUM_CATEGORICAL = 14
class SelectiveScaler:
    def __init__(self, n_categorical):
        self.n_cat = n_categorical
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        X = X.copy()
        X[:, self.n_cat:] = self.scaler.fit_transform(X[:, self.n_cat:])
        return X

    def transform(self, X):
        X = X.copy()
        X[:, self.n_cat:] = self.scaler.transform(X[:, self.n_cat:])
        return X


##### training  and eval functions:

def run_training(train_graphs, train_data, train_loader, in_dim, out_dim, id2name=None, model_name = "gat", use_weighted_loss = False, val_graphs=None, test_graphs=None):

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
    elif model_name =="mlp":
        model = MLP(in_dim, 256, out_dim)
        print("[INFO] using MLP model")
    elif model_name =="gatv2":
        model = gatv2(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim).to(device)
        print("[INFO] using gatv2")
    elif model_name =="gin":
        model = GIN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using gatv2")
    elif model_name == 'dGNN':
        model = DirectedGAT(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using DirectedGAT")
    elif model_name == 'hGNN':
        model = HierarchicalGAT(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
        print("[INFO] using HierarchicalGAT")
    elif model_name == 'hdGNN':
        model = HierarchicalDirectedGAT(in_channels = in_dim, hidden_channels = 512, out_channels = out_dim, dropout=0.1).to(device)
        print("[INFO] using HierarchicalDirectedGAT")
    elif model_name =="FlatDirectedGAT":
        model = FlatDirectedGAT(in_channels = in_dim, hidden_channels = 256,num_layers=6,  out_channels = out_dim, dropout=0.1)
        print("[INFO] using FlatDirectedGAT")
    elif model_name =="DirectedOnlyGAT":
        model = DirectedOnlyGAT(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim, dropout=0.1).to(device)
        print("[INFO] using DirectedOnlyGAT")
    elif model_name =="HierarchicalOnlyGAT4":
        model = HierarchicalOnlyGAT4(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim, dropout=0.1)
        print("[INFO] using HierarchicalOnlyGAT4")
    elif model_name =="HierarchicalOnlyGAT6":
        model = HierarchicalOnlyGAT6(in_channels = in_dim, hidden_channels = 256,  out_channels = out_dim, dropout=0.1)
        print("[INFO] using HierarchicalOnlyGAT6")
    elif model_name =="HierarchicalDirectedGAT_v2":
        model = HierarchicalDirectedGAT_v2(in_channels = in_dim, hidden_channels = 256,  out_channels = out_dim, dropout=0.1)
        print("[INFO] using HierarchicalDirectedGAT_v2")
    elif model_name == "DirectedOnlyGATWithGlobal":
        model = DirectedOnlyGATWithGlobal(
            in_channels=in_dim, hidden_channels=256, out_channels=out_dim, dropout=0.1
        )
    elif model_name =="GAAN":
        model = GAAN( in_channels=in_dim, hidden_channels=256, out_channels=out_dim, heads=6)
        print("[INFO] using GaAN")
    elif model_name =="GraphSAGE_ResNorm":
        model = GraphSAGE_ResNorm(in_channels=in_dim, hidden_channels=256, out_channels=out_dim, num_layers=2)
        print("[INFO] using GraphSAGE_ResNorm ")


    elif model_name =="BiDirectedGraphSAGE":
        model = BiDirectedGraphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
        print("[INFO] using BiDirectedGraphSAGE ")
    elif model_name =="JK_GraphSAGE":
        model = JK_GraphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
        print("[INFO] using JK_GraphSAGE -- 6 layers ")
    elif model_name =="BiDirectedJK_GraphSAGE":
        model = BiDirectedJK_GraphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
        print("[INFO] using BiDirectedJK_GraphSAGE -- 6 layers ")
    model = model.to(device)

    ##### training parameters 
    # base_lr = 0.01 
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # if args.use_scheduler:
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5 )  
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6 )

    # ---- coverage tracking ----
    ever_seen_nodes = set()
    num_nodes = train_data.num_nodes

    boundary_nodes = set(
        torch.where(train_data.y == 1)[0].cpu().tolist()
    )
    num_boundary = len(boundary_nodes)



    class_weights = None
    soft_class_weights = None

    if args.loss_type in ["ce_weighted", "ce_soft"]:
        print("[INFO] Computing per-design averaged class weights")

        per_graph_weights = []

        for g in train_graphs:
            # Only use labeled nodes for computing class weights
            labeled_mask = g.label_mask if hasattr(g, 'label_mask') else (g.y >= 0)
            y = g.y[labeled_mask].cpu().numpy()  # Filter to labeled only
            
            classes = np.unique(y)

            # safety: skip degenerate graphs (need both class 0 and 1)
            if len(classes) < 2 or not (0 in classes and 1 in classes):
                continue

            w = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),  # Explicitly specify the two classes
                y=y
            )
            w = w / np.mean(w)   # normalize per graph
            per_graph_weights.append(w)

        if len(per_graph_weights) == 0:
            raise ValueError("No graphs with both classes 0 and 1 found!")

        # average across designs
        weights = np.mean(np.stack(per_graph_weights), axis=0)
        class_weights = torch.tensor(weights, dtype=torch.float)
        # class_weights = torch.tensor([0.05, 1000.0], dtype=torch.float)

        # class_weights = torch.tensor([0.1, 10.0], dtype=torch.float)


        # soften (same as before)
        alpha = 0.4 # 0.4, 0.6, 0.8
        soft_weights = weights ** alpha
        soft_weights = soft_weights / np.mean(soft_weights)
        soft_class_weights = torch.tensor(soft_weights, dtype=torch.float)

        print("[INFO] CE weights (design-avg)     :", class_weights.tolist())
        print("[INFO] Soft CE weights (design-avg):", soft_class_weights.tolist())


    #### main training loop now 
    ## block also for early stopping
    best_val_score = -float("inf")
    best_model_state = None
    patience = 5
    patience_counter = 0
    for epoch in range (1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()

        if args.training_mode == "graphsaint":
            loss, epoch_nodes = train(
                model,
                train_loader,
                optimizer,
                class_weights=class_weights if args.loss_type == "ce_weighted" else None,
                soft_class_weights=soft_class_weights if args.loss_type == "ce_soft" else None,
            )
            # ---- coverage stats ----
            epoch_seen = len(epoch_nodes)
            ever_seen_nodes.update(epoch_nodes)
            cumulative_seen = len(ever_seen_nodes)

            epoch_coverage = epoch_seen / num_nodes
            cumulative_coverage = cumulative_seen / num_nodes

            epoch_boundary_seen = len(epoch_nodes & boundary_nodes)
            cumulative_boundary_seen = len(ever_seen_nodes & boundary_nodes)
            boundary_coverage = (
                cumulative_boundary_seen / num_boundary if num_boundary > 0 else 0.0
            )

            print(
                f"[COVERAGE] epoch={epoch:03d} | "
                f"epoch_seen={epoch_seen}/{num_nodes} ({epoch_coverage:.3f}) | "
                f"cumulative_seen={cumulative_seen}/{num_nodes} ({cumulative_coverage:.3f})"
            )

            print(
                f"[BOUNDARY] epoch={epoch:03d} | "
                f"epoch_seen={epoch_boundary_seen}/{num_boundary} | "
                f"cumulative_seen={cumulative_boundary_seen}/{num_boundary} "
                f"({boundary_coverage:.3f})"
            )

            # wandb.log({
            #     "coverage/epoch_ratio": epoch_coverage,
            #     "coverage/cumulative_ratio": cumulative_coverage,
            #     "coverage_boundary/cumulative_ratio": boundary_coverage,
            #     "epoch": epoch,
            # })

        elif args.training_mode == "fullgraph":
            if args.fullgraph_mode == "merged":
                loss = train_fullgraph(
                    model,
                    train_data,
                    optimizer,
                    class_weights=class_weights if args.loss_type == "ce_weighted" else None,
                    soft_class_weights=soft_class_weights if args.loss_type == "ce_soft" else None,
                )

            elif args.fullgraph_mode == "per_design":
                loss = train_equal_design_weight(
                    model,
                    train_graphs,
                    optimizer,
                    class_weights=class_weights if args.loss_type == "ce_weighted" else None,
                    soft_class_weights=soft_class_weights  if args.loss_type == "ce_soft" else None,
                )

            elif args.fullgraph_mode == "per_family":
                loss = train_family_weighted(
                    model, train_graphs, args.train_gml, optimizer,
                    class_weights=class_weights if args.loss_type == "ce_weighted" else None,
                    soft_class_weights=soft_class_weights  if args.loss_type == "ce_soft" else None,
                )

            else:
                raise ValueError(f"Unknown fullgraph_mode: {args.fullgraph_mode}")

        # loss, epoch_nodes = train(model, train_loader, optimizer, class_weights=class_weights if args.loss_type== "ce_weighted" else None, soft_class_weights=soft_class_weights if args.loss_type=="ce_soft" else None)
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
        # scheduler.step()

        if epoch % 5 == 0 :
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
                name = os.path.splitext(os.path.basename(path))[0]
                m = evaluate_test(g, model)
                probs, labels = get_probs_and_labels(model, g)   # probs = P(y=1)


                # print(f"[SANITY] {name} | Mean prob boundary=1: {probs[labels==1].mean():.4f} | Mean prob boundary=0: {probs[labels==0].mean():.4f}")
                mean_prob_pos = float(probs[labels == 1].mean()) if np.any(labels == 1) else float("nan")
                mean_prob_neg = float(probs[labels == 0].mean()) if np.any(labels == 0) else float("nan")
                prob_gap = mean_prob_pos - mean_prob_neg

                print(
                    f"[CHECK] {name} | "
                    f"Mean prob boundary=1: {mean_prob_pos:.4f} | "
                    f"Mean prob boundary=0: {mean_prob_neg:.4f}"
                )

                # logging for mean prob metrics
                wandb.log({
                    "epoch": epoch,
                    f"val_prob/{name}/mean_prob_boundary": mean_prob_pos,
                    f"val_prob/{name}/mean_prob_not_boundary": mean_prob_neg,
                    f"val_prob/{name}/prob_gap": prob_gap,
                })
                pr_auc = pr_auc_from_probs(labels, probs)
                val_metrics["pr_auc"].append(pr_auc)
                
                val_loss = evaluate_loss(g, model, class_weights, soft_class_weights)
                

                print(
                    f"[VAL][Epoch {epoch:03d}] {name} | "
                    f"Val Loss={val_loss:.4f}, "
                    f"F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, "
                    f"PR-AUC={pr_auc:.4f}, "
                    f"Acc={m['total_acc']:.4f}, b1={m['boundary_1_acc']:.4f}, b0={m['boundary_0_acc']:.4f}"
                )

                # store for macro avg
                for k, v in m.items():
                    val_metrics[k].append(float(v))

                # per-graph wandb
                wandb.log({
                    "epoch": epoch,
                    f"val_pgraph/{name}/f1": m["f1"],
                    f"val_pgraph/{name}/pr_auc": pr_auc,
                    f"val_pgraph/{name}/loss": val_loss,
                    f"val_pgraph/{name}/precision": m["precision"],
                    f"val_pgraph/{name}/recall": m["recall"],
                    f"val_pgraph/{name}/accuracy": m["total_acc"],
                    f"val_pgraph/{name}/boundary_acc": m["boundary_1_acc"],
                    f"val_pgraph/{name}/not_boundary_acc": m["boundary_0_acc"],
                })

                val_losses.append(val_loss)

                density_stats = compute_density_stats(model, g)
                print(
                    f"[DENSITY][Epoch {epoch:03d}] {name} | "
                    f"TrueRatio={density_stats['true_ratio']:.4f}, "
                    f"MeanProb={density_stats['mean_pred_prob']:.4f}, "
                    f"ArgmaxRatio@0.5={density_stats['argmax_ratio_0.5']:.4f}"
                )

                wandb.log({
                    "epoch": epoch,
                    f"val_density/{name}/true_ratio": density_stats["true_ratio"],
                    f"val_density/{name}/mean_pred_prob": density_stats["mean_pred_prob"],
                    f"val_density/{name}/argmax_ratio_0.5": density_stats["argmax_ratio_0.5"],
                })

                region_metrics = evaluate_region_metrics(g, model, k_percent=2.0, r=2)

                if region_metrics:
                    print(
                        f"[REGION] {name} | "
                        f"mean_dist={region_metrics['mean_dist_topk']:.2f}, "
                        f"pct_within_2={region_metrics['pct_within_2_topk']:.4f}, "
                        f"boundary_coverage_topk_1hop={region_metrics['boundary_coverage_topk_1hop']:.4f}, "
                        f"boundary_coverage_topk_2hop={region_metrics['boundary_coverage_topk_2hop']:.4f}, "
                        f"boundary_coverage_thresh_0={region_metrics['boundary_coverage_thresh_0']:.4f}, "
                        f"boundary_coverage_thresh_1={region_metrics['boundary_coverage_thresh_1']:.4f}, "
                        f"boundary_coverage_thresh_2={region_metrics['boundary_coverage_thresh_2']:.4f}, "
                        f"mean_dist_thresh={region_metrics['mean_dist_thresh']:.4f}, "
                        f"pct_thresh_within_1={region_metrics['pct_thresh_within_1']:.4f}, "
                        f"pct_thresh_within_2={region_metrics['pct_thresh_within_2']:.4f}, "
                        f"region_iou_k1={region_metrics['region_iou_k1']:.4f}, "
                        f"region_iou_k2={region_metrics['region_iou_k2']:.4f}, "
                    )
                else: 
                    print(f"[REGION] {name} | Skipped (no boundary nodes)")

            # macro avg over val designs
            val_macro = {k: float(np.mean(v)) for k, v in val_metrics.items()} if len(val_graphs) else {}
            print(
                f"[VAL][Epoch {epoch:03d}] MACRO | "
                f"Val Macro Loss={float(np.mean(val_losses))}, "
                f"F1={val_macro.get('f1', 0.0):.4f}, "
                f"P={val_macro.get('precision', 0.0):.4f}, "
                f"R={val_macro.get('recall', 0.0):.4f}, "
                f"PR-AUC={val_macro.get('pr_auc', 0.0):.4f}, "
                f"Acc={val_macro.get('total_acc', 0.0):.4f}, "
                f"b1={val_macro.get('boundary_1_acc', 0.0):.4f}, "
                f"b0={val_macro.get('boundary_0_acc', 0.0):.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "val_macro/f1": val_macro.get("f1", 0.0),
                "val_macro/pr_auc": val_macro.get("pr_auc", 0.0),
                "val_macro/loss": float(np.mean(val_losses)),
                "val_macro/precision": val_macro.get("precision", 0.0),
                "val_macro/recall": val_macro.get("recall", 0.0),
                "val_macro/accuracy": val_macro.get("total_acc", 0.0),
                "val_macro/boundary_acc": val_macro.get("boundary_1_acc", 0.0),
                "val_macro/not_boundary_acc": val_macro.get("boundary_0_acc", 0.0),
                "time/val_eval_sec": time.perf_counter() - eval_start,
            })

            # ---- EARLY STOPPING ----
            current_score = val_macro.get("pr_auc", 0.0)

            # add these lines:
            if args.use_scheduler:
                scheduler.step(current_score)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[SCHEDULER] Current LR: {current_lr:.6f}")
                wandb.log({"train/lr": current_lr, "epoch": epoch})


            if current_score > best_val_score + 1e-4:
                best_val_score = current_score
                patience_counter = 0
                # best_model_state = model.state_dict()
                best_model_state = copy.deepcopy(model.state_dict())

                print(f"[EARLY STOP] New best PR-AUC: {best_val_score:.4f}")

            else:
                patience_counter += 1
                print(f"[EARLY STOP] No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"[EARLY STOP] Triggered at epoch {epoch}")
                break


        if epoch % 20 == 0:
            print(f"\n[TEST PEEK][Epoch {epoch:03d}]")
            for path, g in test_graphs:
                name = os.path.splitext(os.path.basename(path))[0]
                n = evaluate_test(g, model)
                probs, labels = get_probs_and_labels(model, g)
                pr_auc = pr_auc_from_probs(labels, probs)

                print(
                    f"[TEST PEEK] {name} | "
                    f"F1={n['f1']:.4f}, P={n['precision']:.4f}, R={n['recall']:.4f}, "
                    f"PR-AUC={pr_auc:.4f}"
                )

                wandb.log({
                    "epoch": epoch,
                    f"test_peek/{name}/f1": n["f1"],
                    f"test_peek/{name}/precision": n["precision"],
                    f"test_peek/{name}/recall": n["recall"],
                    f"test_peek/{name}/pr_auc": pr_auc,
                    f"test_peek/{name}/boundary_acc": n["boundary_1_acc"],
                    f"test_peek/{name}/not_boundary_acc": n["boundary_0_acc"],
                })    

    ### Save model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("[INFO] Loaded best model from early stopping")
    else:
        print("[INFO] no best model found, saving current model")
    run_dir =  os.path.join("models", "gnns", wandb.run.name)
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
        sanity_check_masks(graph_data, name=f"TRAIN[{i}] {os.path.basename(gml_path)}")
        print("Label counts:", Counter(graph_data.y.tolist())) # debug for -1  label
        assert (graph_data.y[graph_data.label_mask] < 0).sum().item() == 0, "Labeled nodes have invalid labels!"


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
        assert (g.y[g.label_mask] < 0).sum().item() == 0

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
        assert (g.y[g.label_mask] < 0).sum().item() == 0
        g.train_mask[:] = False
        g.val_mask[:] = False
        g.test_mask[:] = False
        test_graphs.append((gml_path, g))


    ### merges / combines -- using reduce 
    combined_data = reduce(merge_data, train_graphs)

    ### saving also the unscaled features 

    # train unscaled
    X_train_raw = []
    y_train_raw = []
    for g in train_graphs:
        X_train_raw.append(g.x.cpu().numpy())
        y_train_raw.append(g.y.cpu().numpy())
    X_train_raw = np.vstack(X_train_raw)
    y_train_raw = np.concatenate(y_train_raw)

    # val unscaled
    val_raw = []
    for path, g in val_graphs:
        val_raw.append((path, g.x.cpu().numpy(), g.y.cpu().numpy()))
        
    # test unscaled
    test_raw = []
    for path, g in test_graphs:
        test_raw.append((path, g.x.cpu().numpy(), g.y.cpu().numpy()))


    # train has the scaler and its fit here
    scaler = SelectiveScaler(NUM_CATEGORICAL)
    combined_data.x = torch.tensor(
        scaler.fit_transform(combined_data.x.cpu().numpy()),
        dtype=torch.float32
    )


    
    for g in train_graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)
        
    # we use the train sclaer to transform val and test
    for path, g in val_graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)

    for path, g in test_graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)

    # # train: scale + augment (write back by index)
    # for i, g in enumerate(train_graphs):
    #     g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)
    #     if args.use_augmentation:
    #         train_graphs[i] = augment_graph_noise(g, n_categorical=NUM_CATEGORICAL,
    #                                                gate_flip_frac=args.aug_gate_flip_frac,
    #                                                edge_corrupt_frac=args.aug_edge_corrupt_frac)

    # # val: scale only, always clean
    # for path, g in val_graphs:
    #     g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)

    # # test: scale + augment (write back by index, preserve path)
    # for i, (path, g) in enumerate(test_graphs):
    #     g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float32)
    #     if args.use_augmentation:
    #         test_graphs[i] = (path, augment_graph_noise(g, n_categorical=NUM_CATEGORICAL,
    #                                                      gate_flip_frac=args.aug_gate_flip_frac,
    #                                                      edge_corrupt_frac=args.aug_edge_corrupt_frac))


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
    training_data_loader = None
    if args.training_mode  == "graphsaint":
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
        else: 
            raise ValueError(f"Unsupported sampling_method: {args.sampling_method}")
    # elif args.sampling_method ==  "graphsaint":
    #     data_loader = GraphSAINTSampler(
    #         combined_data, 
    #         batch_size = int((args.perc_batchsize)*combined_data.num_nodes ), 
    #         num_steps = args.num_steps, 
    #         sample_coverage = args.sample_coverage
    #     )

    # (b) khop sampler (to do )


    ## calling the model 
    # model = run_training(
    #     train_data=combined_data,
    #     train_loader=training_data_loader,
    #     in_dim=combined_data.num_features,
    #     out_dim=2,
    #     id2name=id2label,
    #     model_name=args.model,
    #     use_weighted_loss=False,
    #     val_graphs=val_graphs,
    #     test_graphs=test_graphs,
    # )


    model = run_training(
        train_graphs=train_graphs,
        train_data=combined_data,
        train_loader=training_data_loader,
        in_dim=combined_data.num_features,
        out_dim=2,
        id2name=id2label,
        model_name=args.model,
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
        name = os.path.splitext(os.path.basename(path))[0]
        n = evaluate_test(g, model)

        probs, labels = get_probs_and_labels(model, g)
        pr_auc = pr_auc_from_probs(labels, probs)

        mean_prob_pos = float(probs[labels == 1].mean()) if np.any(labels == 1) else float("nan")
        mean_prob_neg = float(probs[labels == 0].mean()) if np.any(labels == 0) else float("nan")

        print(
            f"[TEST] {name} | "
            f"F1={n['f1']:.4f}, P={n['precision']:.4f}, R={n['recall']:.4f}, "
            f"Acc={n['total_acc']:.4f}, PR-AUC={pr_auc:.4f}, "
            f"MeanProb b=1: {mean_prob_pos:.4f}, b=0: {mean_prob_neg:.4f}"
        )

        for k, v in n.items():
            test_metrics[k].append(float(v))
        test_metrics["pr_auc"].append(pr_auc)

        wandb.log({
            f"test/{name}/f1": n["f1"],
            f"test/{name}/precision": n["precision"],
            f"test/{name}/recall": n["recall"],
            f"test/{name}/accuracy": n["total_acc"],
            f"test/{name}/boundary_acc": n["boundary_1_acc"],
            f"test/{name}/not_boundary_acc": n["boundary_0_acc"],
            f"test/{name}/pr_auc": pr_auc,
            f"test/{name}/mean_prob_boundary": mean_prob_pos,
            f"test/{name}/mean_prob_not_boundary": mean_prob_neg,
        })

        region_metrics = evaluate_region_metrics(g, model, k_percent=2.0, r=2)
        if region_metrics:
            print(
                f"[REGION] {name} | "
                f"mean_dist_topk={region_metrics['mean_dist_topk']:.2f}, "
                f"pct_within_2_topk={region_metrics['pct_within_2_topk']:.4f}, "
                f"boundary_coverage_topk_1hop={region_metrics['boundary_coverage_topk_1hop']:.4f}, "
                f"boundary_coverage_topk_2hop={region_metrics['boundary_coverage_topk_2hop']:.4f}, "
                f"boundary_coverage_thresh_0={region_metrics['boundary_coverage_thresh_0']:.4f}, "
                f"boundary_coverage_thresh_1={region_metrics['boundary_coverage_thresh_1']:.4f}, "
                f"boundary_coverage_thresh_2={region_metrics['boundary_coverage_thresh_2']:.4f}, "
                f"mean_dist_thresh={region_metrics['mean_dist_thresh']:.4f}, "
                f"pct_thresh_within_1={region_metrics['pct_thresh_within_1']:.4f}, "
                f"pct_thresh_within_2={region_metrics['pct_thresh_within_2']:.4f}, "
            )
        else:
            print(f"[REGION] {name} | skipped (no boundary nodes)")


    test_macro = {k: float(np.mean(v)) for k, v in test_metrics.items()} if len(test_graphs) else {}
    print(
        f"[TEST] MACRO | "
        f"F1={test_macro.get('f1', 0.0):.4f}, "
        f"P={test_macro.get('precision', 0.0):.4f}, "
        f"R={test_macro.get('recall', 0.0):.4f}, "
        f"Acc={test_macro.get('total_acc', 0.0):.4f}, "
        f"b1={test_macro.get('boundary_1_acc', 0.0):.4f}, "
        f"b0={test_macro.get('boundary_0_acc', 0.0):.4f},"
        f"PR-AUC={test_macro.get('pr_auc', 0.0):.4f}, "
    )

    wandb.log({
        "test_macro/f1": test_macro.get("f1", 0.0),
        "test_macro/precision": test_macro.get("precision", 0.0),
        "test_macro/recall": test_macro.get("recall", 0.0),
        "test_macro/accuracy": test_macro.get("total_acc", 0.0),
        "test_macro/boundary_acc": test_macro.get("boundary_1_acc", 0.0),
        "test_macro/not_boundary_acc": test_macro.get("boundary_0_acc", 0.0),
        "test_macro/pr_auc": test_macro.get("pr_auc", 0.0),
    })

    # saving the results in gml 
    output_dir = make_result_dir("gnns")
    print("[INFO] GNN results will be saved to:", output_dir)

    output_labels = {0: "not_boundary", 1: "boundary"}
    print("[INFO] Output Labels:", output_labels)

    for path, g in test_graphs:
        p = Path(path)
        test_graph_name = p.stem
        lib_name = p.parent.name
        design_family = p.parent.parent.name

        safe_name = f"{design_family}__{lib_name}__{test_graph_name}"
        output_path = os.path.join(output_dir, f"{safe_name}_predictions.gml")

        print("[INFO] Saving predictions to:", output_path)

        save_predictions_to_gml(
            original_gml_path=path,
            data=g,
            model=model,
            id2name=output_labels,
            output_gml_path=output_path
        )