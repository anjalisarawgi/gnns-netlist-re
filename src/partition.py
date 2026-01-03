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
parser.add_argument("--test_gml", type = str, help="which graph (gml_path) do you want to test on?")
parser.add_argument("--epochs", type = int, default=250)

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
parser.add_argument("--normalize_class_weights", action="store_true", help="kinda confused - but to stabalize training? (i think its just like scaling the weights to avoid exploding gradients)")

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
test_name = os.path.splitext(os.path.basename(args.test_gml))[0]
train_roots = [os.path.splitext(os.path.basename(p))[0] for p in args.train_gml]
train_name = "+".join(train_roots[:2]) + ("+test" if len(train_roots) > 2 else "")

if args.sampling_method == "graphsaint_rw":
    sampling_suffix = f"graphsaint_walk{args.walk_length}"
elif args.sampling_method == "khop":
    sampling_suffix = f"khop_r{args.radius}_n{args.num_subgraphs}"
else:
    sampling_suffix = args.sampling_method

run_name = f"{args.perc_batchsize}perc_{config_tag}_{args.model}_{sampling_suffix}_{args.epochs}ep_for_{test_name}"


wandb.init(project="gnn-boundary-detection", name=run_name)
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
    return torch.tensor(features, dtype=torch.float)

    
#### merge_data
def merge_data(gml_1, gml_2):
    # note here we make offsets so we dont have overlapping edge indexes 
    offset = gml_1.num_nodes
    gml_2_edgeIndex = gml_2.edge_index + offset

    # concat 
    x = torch.cat([gml_1.x, gml_2.x], dim=0)
    edge_index = torch.cat([gml_1.edge_index, gml_2_edgeIndex], dim=1)
    y = torch.cat([gml_1.y, gml_2_edgeIndex.y], dim=0)

    train_mask = torch.cat([gml_1.train_mask, gml_2_edgeIndex.train_mask], dim = 0)
    val_mask = torch.cat([gml_1.val_mask, gml_2_edgeIndex.val_mask], dim = 0)
    test_mask = torch.cat([gml_1.test_mask, gml_2_edgeIndex.test_mask], dim = 0)

    # data obj
    mertged_data = Data(
        x = x, 
        edge_index=edges,
        y = y, 
        train_mask = train_mask, 
        val_mask = val_mask, 
        test_mask = test_mask
    )

    return mertged_data

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
        features.append(feat)
    
        boundary_value = attr.get("boundary", 0)
        try:
            label = int(boundary_value)
        except (ValueError, TypeError): ######??? - we wanna change this ***s
            label= 0
        labels.append(label)

    id2label = {0: "not_boundary", 1:"boundary"}
    labels = torch.tensor(labels, dtype = torch.long)

    ## normalizing features ***
    # features = normalize_features(np.array(features, dtype = np.float32)) ### - this becomes one scaler for each graph

    # debugging for checking if everything is okay
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

    # splits --- train / test / val s
    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_cutoff = int(0.90 * num_nodes) # ***
    val_cutoff = train_cutoff + int(0.05 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_cutoff]] = True
    val_mask[indices[train_cutoff:val_cutoff]] = True
    test_mask[indices[val_cutoff:]] = True

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

    data = Data(
        # x = torch.tensor(featues, dtype = torch.float), 
        x = torch.as_tensor(features, dtype = torch.float32), 
        edge_index = edge_index,
        y = labels, 
        train_mask = train_mask, 
        val_mask = val_mask, 
        test_mask = test_mask
    )

    return data, id2label











########
# main #
########
if __name__ == "__main__":
    print("[INFO] Loading training graphs:")
    train_graphs = []
    for i, gml_path in enumerate(args.train_gml):
        print(f"[{i+1}] {gml_path}")
        graph_data, label_map = load_single_gml(gml_path = gml_path, remove_edges=True)

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
    testgml_data, _ = load_single_gml(gml_path = args.test_gml, remove_edges=True)
    testgml_data.train_mask[:]= False
    testgml_data.val_mask[:]= False
    testgml_data.test_mask[:]= False
    print("[INFO] Total number of nodes:", testgml_data.num_nodes)

    ### merges / combines -- using reduce 
    combined_data = reduce(merge_data, train_graphs)
    combined_data.global_id = torch.arange(combined_data.num_nodes)  # setting global ids now which is permanent 
    print("[INFO] training on:", args.train_gml)
    print("[INFO] Number of features:", full_data.num_features)
    print("[INFO] Feature matrix shape:", full_data.x.shape)
    print(f"[INFO] Number of nodes for boundary = 1:", (full_data.y == 1).sum().item())
    print(f"[INFO] Number of nodes for boundary = 0:", (full_data.y == 0).sum().item())
    print("[INFO] Total training nodes:", full_data.num_nodes)
    print("[INFO] (a) Train nodes:", full_data.train_mask.sum().item())
    print("[INFO] (b) Val nodes  :", full_data.val_mask.sum().item())
    print("[INFO] (c) Test nodes :", full_data.test_mask.sum().item())

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
    

    