import torch
import os
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from main import (
    set_seed,
    save_predictions_to_gml
)
import wandb
import random
import networkx as nx
from preprocessing import normalize_features
import numpy as np
from collections import defaultdict, Counter
from torch_geometric.data import Data
from gnn.graphSAGE import graphSAGE
from gnn.gcn import GCN
from gnn.gat import gat
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_method", type=str, choices=["graphsaint", "khop"], default="graphsaint",
                    help="Sampling method: 'graphsaint' or 'khop'")
args = parser.parse_args()

wandb.init(project="gnn-subcircuit-detection", name="aes_to_des_test")
set_seed(42)


def ego_subgraphs_from_data(full_data, radius=2, num_subgraphs=10, seed=42):
    random.seed(seed)
    G = nx.Graph()
    edge_list = full_data.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    subgraph_data_list = []

    for _ in range(num_subgraphs):
        center = random.choice(range(full_data.num_nodes))
        nodes = nx.ego_graph(G, center, radius=radius).nodes()
        nodes = list(nodes)

        sub_nodes = torch.tensor(nodes, dtype=torch.long)
        sub_edge_index, _ = subgraph(sub_nodes, full_data.edge_index, relabel_nodes=True)

        sub_data = Data(
            x=full_data.x[sub_nodes],
            edge_index=sub_edge_index,
            y=full_data.y[sub_nodes],
            train_mask=full_data.train_mask[sub_nodes],
            val_mask=full_data.val_mask[sub_nodes],
            test_mask=full_data.test_mask[sub_nodes],
        )
        subgraph_data_list.append(sub_data)

    return subgraph_data_list

# def train(model, loader, optimizer, class_weights = None):
#     model.train()
#     total_loss = 0
    

#     # batch.x = node features
#     # batch.edge_index = edge index of the subgraph
#     # batch.y = true labels

#     for batch in loader: # iterates over batches from the graph sampler ( each batch is a subgraph)
#         optimizer.zero_grad()
#         out = model(batch.x, batch.edge_index) # (node features, edge index) and out is prediction logits of shape: [num_nodes_in_batch, num_classes]

#         # we use -1 now to mark nodes we dont want in training
#         # also input and ouput nodes are already labeled -1 
#         valid_mask = (batch.y != -1) & (batch.train_mask)
#         if valid_mask.sum() == 0:
#             continue

#         if class_weights is not None:
#             loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight=class_weights)
#         else:
#             loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])
            
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * valid_mask.sum().item()  # sum loss over valid nodes
#         total_valid_nodes = valid_mask.sum().item()
        
#     return total_loss / total_valid_nodes


def train(model, loader, optimizer, class_weights=None):
    model.train()
    total_loss = 0
    batch_count = 0  # new counter

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        # only train on valid, labeled nodes
        valid_mask = (batch.y != -1) & (batch.train_mask)
        if valid_mask.sum() == 0:
            continue

        if class_weights is not None:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask], weight=class_weights)
        else:
            loss = F.cross_entropy(out[valid_mask], batch.y[valid_mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()   # no multiplication
        batch_count += 1            # count batches

    # return average loss per batch (easy to read)
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    valid_mask = mask & (data.y != -1) 
    correct = (pred[valid_mask] == data.y[valid_mask]).sum().item()

    accuracy = correct / valid_mask.sum().item() 
    return accuracy

@torch.no_grad()
def evaluate_binary(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    
    # pred = out.argmax(dim=1) # agressive 
    probs = torch.softmax(out, dim=1) # not so agressive (1) 
    pred = (probs[:, 1] > 0.7).long() # not so agressive (2) 

    valid_mask = mask & (data.y != -1)

    y_true = data.y[valid_mask].cpu().numpy()
    y_pred = pred[valid_mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall



def classwise_accuracy(model, data, mask, id2name=None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # y_true = data.y[mask]
    # y_pred = pred[mask]
    valid_mask = mask & (data.y != -1)
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

def run_training(data, train_loader, in_dim, out_dim, id2name=None, model_name="graphsage", use_weighted_loss=False):
    if model_name == "graphsage":
        model = graphSAGE(in_channels=in_dim, hidden_channels=256, out_channels=out_dim)
    elif model_name == "gcn":
        model = GCN(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)
    elif model_name == "gat":
        model = gat(in_channels = in_dim, hidden_channels = 256, out_channels = out_dim)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01 ) # weight_decay=5e-4

    if use_weighted_loss:
        train_labels = data.y[data.train_mask].cpu().numpy()
        train_labels = train_labels[train_labels != -1]
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float, device=data.x.device)
        print("Using weighted cross entropy loss.")
    else:
        class_weights = None
        print("Using standard cross entropy loss.")
        

    epochs = 500
    for epoch in range(1, epochs+1):
        loss = train(model, train_loader, optimizer, class_weights)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)

        log_data = {
            "epoch": epoch,
            "loss": loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        }

        if out_dim == 2:
            f1, precision, recall = evaluate_binary(model, data, data.val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        else:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}')

        if epoch % 100 == 0 or epoch == 1:
            classwise_acc = classwise_accuracy(model, data, data.val_mask, id2name)
        #     wandb.log({
        #         "epoch": epoch,
        #         "val_classwise_accuracy": {
        #             cls: acc for cls, acc in classwise_acc.items()
        #         }
        #     })
        #     print("  Val Class-wise Accuracy:")
        #     for cls, acc in classwise_acc.items():
        #         print(f"    Class {cls}: {acc:.4f}")
        
        # wandb.log(log_data)



    test_acc = evaluate(model, data, data.test_mask)
    # wandb.log({"final_test_accuracy": test_acc})
    print(f"Final test accuracy: {test_acc:.4f}")

    if out_dim == 2:  # binary classification
        f1, precision, recall = evaluate_binary(model, data, data.test_mask)
        print(f"Binary classification metrics:")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
    
    classwise_acc = classwise_accuracy(model, data, data.test_mask, id2name)
    print("  Test Class-wise Accuracy:")
    for cls, acc in classwise_acc.items():
        # wandb.log({f"test_acc/{cls}": acc})
        print(f"    Class {cls}: {acc:.4f}")
        
    return model


def load_aisec_single_gml(gml_path, label_type="subcircuit", binary_label=False, positive_class=None, remove_edges=False):
    print("calling gnn from path:", gml_path)
    random.seed(42)

    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features = []
    labels = []

    for node in nodes:
        attr = G.nodes[node]

        # feat = list(map(int, attr.get("features", [])))
        # features.append(feat)

        feat = attr.get("features", [])
        if isinstance(feat, (list, tuple, np.ndarray)):
            features.append(feat)
        else:
            features.append([])

        if binary_label:
            label_name = attr.get("subcircuit_name", "unknown")
            if label_name == "sbox":
                labels.append(1)   # positive
            else:
                labels.append(0)   # negative
        elif label_type == "subcircuit":
            labels.append(attr.get("subcircuit", -1))
        else:
            labels.append(int(attr.get("subcircuit_original", -1)))

    features = normalize_features(np.array(features))

    
    if binary_label:
        id2label = {0: "not_sbox", 1: "sbox"}
        labels = torch.tensor(labels, dtype=torch.long)
    elif label_type == "subcircuit":
        label_set = sorted(set(labels))
        label_map = {v: i for i, v in enumerate(label_set)}
        labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
        id2label = {i: l for l, i in label_map.items()}
    else:
        labels = torch.tensor(labels, dtype=torch.long)
        id2label = {int(l): str(l) for l in sorted(set(labels.tolist()))}

    print(f"Total labels: {len(labels)}")
    print(f"Unique labels: {sorted(set(labels.tolist()))}")
    label_counts = Counter(labels.tolist())
    print("Label counts:", label_counts)

    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    edges = [(node_map[src], node_map[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_nodes = len(nodes)
    indices = list(range(num_nodes))
    random.shuffle(indices)

    train_cutoff = int(0.6 * num_nodes)
    val_cutoff = train_cutoff + int(0.2 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_cutoff]] = True
    val_mask[indices[train_cutoff:val_cutoff]] = True
    test_mask[indices[val_cutoff:]] = True

    # === Optionally remove cross-split edges ===
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

    # === Final Data object ===
    data = Data(
        # x=torch.tensor(features, dtype=torch.float),
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, id2label


def merge_data(data1, data2):
    # Offset node indices in data2's edge_index
    offset = data1.num_nodes
    data2_edge_index = data2.edge_index + offset

    # Concatenate all fields
    x = torch.cat([data1.x, data2.x], dim=0)
    edge_index = torch.cat([data1.edge_index, data2_edge_index], dim=1)
    y = torch.cat([data1.y, data2.y], dim=0)

    train_mask = torch.cat([data1.train_mask, data2.train_mask], dim=0)
    val_mask = torch.cat([data1.val_mask, data2.val_mask], dim=0)
    test_mask = torch.cat([data1.test_mask, data2.test_mask], dim=0)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

aes_data, id2label = load_aisec_single_gml(
    # gml_path="graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml",
    gml_path="graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi.gml",
    binary_label=True,   # sbox vs not_sbox
)


# second_data, _ = load_aisec_single_gml(
#     gml_path="graphs/processed/aes_encryption_latest/osu035/aes_key_expand_128_gephi.gml",
#     # gml_path="graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi.gml",
#     # gml_path="graphs/processed/mips_16_latest/osu035/mips_16_core_top_gephi.gml",
#     binary_label=True,
# )

# third_data, _ = load_aisec_single_gml(
#     gml_path= "graphs/processed/aes_encryption_latest/nangate/aes_cipher_top_gephi.gml",
#     binary_label=True,
# )

# combined_data_a = merge_data(aes_data, second_data)
# combined_data = merge_data(combined_data_a, third_data) # for third

######################################
# gml_paths = [
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_1.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_2.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_3.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_4.gml",
#     "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_5.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_6.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_7.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_8.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_9.gml",
#     # "graphs/synthetic/subgraphs_khop/aes_encryption_latest_noise1/osu035/subgraph_10.gml",

# ]

# data_list = []
# for i, path in enumerate(gml_paths):
#     data, labels = load_aisec_single_gml(gml_path=path, binary_label=True)
#     if i == 0:
#         id2label = labels  # Save from first
#     data_list.append(data)

# from functools import reduce

# combined_data = reduce(merge_data, data_list)
# # aes_data = data_list[0]  # To keep masks for logging or sampling

########################################

print("\n[DEBUG] AES dataset:")
print("id2label:", id2label)
print("Total nodes:", aes_data.num_nodes)
print("SBOX nodes:", (aes_data.y == 1).sum().item())
print("Not-SBOX nodes:", (aes_data.y == 0).sum().item())

# Double-check training nodes come only from AES
print("\n[DEBUG] AES train/val/test split:")
print("  Train nodes:", aes_data.train_mask.sum().item())
print("  Val nodes:", aes_data.val_mask.sum().item())
print("  Test nodes:", aes_data.test_mask.sum().item())

# Sanity: make sure all masks add up to total AES nodes
print("  Total AES nodes:", aes_data.num_nodes)
print("  Sum of masks   :", (aes_data.train_mask.sum() +
                              aes_data.val_mask.sum() +
                              aes_data.test_mask.sum()).item())

# aes_loader = GraphSAINTRandomWalkSampler(
#     # aes_data,
#     combined_data,
#     batch_size=int(0.3 * aes_data.num_nodes),
#     walk_length=5,
#     # num_steps=20,  
#     shuffle=True,
# )

if args.sampling_method == "graphsaint":
    print("[INFO] Using GraphSAINT sampling...")
    aes_loader = GraphSAINTRandomWalkSampler(
        aes_data, 
        # combined_data,
        batch_size=int(0.35 * aes_data.num_nodes),
        walk_length=10,
        # num_steps=5,
        shuffle=True,
    )
    print("aes_loader lemgth - batch size:", len(aes_loader))


elif args.sampling_method == "khop":
    print(f"[INFO] Using k-hop sampling...")
    subgraph_list = ego_subgraphs_from_data(
        aes_data,
        # combined_data,
        radius=3,
        num_subgraphs=1500  #int(0.3 * aes_data.num_nodes)
    )
    aes_loader = DataLoader(subgraph_list, batch_size=4, shuffle=True)


model = run_training(
        data=aes_data,
        # data = combined_data,
        train_loader=aes_loader,
        in_dim= aes_data.num_features, # aes_data.num_features, # combined_data.num_features
        out_dim=2,   # binary classification
        id2name=id2label,
        model_name="gat",
        use_weighted_loss=True,
)


des_data, _ = load_aisec_single_gml(
    # gml_path="graphs/processed/des_latest/osu035/des_gephi.gml",
    gml_path = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml",
    binary_label=True, 
)
des_data.train_mask[:] = False
des_data.val_mask[:] = False
des_data.test_mask[:] = False

print("\n[DEBUG] DES dataset:")
print("Total nodes:", des_data.num_nodes)
print("SBOX nodes:", (des_data.y == 1).sum().item())
print("Not-SBOX nodes:", (des_data.y == 0).sum().item())
print("\n[DEBUG] DES dataset check (should have *no* training here):")
print("  Train nodes:", des_data.train_mask.sum().item())
print("  Val nodes:", des_data.val_mask.sum().item())
print("  Test nodes:", des_data.test_mask.sum().item())
mask = torch.ones_like(des_data.y, dtype=torch.bool)

f1, precision, recall = evaluate_binary(model, des_data, mask)
print(f"\n=== Cross-graph test (AES→DES) ===")
print(f"F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")

model.eval()
out = model(des_data.x, des_data.edge_index)
pred = out.argmax(dim=1)

y_true = des_data.y.cpu().numpy()
y_pred = pred.cpu().numpy()

total_correct = (y_true == y_pred).sum()
total_acc = total_correct / len(y_true)

sbox_mask = (y_true == 1)
not_sbox_mask = (y_true == 0)

sbox_acc = (y_pred[sbox_mask] == y_true[sbox_mask]).sum() / sbox_mask.sum()
not_sbox_acc = (y_pred[not_sbox_mask] == y_true[not_sbox_mask]).sum() / not_sbox_mask.sum()

print("\n=== DES Accuracy Breakdown ===")
print(f"Total Accuracy   : {total_acc:.4f}")
print(f"SBOX Accuracy    : {sbox_acc:.4f}")
print(f"Not-SBOX Accuracy: {not_sbox_acc:.4f}")

output_dir = "results/aes_to_des"
os.makedirs(output_dir, exist_ok=True)
print("\n[DEBUG] Saving DES predictions to results/aes_to_des/aes_cipher_noise1.gml")
save_predictions_to_gml(
    # original_gml_path="graphs/processed/des_latest/osu035/des_gephi.gml",
    original_gml_path = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml", #aes_key_expand_128_gephi #aes_cipher_top_gephi
    data=des_data,
    model=model,
    id2name={0: "not_sbox", 1: "sbox"},
    output_gml_path=os.path.join(output_dir, "des_predictions.gml"),
)  


