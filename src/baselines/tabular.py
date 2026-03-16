
# import argparse
# import os
# import sys
# import numpy as np
# import networkx as nx
# import torch
# import wandb
# import yaml
# from pathlib import Path
# from functools import reduce
# from datetime import datetime

# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
# from torch_geometric.data import Data
# from tqdm import tqdm
# import random
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# # ── logging ────────────────────────────────────────────────────────────────────
# class Tee:
#     def __init__(self, *files): self.files = files
#     def write(self, obj):
#         for f in self.files: f.write(obj); f.flush()
#     def flush(self):
#         for f in self.files: f.flush()
#     def isatty(self):
#         return any(getattr(f, "isatty", lambda: False)() for f in self.files)

# os.makedirs("logs", exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_file  = open(f"logs/baseline_{timestamp}.log", "w")
# sys.stdout = Tee(sys.__stdout__, log_file)
# sys.stderr = Tee(sys.__stderr__, log_file)


# # ── args ───────────────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser(description="Tabular baselines: RF, TabPFN v2, TabICL")
# parser.add_argument("--train_gml", type=str, nargs="+", default=None)
# parser.add_argument("--val_gml",   type=str, nargs="+", default=None)
# parser.add_argument("--test_gml",  type=str, nargs="+", default=None)
# parser.add_argument("--config",    type=str, default=None)
# parser.add_argument("--use_partition_features", action="store_true")
# parser.add_argument("--use_graph_features",     action="store_true")
# parser.add_argument("--rf_n_estimators", type=int, default=200)
# parser.add_argument("--rf_max_depth",    type=int, default=None)
# parser.add_argument("--max_tabpfn",      type=int, default=1000,
#                     help="Max train samples for TabPFN (stratified subsample if exceeded)")
# parser.add_argument("--max_tabicl",      type=int, default=4_000,
#                     help="Max train samples for TabICL (stratified subsample if exceeded)")
# parser.add_argument("--skip_rf",     action="store_true")
# parser.add_argument("--skip_tabpfn", action="store_true")
# parser.add_argument("--skip_tabicl", action="store_true")
# args = parser.parse_args()

# # import os
# # import torch

# # num_cores = 20
# # torch.set_num_threads(num_cores)
# # torch.set_num_interop_threads(num_cores)

# # print(f"[INFO] Using {num_cores} CPU threads for PyTorch")

# if args.config:
#     with open(args.config) as f:
#         for k, v in yaml.safe_load(f).items():
#             setattr(args, k, v)

# for required_arg in ["train_gml", "val_gml", "test_gml"]:
#     if getattr(args, required_arg, None) is None:
#         parser.error(f"--{required_arg} is required (via CLI or config file)")

# # ── wandb ──────────────────────────────────────────────────────────────────────
# test_roots = [Path(p).stem for p in args.test_gml]
# test_name  = "+".join(test_roots[:2]) + ("+more" if len(test_roots) > 2 else "")
# wandb.init(project="gnn-parition-detection", name=f"tabular_baselines_for_{test_name}")
# wandb.config.update(vars(args))


# # ── helpers ────────────────────────────────────────────────────────────────────
# def pr_auc_from_probs(y_true, y_prob):
#     y_true = np.asarray(y_true).astype(int)
#     y_prob = np.asarray(y_prob).astype(float)
#     if y_true.min() == y_true.max():
#         return float("nan")
#     return float(average_precision_score(y_true, y_prob))


# def stratified_subsample(X, y, max_n, seed=42):
#     rng     = np.random.default_rng(seed)
#     pos_idx = np.where(y == 1)[0]
#     neg_idx = np.where(y == 0)[0]
#     n_pos   = min(len(pos_idx), max_n // 2)
#     n_neg   = min(len(neg_idx), max_n - n_pos)
#     idx     = np.concatenate([
#         rng.choice(pos_idx, n_pos, replace=False),
#         rng.choice(neg_idx, n_neg, replace=False),
#     ])
#     rng.shuffle(idx)
#     return X[idx], y[idx], n_pos, n_neg


# def eval_metrics(y_true, y_pred, y_prob):
#     return {
#         "f1":        f1_score(y_true, y_pred,        zero_division=0),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall":    recall_score(y_true, y_pred,    zero_division=0),
#         "pr_auc":    pr_auc_from_probs(y_true, y_prob),
#     }


# def load_single_gml(gml_path, use_partition_features=False, use_graph_features=False):
#     print(f"[INFO] Loading: {gml_path}")
#     G     = nx.read_gml(gml_path)
#     nodes = list(G.nodes())
#     features, labels = [], []

#     for node in nodes:
#         attr = G.nodes[node]
#         feat = list(attr.get("features", []))
#         if use_partition_features:
#             feat += list(attr.get("partition_features", [0.0, 0.0]))
#         if use_graph_features:
#             feat += list(attr.get("graph_features", [0.0, 0.0])[:2])
#         features.append(feat)

#         bv = attr.get("boundary", 0)
#         try:    label = int(bv)
#         except: label = 0
#         labels.append(label)

#     labels   = np.array(labels, dtype=np.int64)
#     labels[labels == -1] = 0
#     features = np.array(features, dtype=np.float32)

#     unique, counts = np.unique(labels, return_counts=True)
#     for u, c in zip(unique, counts):
#         print(f"  Class {u}: {c} samples")

#     node_map   = {n: i for i, n in enumerate(nodes)}
#     edges      = [(node_map[s], node_map[d]) for s, d in G.edges()]
#     edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
#                   if edges else torch.zeros((2, 0), dtype=torch.long))
#     num_nodes  = len(nodes)

#     return Data(
#         x          = torch.as_tensor(features, dtype=torch.float32),
#         edge_index = edge_index,
#         y          = torch.tensor(labels, dtype=torch.long),
#         train_mask = torch.ones(num_nodes,  dtype=torch.bool),
#         val_mask   = torch.zeros(num_nodes, dtype=torch.bool),
#         test_mask  = torch.zeros(num_nodes, dtype=torch.bool),
#     )


# def merge_data(a, b):
#     offset = a.num_nodes
#     return Data(
#         x          = torch.cat([a.x, b.x],                           dim=0),
#         edge_index = torch.cat([a.edge_index, b.edge_index + offset], dim=1),
#         y          = torch.cat([a.y, b.y],                           dim=0),
#         train_mask = torch.cat([a.train_mask, b.train_mask],         dim=0),
#         val_mask   = torch.cat([a.val_mask,   b.val_mask],           dim=0),
#         test_mask  = torch.cat([a.test_mask,  b.test_mask],          dim=0),
#     )


# # ── load & scale data ──────────────────────────────────────────────────────────
# print("\n[INFO] Loading TRAIN graphs")
# train_graphs = [load_single_gml(p, args.use_partition_features, args.use_graph_features)
#                 for p in args.train_gml]

# print("\n[INFO] Loading VAL graphs")
# val_graphs = [(p, load_single_gml(p, args.use_partition_features, args.use_graph_features))
#               for p in args.val_gml]

# print("\n[INFO] Loading TEST graphs")
# test_graphs = [(p, load_single_gml(p, args.use_partition_features, args.use_graph_features))
#                for p in args.test_gml]

# combined = reduce(merge_data, train_graphs)
# scaler   = StandardScaler()
# combined.x = torch.tensor(scaler.fit_transform(combined.x.numpy()), dtype=torch.float32)
# for g in train_graphs:          g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
# for _, g in val_graphs:         g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
# for _, g in test_graphs:        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)

# X_train = np.vstack([g.x.numpy() for g in train_graphs])
# y_train = np.concatenate([g.y.numpy() for g in train_graphs])
# print(f"\n[INFO] Train nodes: {len(X_train)}  pos={(y_train==1).sum()}  neg={(y_train==0).sum()}")


# # ── eval helper ────────────────────────────────────────────────────────────────
# def run_eval(tag, predict_fn, splits, wandb_prefix):
#     all_metrics = []
#     for path, g in splits:
#         name   = "/".join(Path(path).parts[-3:])
#         y_true = g.y.numpy()
#         y_prob = predict_fn(g.x.numpy())
#         y_pred = (y_prob >= 0.5).astype(int)
#         m      = eval_metrics(y_true, y_pred, y_prob)
#         all_metrics.append(m)
#         print(f"[{tag}][{wandb_prefix.upper()}] {name} | "
#               f"F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, PR-AUC={m['pr_auc']:.4f}")
#         wandb.log({f"{wandb_prefix}/{tag}/{Path(path).stem}/{k}": v for k, v in m.items()})

#     if all_metrics:
#         macro = {k: float(np.mean([d[k] for d in all_metrics])) for k in all_metrics[0]}
#         print(f"[{tag}][{wandb_prefix.upper()}] MACRO | "
#               f"F1={macro['f1']:.4f}, P={macro['precision']:.4f}, "
#               f"R={macro['recall']:.4f}, PR-AUC={macro['pr_auc']:.4f}")
#         wandb.log({f"{wandb_prefix}/{tag}/macro/{k}": v for k, v in macro.items()})


# # # ══════════════════════════════════════════════════════════════════════════════
# # # Random Forest
# # # ══════════════════════════════════════════════════════════════════════════════
# # if not args.skip_rf:
# #     print("\n" + "="*60 + "\n[INFO] Random Forest\n" + "="*60)
# #     rf = RandomForestClassifier(
# #         n_estimators=args.rf_n_estimators,
# #         max_depth=args.rf_max_depth,
# #         class_weight="balanced",
# #         n_jobs=10, random_state=42,
# #     )
# #     rf.fit(X_train, y_train)
# #     print("[INFO] RF training complete.")
# #     run_eval("RF", lambda X: rf.predict_proba(X)[:, 1], val_graphs,  "val")
# #     run_eval("RF", lambda X: rf.predict_proba(X)[:, 1], test_graphs, "test")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # TabPFN v2
# # # ══════════════════════════════════════════════════════════════════════════════
# import torch
# torch.set_num_threads(10)
# torch.set_num_interop_threads(10)

# from tqdm import tqdm
# import numpy as np

# def tabpfn_predict_with_progress(X, batch_size=256):
#     probs = []
#     for i in tqdm(range(0, len(X), batch_size)):
#         batch = X[i:i+batch_size]
#         batch_prob = tabpfn.predict_proba(batch)[:, 1]
#         probs.append(batch_prob)
#     return np.concatenate(probs)

# if not args.skip_tabpfn:
#     print("\n" + "="*60 + "\n[INFO] TabPFN v2\n" + "="*60)
#     try:
#         from tabpfn import TabPFNClassifier
#         X_tab, y_tab = X_train, y_train
#         if len(X_train) > args.max_tabpfn:
#             X_tab, y_tab, n_pos, n_neg = stratified_subsample(X_train, y_train, args.max_tabpfn)
#             print(f"[TabPFN] Subsampled {len(X_tab)} (pos={n_pos}, neg={n_neg}) from {len(X_train)}")
#         tabpfn = TabPFNClassifier(ignore_pretraining_limits=True)
#         tabpfn.fit(X_tab, y_tab)
#         print("[INFO] TabPFN training complete.")
#         run_eval("TabPFN", tabpfn_predict_with_progress, val_graphs, "val")
#         run_eval("TabPFN", tabpfn_predict_with_progress, test_graphs, "test")
#     except ImportError:
#         print("[TabPFN] Not installed — skip.  pip install tabpfn")
#     except Exception as e:
#         print(f"[TabPFN] Failed: {e}")


# # # ══════════════════════════════════════════════════════════════════════════════
# # # TabICL
# # # ══════════════════════════════════════════════════════════════════════════════
# # def predict_with_progress(model, X, batch_size=256, desc="Predicting"):
# #     import numpy as np
# #     probs = []
# #     for i in tqdm(range(0, len(X), batch_size), desc=desc):
# #         batch = X[i:i+batch_size]
# #         batch_prob = model.predict_proba(batch)[:, 1]
# #         probs.append(batch_prob)
# #     return np.concatenate(probs)
# # if not args.skip_tabicl:
# #     print("\n" + "="*60 + "\n[INFO] TabICL\n" + "="*60)
# #     try:
# #         from tabicl import TabICLClassifier
# #         X_icl, y_icl = X_train, y_train
# #         if len(X_train) > args.max_tabicl:
# #             X_icl, y_icl, n_pos, n_neg = stratified_subsample(X_train, y_train, args.max_tabicl)
# #             print(f"[TabICL] Subsampled {len(X_icl)} (pos={n_pos}, neg={n_neg}) from {len(X_train)}")
# #         tabicl = TabICLClassifier()
# #         tabicl.fit(X_icl, y_icl)
# #         print("[INFO] TabICL training complete.")
# #         run_eval(
# #             "TabICL",
# #             lambda X: predict_with_progress(tabicl, X, batch_size=128, desc="TabICL"),
# #             val_graphs,
# #             "val"
# #         )

# #         run_eval(
# #             "TabICL",
# #             lambda X: predict_with_progress(tabicl, X, batch_size=128, desc="TabICL"),
# #             test_graphs,
# #             "test"
# #         )

# #     except ImportError:
# #         print("[TabICL] Not installed — skip.  pip install tabicl")
# #     except Exception as e:
# #         print(f"[TabICL] Failed: {e}")


# wandb.finish()
# print("\n[INFO] Done.")