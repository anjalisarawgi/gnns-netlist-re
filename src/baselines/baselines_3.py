"""
train_baseline.py — Unified tabular baseline for boundary node classification.
Supports: Random Forest, TabICL, TabPFN (v1 or v2).

Usage:
    python train_baseline.py --config config/baseline_config.yml

    # or via CLI:
    python train_baseline.py --model rf      --train_gml ... --val_gml ... --test_gml ...
    python train_baseline.py --model tabicl  --train_gml ... --val_gml ... --test_gml ...
    python train_baseline.py --model tabpfn  --train_gml ... --val_gml ... --test_gml ...

Example YAML config:
    model: rf          # rf | tabicl | tabpfn
    train_gml:
      - path/to/train.gml
    val_gml:
      - path/to/val.gml
    test_gml:
      - path/to/test.gml

    # RF-specific
    n_estimators: 300
    top_k_features: 10

    # TabICL-specific
    n_d: 64
    n_steps: 5
    max_epochs: 200
    patience: 15

    # TabPFN-specific
    tabpfn_version: v1       # v1 | v2
    n_ensemble_configurations: 32
    subsample_train: 8000    # cap rows for TabPFN v1
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import networkx as nx
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(
        description="Unified tabular baseline: rf | tabicl | tabpfn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── routing ───────────────────────────────
    parser.add_argument("--model", type=str, default="rf",
                        choices=["rf", "tabicl", "tabpfn"],
                        help="Which model to run")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides CLI args)")

    # ── data ──────────────────────────────────
    parser.add_argument("--train_gml", type=str, nargs="+", default=None)
    parser.add_argument("--val_gml",   type=str, nargs="+", default=None)
    parser.add_argument("--test_gml",  type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--use_partition_features", action="store_true")
    parser.add_argument("--use_graph_features",     action="store_true")
    parser.add_argument("--save_predictions",       action="store_true",
                        help="Write per-node predictions back into GML files")
    parser.add_argument("--seed", type=int, default=42)

    # ── RF args ───────────────────────────────
    parser.add_argument("--n_estimators",   type=int,   default=200)
    parser.add_argument("--max_depth",      type=int,   default=None)
    parser.add_argument("--top_k_features", type=int,   default=10,
                        help="[RF] Also train a second RF on the top-k most important features")
    parser.add_argument("--n_jobs",         type=int,   default=10)

    # ── TabICL args ───────────────────────────
    parser.add_argument("--tabicl_n_estimators", type=int, default=8,
                        help="[TabICL] number of ensemble members")
    parser.add_argument("--tabicl_batch_size",   type=int, default=8,
                        help="[TabICL] ensemble members processed together (reduce RAM)")
    parser.add_argument("--tabicl_device",       type=str, default="cpu",
                        help="[TabICL] cpu or cuda")

    # ── TabPFN args ───────────────────────────
    parser.add_argument("--tabpfn_version",             type=str, default="v1",
                        choices=["v1", "v2"])
    parser.add_argument("--n_ensemble_configurations",  type=int,   default=32,
                        help="[TabPFN v1] ensemble size")
    parser.add_argument("--device",                     type=str,   default="cpu",
                        help="[TabPFN v1] cpu or cuda")
    parser.add_argument("--time_budget",                type=int,   default=120,
                        help="[TabPFN v2] optimisation time budget in seconds")
    parser.add_argument("--preset",                     type=str,   default="default",
                        choices=["default", "interpretable", "avoid_overfitting"],
                        help="[TabPFN v2] AutoTabPFN preset")
    parser.add_argument("--subsample_train",            type=int,   default=None,
                        help="[TabPFN] stratified subsample cap (v1 max ~10k rows)")

    args = parser.parse_args()

    # ── apply YAML config ─────────────────────
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            setattr(args, key, value)

    # ── validate required data args ───────────
    missing = [f"--{k}" for k in ("train_gml", "val_gml", "test_gml")
               if not getattr(args, k, None)]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    return args


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────
# Scaler (skips one-hot prefix)
# ──────────────────────────────────────────────
NUM_CATEGORICAL = 14

class SelectiveScaler:
    def __init__(self, n_categorical: int = NUM_CATEGORICAL):
        self.n_cat = n_categorical
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.n_cat:] = self.scaler.fit_transform(X[:, self.n_cat:])
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.n_cat:] = self.scaler.transform(X[:, self.n_cat:])
        return X


# ──────────────────────────────────────────────
# GML loader
# ──────────────────────────────────────────────
def load_gml(gml_path: str, args) -> tuple[np.ndarray, np.ndarray]:
    G = nx.read_gml(gml_path)
    features, labels = [], []
    for node in G.nodes():
        attr = G.nodes[node]
        feat = list(attr.get("features", []))
        if args.use_partition_features:
            feat += list(attr.get("partition_features", [0.0, 0.0]))
        if args.use_graph_features:
            feat += list(attr.get("graph_features", [0.0, 0.0])[:2])
        features.append(feat)
        boundary_val = attr.get("boundary", 0)
        try:
            label = int(boundary_val)
        except (ValueError, TypeError):
            label = 0
        labels.append(max(label, 0))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if y_true.min() != y_true.max() else float("nan")
    )
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    return {
        "f1":             float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":      float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc":         pr_auc,
        "accuracy":       float((y_true == y_pred).mean()),
        "boundary_1_acc": float((y_pred[pos_mask] == 1).mean()) if pos_mask.any() else 0.0,
        "boundary_0_acc": float((y_pred[neg_mask] == 0).mean()) if neg_mask.any() else 0.0,
    }

def print_metrics(tag: str, name: str, m: dict):
    print(
        f"[{tag}] {name} | "
        f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
        f"PR-AUC={m['pr_auc']:.4f}  Acc={m['accuracy']:.4f}  "
        f"b1={m['boundary_1_acc']:.4f}  b0={m['boundary_0_acc']:.4f}"
    )

def macro_avg(metrics_list: list[dict]) -> dict:
    keys = metrics_list[0].keys()
    return {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}


# ──────────────────────────────────────────────
# Save predictions back to GML
# ──────────────────────────────────────────────
def save_preds_to_gml(gml_path, y_pred, y_prob, output_path):
    G = nx.read_gml(gml_path)
    id2label = {0: "not_boundary", 1: "boundary"}
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["predicted_label"] = id2label[int(y_pred[i])]
        G.nodes[node]["predicted_prob"]  = float(y_prob[i])
    nx.write_gml(G, output_path)
    print(f"  Saved predictions → {output_path}")


# ──────────────────────────────────────────────
# Stratified subsample (for TabPFN)
# ──────────────────────────────────────────────
def stratified_subsample(X, y, n, seed):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    indices = []
    for cls, frac in zip(classes, fracs):
        cls_idx = np.where(y == cls)[0]
        k = min(max(1, int(round(n * frac))), len(cls_idx))
        indices.extend(rng.choice(cls_idx, k, replace=False).tolist())
    rng.shuffle(indices)
    return X[indices], y[indices]


# ──────────────────────────────────────────────
# Evaluate a fitted clf on a list of graphs
# ──────────────────────────────────────────────
def evaluate_all(clf, data_list, tag, args, model_dir, feature_ids=None, show_progress=False):
    """
    clf        : fitted sklearn-compatible classifier
    data_list  : list of (path, X, y)
    feature_ids: optional index array to select columns (for RF top-k)
    show_progress: show per-row tqdm bar during predict (for slow models)
    """
    metrics_list = []
    graph_iter = tqdm(data_list, desc=f"{tag}", unit="graph", ncols=80)
    for p, X, y in graph_iter:
        name = Path(p).stem
        X_in = X[:, feature_ids] if feature_ids is not None else X
        graph_iter.set_postfix({"graph": name, "rows": len(y)})

        if show_progress:
            # run predict in chunks so we can show a row-level progress bar
            # chunk = max(1, len(X_in) // 100)   # 100 updates across the graph
            chunk = 2000
            preds, probs = [], []
            row_iter = tqdm(range(0, len(X_in), chunk),
                            desc=f"  predicting {name}", unit="chunk",
                            leave=False, ncols=80)
            for start in row_iter:
                end = min(start + chunk, len(X_in))
                batch = X_in[start:end]
                preds.append(clf.predict(batch))
                probs.append(clf.predict_proba(batch)[:, 1])
            y_pred = np.concatenate(preds)
            y_prob = np.concatenate(probs)
        else:
            y_pred = clf.predict(X_in)
            y_prob = clf.predict_proba(X_in)[:, 1]

        m = compute_metrics(y, y_pred, y_prob)
        print_metrics(tag, name, m)
        metrics_list.append(m)

        if args.save_predictions and "TEST" in tag:
            out = os.path.join(model_dir, f"{name}_{tag.lower().replace(' ', '_')}_preds.gml")
            save_preds_to_gml(p, y_pred, y_prob, out)

    print_metrics(f"{tag} MACRO", "ALL", macro_avg(metrics_list))
    return metrics_list


# ══════════════════════════════════════════════
# MODEL RUNNERS
# ══════════════════════════════════════════════

# ── Random Forest ─────────────────────────────
def run_rf(X_train, y_train, val_scaled, test_scaled, args, model_dir):
    print("\n[RF] Training (all features) …")
    t0 = time.perf_counter()
    rf_all = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    rf_all.fit(X_train, y_train)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    val_metrics_all  = evaluate_all(rf_all, val_scaled,  "RF-ALL VAL",  args, model_dir)
    test_metrics_all = evaluate_all(rf_all, test_scaled, "RF-ALL TEST", args, model_dir)

    # ── Feature importance → top-k RF ─────────
    importances = rf_all.feature_importances_
    feat_df = (
        pd.DataFrame({"feature_id": np.arange(len(importances)), "importance": importances})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )
    top_k   = args.top_k_features
    top_ids = feat_df["feature_id"].values[:top_k]
    print(f"\n[RF] Top-{top_k} feature indices: {top_ids.tolist()}")
    print(feat_df.head(20).to_string(index=False))

    print(f"\n[RF] Training (top-{top_k} features) …")
    t0 = time.perf_counter()
    rf_topk = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    rf_topk.fit(X_train[:, top_ids], y_train)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    val_metrics_topk  = evaluate_all(rf_topk, val_scaled,  f"RF-TOP{top_k} VAL",  args, model_dir, top_ids)
    test_metrics_topk = evaluate_all(rf_topk, test_scaled, f"RF-TOP{top_k} TEST", args, model_dir, top_ids)

    # ── Save ──────────────────────────────────
    joblib.dump(rf_all,  os.path.join(model_dir, "rf_all.joblib"))
    joblib.dump(rf_topk, os.path.join(model_dir, f"rf_top{top_k}.joblib"))
    feat_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    np.save(os.path.join(model_dir, "top_feature_ids.npy"), top_ids)

    return {
        "val_macro_all":          macro_avg(val_metrics_all),
        "test_macro_all":         macro_avg(test_metrics_all),
        f"val_macro_top{top_k}":  macro_avg(val_metrics_topk),
        f"test_macro_top{top_k}": macro_avg(test_metrics_topk),
    }


# ── TabICL ────────────────────────────────────
def run_tabicl(X_train, y_train, val_scaled, test_scaled, args, model_dir):
    if len(X_train) > 10000:
        print(f"[TabICL] Subsampling train rows: {len(X_train)} → 5000")
        X_train, y_train = stratified_subsample(X_train, y_train, 5000, args.seed)

    try:
        from tabicl import TabICLClassifier
    except ImportError:
        print("[ERROR] tabicl not installed.  pip install tabicl")
        sys.exit(1)

    print("\n[TabICL] Building classifier …")
    clf = TabICLClassifier(
        n_estimators=args.tabicl_n_estimators,
        batch_size=args.tabicl_batch_size,
        device=args.tabicl_device,
        random_state=args.seed,
    )

    # fit is cheap — just stores the training data and builds KV cache
    print(f"[TabICL] Fitting (caching {len(X_train)} training rows) …")
    t0 = time.perf_counter()
    with tqdm(total=1, desc="  fit tabicl", unit="step", ncols=80) as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)
    print(f"  Fit time: {time.perf_counter() - t0:.1f}s  (heavy compute happens at predict time)")

    val_metrics  = evaluate_all(clf, val_scaled,  "TABICL VAL",  args, model_dir, show_progress=True)
    test_metrics = evaluate_all(clf, test_scaled, "TABICL TEST", args, model_dir, show_progress=True)

    # ── Save ──────────────────────────────────
    joblib.dump(clf, os.path.join(model_dir, "tabicl_model.joblib"))

    return {
        "val_macro":  macro_avg(val_metrics),
        "test_macro": macro_avg(test_metrics),
    }


# ── TabPFN ────────────────────────────────────
def run_tabpfn(X_train, y_train, val_scaled, test_scaled, args, model_dir):
    version = args.tabpfn_version

    if version == "v1":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            print("[ERROR] TabPFN not installed.  pip install tabpfn")
            sys.exit(1)
        # newer tabpfn (>=2.0) renamed N_ensemble_configurations -> n_estimators
        import inspect
        tabpfn_params = inspect.signature(TabPFNClassifier.__init__).parameters
        if "n_estimators" in tabpfn_params:
            clf = TabPFNClassifier(
                device=args.device,
                n_estimators=args.n_ensemble_configurations,
            )
        else:
            clf = TabPFNClassifier(
                device=args.device,
                N_ensemble_configurations=args.n_ensemble_configurations,
            )
        print(f"[TabPFN-v1] ensemble={args.n_ensemble_configurations}  device={args.device}")
    else:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        # if your installed version supports passing device here, use it;
        # otherwise keep default or check signature first
        print(f"[TabPFN-v2] using official TabPFNClassifier V2")
        
    # subsample if requested (v1 is limited to ~10k rows)
    if args.subsample_train and len(X_train) > args.subsample_train:
        print(f"[TabPFN] Subsampling: {len(X_train)} → {args.subsample_train} rows …")
        with tqdm(total=1, desc="  subsampling", unit="step", ncols=80) as pbar:
            X_train, y_train = stratified_subsample(X_train, y_train, args.subsample_train, args.seed)
            pbar.update(1)
        print(f"  After subsample: {Counter(y_train.tolist())}")

    if version == "v1" and len(X_train) > 10_000:
        print(f"[WARN] TabPFN v1 is designed for ≤10k rows. You have {len(X_train)}. "
              f"Consider --subsample_train 8000.")

    print(f"\n[TabPFN-{version}] Fitting (storing {len(X_train)} training rows) …")
    t0 = time.perf_counter()
    with tqdm(total=1, desc=f"  fit tabpfn-{version}", unit="step", ncols=80) as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)
    print(f"  Fit time: {time.perf_counter() - t0:.1f}s")

    val_metrics  = evaluate_all(clf, val_scaled,  f"TABPFN-{version.upper()} VAL",  args, model_dir, show_progress=True)
    test_metrics = evaluate_all(clf, test_scaled, f"TABPFN-{version.upper()} TEST", args, model_dir, show_progress=True)

    # ── Save ──────────────────────────────────
    joblib.dump(clf, os.path.join(model_dir, f"tabpfn_{version}.joblib"))

    return {
        "val_macro":  macro_avg(val_metrics),
        "test_macro": macro_avg(test_metrics),
    }


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    args = get_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[INFO] Model: {args.model.upper()}")
    print(f"[INFO] Output dir: {model_dir}")

    # ── Load data ──────────────────────────────
    print("\n[INFO] Loading training graphs …")
    X_parts, y_parts = [], []
    for p in args.train_gml:
        X, y = load_gml(p, args)
        print(f"  {p}  nodes={len(y)}  counts={Counter(y.tolist())}")
        X_parts.append(X)
        y_parts.append(y)
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    print(f"  → Combined: {X_train.shape}  boundary_ratio={y_train.mean():.4f}")

    print("\n[INFO] Loading validation graphs …")
    val_data = []
    for p in args.val_gml:
        X, y = load_gml(p, args)
        print(f"  {p}  nodes={len(y)}  counts={Counter(y.tolist())}")
        val_data.append((p, X, y))

    print("\n[INFO] Loading test graphs …")
    test_data = []
    for p in args.test_gml:
        X, y = load_gml(p, args)
        print(f"  {p}  nodes={len(y)}  counts={Counter(y.tolist())}")
        test_data.append((p, X, y))

    # ── Scale (fit on train, apply to val+test) ─
    scaler = SelectiveScaler(NUM_CATEGORICAL)
    X_train_s    = scaler.fit_transform(X_train)
    val_scaled   = [(p, scaler.transform(X), y) for p, X, y in val_data]
    test_scaled  = [(p, scaler.transform(X), y) for p, X, y in test_data]
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))

    # ── Dispatch ───────────────────────────────
    if args.model == "rf":
        result = run_rf(X_train_s, y_train, val_scaled, test_scaled, args, model_dir)
    elif args.model == "tabicl":
        result = run_tabicl(X_train_s, y_train, val_scaled, test_scaled, args, model_dir)
    elif args.model == "tabpfn":
        result = run_tabpfn(X_train_s, y_train, val_scaled, test_scaled, args, model_dir)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ── Save metadata ─────────────────────────
    metadata = {
        "model":      args.model,
        "timestamp":  timestamp,
        "train_gml":  args.train_gml,
        "val_gml":    args.val_gml,
        "test_gml":   args.test_gml,
        **result,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[INFO] Saved everything to: {model_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()