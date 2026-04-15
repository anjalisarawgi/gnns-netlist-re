import argparse
import os
import sys
import json
import random
import inspect
import time
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabicl import TabICLClassifier
from sklearn.preprocessing import StandardScaler



def parse_args():
    p = argparse.ArgumentParser(description="Baseline models for boundary node detection")

    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default="rf", choices=["rf", "tabpfn", "tabicl"])
    p.add_argument("--train_gml", nargs="+", default=None)
    p.add_argument("--val_gml",   nargs="+", default=None)
    p.add_argument("--test_gml",  nargs="+", default=None)

    # RF 
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--n_estimators", type=int, default=100)

    # TabICL
    p.add_argument("--tabicl_n_estimators", type=int, default=8, help="Number of ensemble estimators for TabICL")
    p.add_argument("--tabicl_batch_size", type=int, default=8, help="Batch size for TabICL inference")

    # feature flags
    p.add_argument("--use_partition_features", action="store_true", default=False)
    p.add_argument("--use_graph_features",     action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Override args with config values (only if not already set via CLI)
        for key, val in cfg.items():
            if val is not None:
                setattr(args, key, val)
    return args



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

NUM_CATEGORICAL = 14
class SelectiveScaler:
    def __init__(self, n_categorical):
        self.n_cat = n_categorical
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        X = X.copy()
        if self.n_cat < X.shape[1]:  # only scale if continuous cols exist
            X[:, self.n_cat:] = self.scaler.fit_transform(X[:, self.n_cat:])
        return X

    def transform(self, X):
        X = X.copy()
        if self.n_cat < X.shape[1]:
            X[:, self.n_cat:] = self.scaler.transform(X[:, self.n_cat:])
        return X

        
def load_graph_features(gml_path: str, args) -> tuple[np.ndarray, np.ndarray]:
    G = nx.read_gml(gml_path)
    nodes = list(G.nodes())

    features, labels = [], []
    skipped_no_label = 0 
    for node in nodes:
        attr = G.nodes[node]

        # skip oded which have no boundary label 
        if "boundary" not in attr:
            skipped_no_label +=1 
            continue

        feat = list(attr.get("features", []))
        feat = feat[:14] # skipping distance io feature (feature 43)
        # if args.use_partition_features:
        #     partition_feat = attr.get("partition_features", [0.0, 0.0, 0.0])
        #     partition_feat = partition_feat[2:3]
        #     feat = list(feat) + list(partition_feat)

        # if args.use_graph_features:
        #     graph_feat = attr.get("graph_features", [0.0, 0.0])
        #     graph_feat_subset = graph_feat[:2]
        #     feat = list(feat) + list(graph_feat_subset)

        features.append(feat)

        boundary_value = attr.get("boundary", 0)
        try:
            label = int(boundary_value)
        except (ValueError, TypeError):
            label = 0
        labels.append(label)

    if skipped_no_label > 0:
        print(f"[WARN] skipped {skipped_no_label} nodes because they did not have boundary labels")
    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int64)
    # y[y == -1] = 0   # can be removed i think but kept as safety check

    print(f"  [{Path(gml_path).name}]  nodes={len(y)}  "
          f"boundary={y.sum()}  non-boundary={(y == 0).sum()}")
    return X, y


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def evaluate(name: str, tag: str, y_true, y_pred, y_prob):
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = pr_auc(y_true, y_prob)
    print(f"[{tag}] {name} | F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  PR-AUC={auc:.4f}")
    return {"f1": f1, "precision": pre, "recall": rec, "pr_auc": auc}


def macro_avg(results: list[dict]) -> dict:
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}



ID2LABEL = {0: "not_boundary", 1: "boundary"}
def evaluate_splits(clf, tag, val_data, test_data, args, feature_mask=None):
    def _slice(X):
        return X[:, feature_mask] if feature_mask is not None else X

    # val
    # val_results = []
    # for name, path, X_val, y_val in val_data:
    #     y_pred = clf.predict(_slice(X_val))
    #     y_prob = clf.predict_proba(_slice(X_val))[:, 1]
    #     val_results.append(evaluate(name, f"{tag} VAL", y_val, y_pred, y_prob))
    # print(f"  VAL MACRO: {macro_avg(val_results)}")

    # test
    print(f"\ntest - results")
    test_results = []
    for name, path, X_test, y_test in test_data:
        y_pred = clf.predict(_slice(X_test))
        y_prob = clf.predict_proba(_slice(X_test))[:, 1]
        test_results.append(evaluate(name, f"{tag} TEST", y_test, y_pred, y_prob))
    print(f" MACRO (TEST): {macro_avg(test_results)}")

    return test_results

# all models
def run_tabpfn(X_train, y_train, val_data, test_data, args):
    print("[TabPFN] using TabPFN v2.5")

    # subsampling
    print(f"  total rows: {len(X_train)} before subsamples …")
    if len(X_train) > 50_000:
        print(f"  Subsampling train rows: {len(X_train)} → {50_000}")
        X_train, y_train = stratified_subsample(X_train, y_train, 50_000, args.seed)
    print(f"  total rows: {len(X_train)} after subsamples …")

    classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device = "cuda")
    classifier.fit(X_train, y_train)
    print("  Training complete.")

    test_results = evaluate_splits(classifier, "TabPFN", val_data, test_data, args)
    return {
        # "val_macro":  macro_avg(val_results),
        "test_macro": macro_avg(test_results),
    }

# subsampling ad maintining the ratios
def stratified_subsample(X, y, max_n, seed):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    ratios = counts / counts.sum()
    per_class = np.maximum((ratios * max_n).astype(int), 1)

    diff = max_n - per_class.sum()
    if diff > 0:
        per_class[np.argmax(counts)] += diff
    elif diff < 0:
        per_class[np.argmax(per_class)] += diff  # diff is negative

    indices = []
    for cls, n in zip(classes, per_class):
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def run_tabicl(X_train, y_train, val_data, test_data, args):
    print("tabicl:")
    print(f"  total rows: {len(X_train)} before subsamples …")
    if len(X_train) > 200_000:
        print(f"  Subsampling train rows: {len(X_train)} → {200_000}")
        X_train, y_train = stratified_subsample(X_train, y_train, 200_000, args.seed)
    print(f"  total rows: {len(X_train)} after subsamples …")
    

    classifier = TabICLClassifier( 
        n_estimators=args.tabicl_n_estimators,
        batch_size=args.tabicl_batch_size,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device="cuda",
        random_state=42,
    )
    classifier.fit(X_train, y_train)
    print("[TABICL] tabical fitted on training data")
    
    test_results = evaluate_splits( classifier, "TabICL", val_data, test_data, args)

    return {
        # "val_macro":  macro_avg(val_results),
        "test_macro": macro_avg(test_results),
    }

def run_rf(X_train, y_train, val_data, test_data, args):
    # on all features
    rf_all = RandomForestClassifier( n_estimators=args.n_estimators, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    rf_all.fit(X_train, y_train)
    print("rf training done complete")
    test_results_all = evaluate_splits(rf_all, "RF-ALL", val_data, test_data, args)

    # # taking top k 
    # importances = rf_all.feature_importances_
    # feat_df = (
    #     pd.DataFrame({"feature_id": np.arange(len(importances)), "importance": importances})
    #     .sort_values("importance", ascending=False)
    #     .reset_index(drop=True)
    # )
    # print(f"top {args.top_k} feature importances")
    # print(feat_df.head(args.top_k).to_string(index=False))
    # top_features = feat_df["feature_id"].values[: args.top_k]

    # rf now on all the seelcted features only 
    # print(f"rf on the top k")
    # rf_sel = RandomForestClassifier( n_estimators=args.n_estimators, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=args.seed)
    # rf_sel.fit(X_train[:, top_features], y_train)
    # print("Training complete.")
    # test_results_sel = evaluate_splits( rf_sel, f"RF-TOP{args.top_k}", val_data, test_data, args, feature_mask=top_features)

    return {
        # "val_macro_all":      macro_avg(val_results_all),
        # "val_macro_selected": macro_avg(val_results_sel),
        "test_macro_all":     macro_avg(test_results_all),
        # "test_macro_selected":macro_avg(test_results_sel),
        # "top_features":       top_features.tolist(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    print("train graphs:::")
    X_parts, y_parts = [], []
    for path in args.train_gml:
        X, y = load_graph_features(path, args)
        X_parts.append(X)
        y_parts.append(y)
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    print(f"  Total train  nodes={len(y_train)}  boundary={y_train.sum()}")

    print("validation graphs:::")
    val_data = []
    for path in args.val_gml:
        X, y = load_graph_features(path, args)
        val_data.append((Path(path).stem, path, X, y))

    print("test graphs:::")
    test_data = []
    for path in args.test_gml:
        X, y = load_graph_features(path, args)
        test_data.append((Path(path).stem, path, X, y))
   
    # scaling - for tabpfn and tabicl 
    scaler = SelectiveScaler(NUM_CATEGORICAL)
    X_train_scaled = scaler.fit_transform(X_train)
    # scaler fit on the trian and tested on val and test respectively
    val_data_scaled = [ (name, path, scaler.transform(X), y)for name, path, X, y in val_data]
    test_data_scaled = [(name, path, scaler.transform(X), y) for name, path, X, y in test_data]

    if args.mode == "rf":
        results_summary = run_rf(X_train, y_train, val_data, test_data, args)
    elif args.mode == "tabpfn":
        # results_summary = run_tabpfn(X_train, y_train, val_data, test_data, args)
        results_summary = run_tabpfn(X_train_scaled, y_train, val_data_scaled, test_data_scaled, args)
    elif args.mode == "tabicl":
        results_summary = run_tabicl(X_train_scaled, y_train, val_data_scaled, test_data_scaled, args)

    print("completed")


if __name__ == "__main__":
    main()