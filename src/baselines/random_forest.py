"""
Example YAML (configs/my_experiment.yaml):
    train_gml:
      - data/design_a/lib1/graph.gml
      - data/design_b/lib1/graph.gml
    val_gml:
      - data/design_c/lib1/graph.gml
    test_gml:
      - data/design_d/lib1/graph.gml
    top_k: 15
    n_estimators: 300
    use_partition_features: true
    save_models: true
    save_predictions: true
    output_dir: results/rf/my_experiment
"""

import argparse
import os
import json
import random
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


# ---------------------------------------------------------------------------
# CLI  +  YAML config  (same pattern as train.py)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random Forest baseline for boundary node detection")

    # config file (optional — overrides all defaults below, then CLI overrides config)
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file. CLI flags take precedence over YAML values.")

    # data
    p.add_argument("--train_gml", nargs="+", default=None)
    p.add_argument("--val_gml",   nargs="+", default=None)
    p.add_argument("--test_gml",  nargs="+", default=None)

    # RF hyper-parameters
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of top features to use for the second (selected) RF")
    p.add_argument("--n_estimators", type=int, default=200)

    # feature flags
    p.add_argument("--use_partition_features", action="store_true", default=False)
    p.add_argument("--use_graph_features",     action="store_true", default=False)

    # output / persistence
    p.add_argument("--save_models",      action="store_true", default=False,
                   help="Persist trained RF models to disk via joblib")
    p.add_argument("--save_predictions", action="store_true", default=False,
                   help="Write per-graph prediction GML files")
    p.add_argument("--output_dir", type=str, default="results/rf",
                   help="Where to write prediction GMLs / model files")

    # misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # ---- YAML loading (mirrors train.py) ----
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Re-parse so that explicit CLI flags still win over YAML values.
        # Strategy: set YAML values as new defaults, then re-parse.
        # For store_true booleans we need special handling.
        bool_flags = {"use_partition_features", "use_graph_features",
                      "save_models", "save_predictions"}

        for key, value in cfg.items():
            if key == "config":
                continue
            if key in bool_flags:
                # Only apply YAML if the flag was NOT explicitly set on CLI
                # (argparse doesn't expose this cleanly, so we check current value)
                if not getattr(args, key, False):
                    setattr(args, key, bool(value))
            else:
                # For non-boolean args: YAML wins unless CLI supplied a non-default value.
                # Simplest safe approach: YAML always sets, CLI parse below overrides.
                setattr(args, key, value)

        # Re-parse CLI on top so explicit flags beat the YAML values just applied.
        # We only override if the user actually passed the flag on the command line.
        cli_args = p.parse_args()  # fresh parse (YAML not involved)
        import sys
        raw_cli = set()
        for token in sys.argv[1:]:
            if token.startswith("--"):
                raw_cli.add(token.lstrip("-").split("=")[0].replace("-", "_"))

        for key in raw_cli:
            if key == "config":
                continue
            if hasattr(cli_args, key):
                setattr(args, key, getattr(cli_args, key))

        print(f"[CONFIG] Loaded from: {args.config}")

    # ---- Validate required fields ----
    missing = [f for f in ("train_gml", "val_gml", "test_gml") if not getattr(args, f)]
    if missing:
        p.error(
            f"The following required arguments are missing (set via CLI or config): "
            f"{', '.join('--' + m for m in missing)}"
        )

    return args


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Feature extraction  (mirrors load_single_gml in train.py)
# ---------------------------------------------------------------------------

def load_graph_features(gml_path: str, args) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays for one GML file."""
    G = nx.read_gml(gml_path)
    nodes = list(G.nodes())

    features, labels = [], []

    for node in nodes:
        attr = G.nodes[node]

        feat = list(attr.get("features", []))
        # feat = feat[:31]            # (14 OHE) + (14 OHE) + indeg, outdeg, ratio

        if args.use_partition_features:
            feat += list(attr.get("partition_features", [0.0, 0.0, 0.0]))

        if args.use_graph_features:
            graph_feat = attr.get("graph_features", [0.0, 0.0])
            feat += list(graph_feat[:2])
        
        # if args.use_unsupervised_features:
        #     f1 = float(attr.get("unsup_louvain_1hop", 0.0))
        #     f2 = float(attr.get("unsup_louvain_2hop", 0.0))
        #     feat = list(feat) + [f1, f2]

        # if args.use_unsupervised_features_louvian:
        #     f1 = float(attr.get("unsup_leiden_1hop", 0.0))
        #     f2 = float(attr.get("unsup_leiden_2hop", 0.0))
        #     feat = list(feat) + [f1, f2]


        features.append(feat)

        boundary_value = attr.get("boundary", 1)
        try:
            label = int(boundary_value)
        except (ValueError, TypeError):
            label = 1
        labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int64)
    y[y == -1] = 1          # treat -1 as non-boundary (matches train.py)

    print(f"  [{Path(gml_path).name}]  nodes={len(y)}  "
          f"boundary={y.sum()}  non-boundary={(y == 0).sum()}")
    return X, y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prediction saving
# ---------------------------------------------------------------------------

ID2LABEL = {0: "not_boundary", 1: "boundary"}


def save_preds_to_gml(gml_path: str, y_pred, y_prob, output_path: str):
    G = nx.read_gml(gml_path)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["predicted_label"] = ID2LABEL[int(y_pred[i])]
        G.nodes[node]["predicted_prob"]  = float(y_prob[i])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_gml(G, output_path)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== RF Experiment Config ===")
    print(f"  config file       : {args.config or '(none)'}")
    print(f"  train graphs      : {args.train_gml}")
    print(f"  val graphs        : {args.val_gml}")
    print(f"  test graphs       : {args.test_gml}")
    print(f"  n_estimators      : {args.n_estimators}")
    print(f"  top_k             : {args.top_k}")
    print(f"  partition features: {args.use_partition_features}")
    print(f"  graph features    : {args.use_graph_features}")
    print(f"  output_dir        : {args.output_dir}")
    print(f"  seed              : {args.seed}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n=== Loading TRAIN graphs ===")
    X_parts, y_parts = [], []
    for path in args.train_gml:
        X, y = load_graph_features(path, args)
        X_parts.append(X)
        y_parts.append(y)
    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    print(f"  Total train  nodes={len(y_train)}  boundary={y_train.sum()}")

    print("\n=== Loading VAL graphs ===")
    val_data = []           # list of (name, path, X, y)
    for path in args.val_gml:
        X, y = load_graph_features(path, args)
        val_data.append((Path(path).stem, path, X, y))

    print("\n=== Loading TEST graphs ===")
    test_data = []
    for path in args.test_gml:
        X, y = load_graph_features(path, args)
        test_data.append((Path(path).stem, path, X, y))

    # ------------------------------------------------------------------
    # 2. RF — all features
    # ------------------------------------------------------------------
    print("\n=== Training RF (all features) ===")
    rf_all = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    rf_all.fit(X_train, y_train)
    print("  Training complete.")

    print("\n--- RF-ALL  |  VAL ---")
    val_results_all = []
    for name, path, X_val, y_val in val_data:
        y_pred = rf_all.predict(X_val)
        y_prob = rf_all.predict_proba(X_val)[:, 1]
        val_results_all.append(evaluate(name, "RF-ALL VAL", y_val, y_pred, y_prob))
        # if args.save_predictions:
            # out = os.path.join(args.output_dir, f"{name}_rf_all.gml")
            # save_preds_to_gml(path, y_pred, y_prob, out)

    print(f"  VAL MACRO: {macro_avg(val_results_all)}")

    # ------------------------------------------------------------------
    # 3. Feature importance → select top-k
    # ------------------------------------------------------------------
    importances = rf_all.feature_importances_
    feat_df = (
        pd.DataFrame({"feature_id": np.arange(len(importances)), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n=== Top-{args.top_k} feature importances ===")
    print(feat_df.head(args.top_k).to_string(index=False))

    top_features = feat_df["feature_id"].values[: args.top_k]

    # ------------------------------------------------------------------
    # 4. RF — selected features
    # ------------------------------------------------------------------
    print(f"\n=== Training RF (top-{args.top_k} features) ===")
    rf_sel = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    rf_sel.fit(X_train[:, top_features], y_train)
    print("  Training complete.")

    print(f"\n--- RF-TOP{args.top_k}  |  VAL ---")
    val_results_sel = []
    for name, path, X_val, y_val in val_data:
        y_pred = rf_sel.predict(X_val[:, top_features])
        y_prob = rf_sel.predict_proba(X_val[:, top_features])[:, 1]
        val_results_sel.append(evaluate(name, f"RF-TOP{args.top_k} VAL", y_val, y_pred, y_prob))
        if args.save_predictions:
            out = os.path.join(args.output_dir, f"{name}_rf_top{args.top_k}.gml")
            save_preds_to_gml(path, y_pred, y_prob, out)

    print(f"  VAL MACRO: {macro_avg(val_results_sel)}")

    # ------------------------------------------------------------------
    # 5. Test evaluation  (both models)
    # ------------------------------------------------------------------
    print("\n=== TEST evaluation ===")
    test_results_all, test_results_sel = [], []

    for name, path, X_test, y_test in test_data:
        y_pred_all  = rf_all.predict(X_test)
        y_prob_all  = rf_all.predict_proba(X_test)[:, 1]
        test_results_all.append(evaluate(name, "RF-ALL TEST", y_test, y_pred_all, y_prob_all))

        y_pred_sel = rf_sel.predict(X_test[:, top_features])
        y_prob_sel = rf_sel.predict_proba(X_test[:, top_features])[:, 1]
        test_results_sel.append(evaluate(name, f"RF-TOP{args.top_k} TEST", y_test, y_pred_sel, y_prob_sel))

        # if args.save_predictions:
        #     save_preds_to_gml(path, y_pred_all, y_prob_all,
        #                       os.path.join(args.output_dir, f"{name}_rf_all_test.gml"))
        #     save_preds_to_gml(path, y_pred_sel, y_prob_sel,
        #                       os.path.join(args.output_dir, f"{name}_rf_top{args.top_k}_test.gml"))

    print(f"\n  TEST MACRO (all features): {macro_avg(test_results_all)}")
    print(f"  TEST MACRO (top-{args.top_k}):       {macro_avg(test_results_sel)}")

    # ------------------------------------------------------------------
    # 6. Persist artefacts
    # ------------------------------------------------------------------
    feat_imp_path = os.path.join(args.output_dir, "rf_feature_importance.csv")
    feat_df.to_csv(feat_imp_path, index=False)
    print(f"\n  Feature importance table → {feat_imp_path}")

    results_summary = {
        "val_macro_all":      macro_avg(val_results_all),
        "val_macro_selected": macro_avg(val_results_sel),
        "test_macro_all":     macro_avg(test_results_all),
        "test_macro_selected":macro_avg(test_results_sel),
        "top_features":       top_features.tolist(),
    }
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Results summary          → {summary_path}")

    # if args.save_models:
    #     joblib.dump(rf_all, os.path.join(args.output_dir, "rf_all.joblib"))
    #     joblib.dump(rf_sel, os.path.join(args.output_dir, f"rf_top{args.top_k}.joblib"))
    #     print(f"  Models saved to {args.output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()