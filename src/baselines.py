"""
run_baselines.py
================
Standalone baseline script for boundary node detection.
Supports YAML config (same config as your main training script).

Usage:

  # Using your existing config (recommended):
  python run_baselines.py --config configs/your_config.yaml

  # Heuristics only (no training data needed):
  python run_baselines.py --config configs/your_config.yaml --heuristics_only

  # Override test graphs from config:
  python run_baselines.py --config configs/your_config.yaml --test_gml data/mips.gml

  # With partition features:
  python run_baselines.py --config configs/your_config.yaml --use_partition_features
"""

import argparse
import os
import numpy as np
import networkx as nx
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import csv
import warnings
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

# -------------------------------------------------------
# LOAD GML
# -------------------------------------------------------
def load_gml(gml_path, use_partition_features=False):
    print(f"[INFO] Loading: {gml_path}")
    G = nx.read_gml(gml_path, label="id")
    nodes = list(G.nodes())

    features, labels, degrees, indegrees = [], [], [], []

    in_deg = dict(G.in_degree()) if G.is_directed() else {n: G.degree(n) for n in G.nodes()}
    total_deg = {n: G.degree(n) for n in G.nodes()}

    for node in nodes:
        attr = G.nodes[node]

        feat = list(attr.get("features", []))
        if use_partition_features:
            feat += list(attr.get("partition_features", [0.0, 0.0]))
        features.append(feat)

        boundary_value = attr.get("boundary", 0)
        try:
            label = int(boundary_value)
        except (ValueError, TypeError):
            label = 0
        labels.append(max(label, 0))

        degrees.append(total_deg.get(node, 0))
        indegrees.append(in_deg.get(node, 0))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=int)
    degrees = np.array(degrees, dtype=int)
    indegrees = np.array(indegrees, dtype=int)

    n_boundary = (y == 1).sum()
    n_total = len(y)
    print(f"  Nodes: {n_total} | Boundary=1: {n_boundary} | Ratio: {n_boundary/n_total:.4f}")

    return X, y, degrees, indegrees


# -------------------------------------------------------
# BASELINES
# -------------------------------------------------------
def random_baseline(y_true, n_runs=10):
    true_ratio = (y_true == 1).mean()
    n_nodes = len(y_true)
    n_pos = max(1, int(true_ratio * n_nodes))

    f1s, precs, recs = [], [], []
    for _ in range(n_runs):
        preds = np.zeros(n_nodes, dtype=int)
        preds[np.random.choice(n_nodes, size=n_pos, replace=False)] = 1
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))

    return {"f1": float(np.mean(f1s)), "precision": float(np.mean(precs)), "recall": float(np.mean(recs))}


def degree_baseline(y_true, degrees):
    n_pos = max(1, int((y_true == 1).mean() * len(y_true)))
    preds = np.zeros(len(y_true), dtype=int)
    preds[np.argsort(-degrees)[:n_pos]] = 1
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }


def indegree_baseline(y_true, indegrees):
    n_pos = max(1, int((y_true == 1).mean() * len(y_true)))
    preds = np.zeros(len(y_true), dtype=int)
    preds[np.argsort(-indegrees)[:n_pos]] = 1
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }


def logistic_regression_baseline(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="saga", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }


def random_forest_baseline(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    # ROC-AUC needs scores; use probability of class 1
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
    except Exception:
        roc_auc = None

    return {
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc,
    }
    

def xgboost_baseline(X_train, y_train, X_test, y_test, seed=42):
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")

    # Handle imbalance: scale_pos_weight = #neg / #pos
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    spw = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    clf.fit(X_train, y_train)

    preds = (clf.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    return {
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }


# -------------------------------------------------------
# PRINT / SAVE HELPERS
# -------------------------------------------------------
def print_result(name, metrics):
    auc_str = f"  AUC={metrics['roc_auc']:.4f}" if 'roc_auc' in metrics and metrics['roc_auc'] is not None else ""
    print(f"  {name:<20} | F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}{auc_str}")


def print_summary_table(all_results, methods):
    print("\n" + "="*90)
    print("SUMMARY TABLE (F1 scores)")
    print("="*90)
    header = f"{'Design':<45}" + "".join(f"{m:<14}" for m in methods)
    print(header)
    print("-" * len(header))
    for design_name, results in all_results.items():
        short_name = design_name[-44:] if len(design_name) > 44 else design_name
        row = f"{short_name:<45}"
        for m in methods:
            row += f"{results[m]['f1']:.4f}        " if m in results else f"{'N/A':<14}"
        print(row)
    print("="*90)


def save_csv(all_results, methods, output_path="baseline_results.csv"):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["design"] + [f"{m}_{metric}" for m in methods for metric in ["f1", "precision", "recall", "roc_auc"]])
        for design_name, results in all_results.items():
            row = [design_name]
            for m in methods:
                if m in results:
                    row += [
                        round(results[m].get("f1", np.nan), 4),
                        round(results[m].get("precision", np.nan), 4),
                        round(results[m].get("recall", np.nan), 4),
                        ("N/A" if results[m].get("roc_auc", None) is None else round(results[m]["roc_auc"], 4)),
                    ]
                else:
                    row += ["N/A", "N/A", "N/A", "N/A"]
            writer.writerow(row)
    print(f"[INFO] Results saved to: {output_path}")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for boundary node detection")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (same config as your main.py)")
    parser.add_argument("--train_gml", nargs="+", default=None,
                        help="Override train GMLs from config")
    parser.add_argument("--test_gml", nargs="+", default=None,
                        help="Override test GMLs from config")
    parser.add_argument("--heuristics_only", action="store_true",
                        help="Run only heuristic baselines (no training data needed)")
    parser.add_argument("--use_partition_features", action="store_true",
                        help="Include partition features (same flag as your main.py)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- load config if provided ----
    train_gmls = args.train_gml
    test_gmls = args.test_gml
    use_partition_features = args.use_partition_features

    if args.config:
        print(f"[INFO] Loading config: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # use config values as defaults, command line args override
        if train_gmls is None:
            train_gmls = cfg.get("train_gml", [])
        if test_gmls is None:
            test_gmls = cfg.get("test_gml", [])
        if not use_partition_features:
            use_partition_features = cfg.get("use_partition_features", False)

    # ---- validate ----
    if not test_gmls:
        print("[ERROR] No test GMLs provided. Use --config or --test_gml")
        return

    if not args.heuristics_only and not train_gmls:
        print("[ERROR] No training GMLs found. Provide --train_gml or use --heuristics_only")
        print("        Tip: your config's train_gml will be used automatically if --config is set")
        return

    print(f"\n[INFO] use_partition_features: {use_partition_features}")
    print(f"[INFO] Test designs: {len(test_gmls)}")

    # ---- decide methods ----
    if args.heuristics_only:
        methods = ["Random", "Degree", "InDegree"]
        print("[INFO] Mode: Heuristics only")
        X_train_scaled, y_train_all, scaler = None, None, None
    else:
        methods = ["Random", "Degree", "InDegree", "LogReg", "RandomForest", "XGBoost"]
        print(f"[INFO] Mode: All baselines | Training designs: {len(train_gmls)}")

        X_trains, y_trains = [], []
        for path in train_gmls:
            if not os.path.exists(path):
                print(f"  [SKIP] File not found: {path}")
                continue
            X, y, _, _ = load_gml(path, use_partition_features)
            X_trains.append(X)
            y_trains.append(y)

        if not X_trains:
            print("[ERROR] No training files could be loaded.")
            return

        X_train_all = np.vstack(X_trains)
        y_train_all = np.concatenate(y_trains)
        print(f"\n[INFO] Total training nodes   : {len(y_train_all)}")
        print(f"[INFO] Training boundary ratio : {(y_train_all==1).mean():.4f}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_all)

    # ---- run baselines on each test graph ----
    print("\n[INFO] Running baselines on test graphs...")
    all_results = {}

    for path in test_gmls:
        if not os.path.exists(path):
            print(f"\n[SKIP] File not found: {path}")
            continue

        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*60}")
        print(f"[BASELINE] {name}")
        print(f"{'='*60}")

        X_test, y_test, degrees, indegrees = load_gml(path, use_partition_features)
        results = {}

        results["Random"] = random_baseline(y_test)
        print_result("Random", results["Random"])

        results["Degree"] = degree_baseline(y_test, degrees)
        print_result("Degree", results["Degree"])

        results["InDegree"] = indegree_baseline(y_test, indegrees)
        print_result("InDegree", results["InDegree"])

        if not args.heuristics_only:
            X_test_scaled = scaler.transform(X_test)

            try:
                results["LogReg"] = logistic_regression_baseline(X_train_scaled, y_train_all, X_test_scaled, y_test)
                print_result("LogReg", results["LogReg"])
            except Exception as e:
                print(f"  LogReg               | FAILED: {e}")

            try:
                results["RandomForest"] = random_forest_baseline(X_train_scaled, y_train_all, X_test_scaled, y_test)
                print_result("RandomForest", results["RandomForest"])
            except Exception as e:
                print(f"  RandomForest         | FAILED: {e}")

            try:
                results["XGBoost"] = xgboost_baseline(X_train_scaled, y_train_all, X_test_scaled, y_test, seed=args.seed)
                print_result("XGBoost", results["XGBoost"])
            except Exception as e:
                print(f"  XGBoost              | FAILED: {e}")

        all_results[name] = results

    print_summary_table(all_results, methods)
    save_csv(all_results, methods)
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()