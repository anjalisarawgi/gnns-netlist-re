import os
import json
import matplotlib.pyplot as plt
import numpy as np

threshold_dir = "thresholds"
plot_dir = os.path.join(threshold_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

json_files = [f for f in os.listdir(threshold_dir) if f.endswith(".json")]

for file in json_files:
    graph_name = os.path.splitext(file)[0]
    path = os.path.join(threshold_dir, file)

    with open(path, "r") as f:
        results = json.load(f)

    thresholds = [r["threshold"] for r in results]
    f1 = [r["f1"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    pred_ratio = [r["predicted_ratio"] for r in results]
    true_ratio = [r["true_ratio"] for r in results]

    # predicted_ratio / vs true ratio
    plt.figure()
    plt.plot(thresholds, pred_ratio, label="Predicted Ratio")
    plt.axhline(y=true_ratio[0], linestyle="--", label="True Ratio")
    plt.xlabel("Threshold")
    plt.ylabel("Ratio")
    plt.title(f"{graph_name} - Predicted Boundary Ratio")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{graph_name}_ratio.png"))
    plt.close()

    # F1 / prec/ recall
    plt.figure()
    plt.plot(thresholds, f1, label="F1")
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{graph_name} - Metrics vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{graph_name}_metrics.png"))
    plt.close()

    print(f"Saved plots for {graph_name}")

print("All plots generated.")