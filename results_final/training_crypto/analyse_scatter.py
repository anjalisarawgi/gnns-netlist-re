import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# --- load data ---
with open("config/file_paths/final/complex_graph.json") as f:
    data = json.load(f)

# --- mapping ---
folder_to_label = {
    "aes_core":               "aes_core",
    "apbtoaes128_latest":     "apbtoes",
    "aes-encryption_latest":  "aes_enc",
    "aes-master":             "aes_master",
    "gcm-aes_latest":         "gcm_aes",
    "chacha-master":          "chacha",
    "cmac-master":            "cmac",
    "csa_latest":             "csa",
    "des_latest":             "des",
    "gost28147-89_latest":    "gost",
    "hight_latest":           "hight",
    "md5_core-master":        "md5",
    "sha1-master":            "sha1",
    "sha3_latest":            "sha3",
    "sha256-master":          "sha256",
    "simon_core_latest":      "simon",
}

# --- aggregate ---
aggregated = {
    label: {"ratio": [], "nodes": [], "edges": []}
    for label in folder_to_label.values()
}

for filepath, stat in data.items():
    parts = filepath.split("/")
    if len(parts) < 3:
        continue

    folder = parts[2]
    if folder not in folder_to_label:
        continue

    label = folder_to_label[folder]
    aggregated[label]["ratio"].append(stat["boundary_1_ratio"])
    aggregated[label]["nodes"].append(stat["num_nodes"])
    aggregated[label]["edges"].append(stat["num_edges"])

# --- PR-AUC values ---
prauc = {
    "aes_core": 0.72, "apbtoes": 0.24, "aes_enc": 0.77, "aes_master": 0.47,
    "gcm_aes":  0.76, "chacha":  0.22, "cmac":    0.24, "csa":        0.59,
    "des":      0.50, "gost":    0.65, "hight":   0.49, "md5":        0.15,
    "sha1":     0.27, "sha3":    0.37, "sha256":  0.50, "simon":      0.17,
}

# --- prepare arrays ---
labels = [k for k in aggregated if aggregated[k]["ratio"]]

ratio = np.array([
    np.mean(aggregated[k]["ratio"]) * 100 for k in labels
])

nodes = np.array([
    np.mean(aggregated[k]["nodes"]) for k in labels
])

prauc_arr = np.array([
    prauc[k] for k in labels
])

# --- style ---
rcParams["font.family"] = "sans-serif"
rcParams["font.size"]   = 10

COLOR     = "#1D9E75"
DOT_COLOR = "#126b50"
GREY      = "#aaaaaa"

# --- plotting helper ---
def scatter_panel(ax, x, y, xlabel, labels):
    ax.scatter(
        x, y,
        color=COLOR,
        s=75,
        edgecolors=DOT_COLOR,
        linewidths=0.5,
        zorder=3
    )

    # regression
    m, b, r, p, _ = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        xline, m * xline + b,
        color=GREY,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=2
    )

    # labels
    for i in range(len(labels)):
        ax.text(x[i], y[i], labels[i], fontsize=7.5, color="#333333")

    # styling
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=0.8, alpha=0.6)

    ax.text(
        0.97, 0.04,
        f"r = {r:+.2f},  p = {p:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        ha="right"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.7)


# --- create figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

scatter_panel(
    axes[0],
    ratio,
    prauc_arr,
    "Boundary ratio (%)",
    labels
)

scatter_panel(
    axes[1],
    nodes,
    prauc_arr,
    "Mean node count",
    labels
)

# shared y-label
axes[0].set_ylabel("PR-AUC")

# title
plt.suptitle(
    "Graph properties vs PR-AUC (Mode: BiDirected GraphSAGE)",
    fontsize=12
)

plt.tight_layout()

plt.savefig("results_final/training_crypto/structural_analysis_clean.png", dpi=150, bbox_inches="tight")
