import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

CSV_PATH = "analysis/all_51/results.csv"         
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

col_map = {
    "PR-AUC":                "prauc",
    "num_nodes":             "num_nodes",
    "BR":                    "br",
    "num_unique_partitions": "num_subcircuits",
    "graph_density":         "graph_density",
    "Library":               "library",
}
df = df.rename(columns=col_map)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

# ─ 
def scatter_plot(ax, x, y, title, xlabel, log_x=False, color=None, use_shaded=False):
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    ax.scatter(x, y, c=color, alpha=0.85, s=28, edgecolors="none", zorder=3)

    # Trend line 
    xv = np.log10(x) if log_x else x
    if len(xv) > 2:
        z  = np.polyfit(xv, y, 1)
        p  = np.poly1d(z)
        xs = np.linspace(xv.min(), xv.max(), 300)
        
        ax.plot(10**xs if log_x else xs, p(xs),
                color="#9C755F", lw=1.8, ls="--", zorder=4, label="trend")

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    if log_x and use_shaded:
        ax.axvspan(0, 2500, alpha=0.10, color="grey", zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("PR-AUC", fontsize=10)
    ax.set_ylim(0, 1.05)


# plot 1
scatter_plot(
    axes[0, 0],
    df["num_nodes"], df["prauc"],
    title="PR-AUC  vs  total nodes in graph",
    xlabel="Total nodes in graph  (log scale)",
    log_x=True,
    use_shaded=True,
)

axes[0, 0].axvline(2500, color="grey", linestyle=":", linewidth=1)
axes[0, 0].text(35, 0.08, "< 2500 nodes", fontsize=9, color="grey")
# plot 2 - box plot we bin it
ax = axes[0, 1]

try:
    df["br_bin"] = pd.qcut(df["br"], q=5, duplicates="drop")
    groups   = [grp["prauc"].dropna().values
                for _, grp in df.groupby("br_bin", observed=True)]
    labels   = [str(interval) for interval in
                df.groupby("br_bin", observed=True).groups.keys()]
except Exception:
    df["br_bin"] = df["br"].round(2)
    grouped  = df.groupby("br_bin")["prauc"]
    groups   = [g.dropna().values for _, g in grouped]
    labels   = [str(k) for k in grouped.groups.keys()]

bp = ax.boxplot(
    groups,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="black", linewidth=1),
    flierprops=dict(marker="o", color="#4E79A7", alpha=0.4,
                    markersize=4, linestyle="none"),
)

for patch in bp["boxes"]:
    patch.set_facecolor("#F28E2B")
    # patch.set_alpha(0.90)


ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
ax.set_title("PR-AUC  vs  boundary ratio ", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Boundary ratio (binned)", fontsize=9)
ax.set_ylabel("PR-AUC", fontsize=9)
ax.set_ylim(0, 1.05)

# plot 3
scatter_plot(
    axes[1, 0],
    df["num_subcircuits"], df["prauc"],
    title="PR-AUC  vs  total sub-circuits",
    xlabel="Number of unique partitions",
    log_x=False,
    color="#E15759",
)

# plot 4
scatter_plot(
    axes[1, 1],
    df["graph_density"], df["prauc"],
    title="PR-AUC  vs  graph density",
    xlabel="Graph density (log scale)",
    log_x=True,
    color="#59A14F",
)


OUTPUT = "analysis/all_51/prauc_analysis.png"
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")

###
# Calculate correlations
corr_linear = df['br'].corr(df['num_nodes'])
corr_log = df['br'].corr(np.log10(df['num_nodes']))

print("Correlation analysis::")
print(f"Pearson correlation (BR vs Graph Size): {corr_linear:.4f}")
print(f"Pearson correlation (BR vs Log10 Graph Size): {corr_log:.4f}")

## print correlation between prauc and density - spearman
corr_density = df['prauc'].corr(df['graph_density'], method='spearman')
print(f"Spearman correlation (PR-AUC vs Graph Density): {corr_density:.4f}")

# pritn correlation between prauc and num_subcircuits - spearman
corr_subcircuits = df['prauc'].corr(df['num_subcircuits'], method = 'spearman')
print(f"Spearman correlation (PR-AUC vs Num Subcircuits): {corr_subcircuits:.4f}")

# print correlation between prauc and num_nodes - spearman
corr_nodes = df['prauc'].corr(df['num_nodes'], method='spearman')
print(f"Spearman correlation (PR-AUC vs Num Nodes): {corr_nodes:.4f}")
