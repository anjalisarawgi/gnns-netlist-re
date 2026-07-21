import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("analysis/gnn_vs_rf/results_gnn_rf.csv")
df.columns = df.columns.str.strip()
df["Design"] = df["Design"].str.replace("Design: ", "", regex=False)
df["lift_diff"] = df["PR-AUC_GNN"] - df["PR-AUC_RF"] # lift


# Plot 1: PR-AUC lift vs modularity (GNN and RF side by side) with a trend line (pearsons)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (col, title, color) in zip(axes, [("PR-AUC_RF",  "RF PR-AUC Lift",  "#4E79A7"), ("PR-AUC_GNN", "GNN PR-AUC Lift", "#F28E2B"),]):
    slope, intercept, r, p, _ = stats.linregress(df["Mod"], df[col] - df["BR"])
    
  x_line = np.linspace(df["Mod"].min(), df["Mod"].max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="#aaaaaa", linewidth=1.2,
            linestyle="--", alpha=0.8, zorder=1, label=f"trend  r = {r:.2f}")
    ax.scatter(df["Mod"], df[col] - df["BR"], color=color, s=70, alpha=0.88,
               edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlabel("Modularity ($Q$)", fontsize=12, labelpad=8)
    ax.set_ylabel(title, fontsize=12, labelpad=8)
    ax.tick_params(labelsize=9)
    ax.grid(color="#eeeeee", linewidth=0.6)
    ax.legend(fontsize=9, frameon=True, edgecolor="#cccccc", facecolor="white",
              framealpha=1, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3)
    ax.set_title(f"{title} vs Modularity across crypto design families", fontsize=13, pad=40)

plt.tight_layout()
plt.savefig("analysis/gnn_vs_rf/prauc_lift_modularity_scatter.png", dpi=150, bbox_inches="tight")

###########################
# Plot 2: (GNN - RF) diff across modularity quartiles
df["Mod_quartile"] = pd.qcut(df["Mod"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
quartile_colors = {"Q1": "#4E79A7", "Q2": "#F28E2B", "Q3": "#E15759", "Q4": "#59A14F"}

plt.figure(figsize=(10, 6))
for quartile, color in quartile_colors.items():
    q_df = df[df["Mod_quartile"] == quartile]
    plt.scatter(q_df["Mod"], q_df["lift_diff"], color=color, s=70, alpha=0.88, edgecolors="white", linewidths=0.8, zorder=3, label=f"{quartile} ({q_df['Mod'].min():.3f}-{q_df['Mod'].max():.3f})")
plt.axhline(y=0, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=1)
plt.xlabel("Modularity ($Q$)", fontsize=12, labelpad=8)
plt.ylabel("PR-AUC difference (GNN - RF)", fontsize=12, labelpad=8)
plt.title("Difference in PR-AUC (GNN vs RF) across modularity (divided by quartiles)", fontsize=13, pad=50)
plt.grid(color="#eeeeee", linewidth=0.6)
plt.legend(fontsize=9, frameon=True, edgecolor="#cccccc", facecolor="white",
           framealpha=1, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)

plt.tight_layout()
plt.savefig("analysis/gnn_vs_rf/lift_diff_modularity_quartiles.png", dpi=150, bbox_inches="tight")
