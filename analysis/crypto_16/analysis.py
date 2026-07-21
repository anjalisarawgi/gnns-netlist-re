import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

df = pd.read_csv("analysis/crypto_16/results.csv")

COLORS = {"Correct": "#4E79A7", "Incorrect": "#F28E2B"}

def make_boxplot(ax, data_groups, title, ylabel, formatter=None):
    bp = ax.boxplot(data_groups, labels=["Correct", "Incorrect"], patch_artist=True, widths=0.45, medianprops=dict(color="white", linewidth=2))
  
    for patch, color in zip(bp["boxes"], COLORS.values()):
        patch.set_facecolor(color)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

  # only for homogenity plot
  if formatter:
        ax.yaxis.set_major_formatter(formatter)
    patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
    ax.legend(handles=patches, fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# boxplot 1 for neighbouthood homoegenity
make_boxplot(axes[0],
    [df["Neighborhood_Homogeneity_Correct"].values, df["Neighborhood_Homogeneity_Incorrect"].values],
    "Neighborhood homogeneity", "Neighborhood homogeneity (%)",
    formatter=plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

# boxplot 2 for distance to boundary
make_boxplot(axes[1],
    [df["Distance_to_Boundary_Correct"].values, df["Distance_to_Boundary_Incorrect"].values],
    "Distance to boundary", "Average distance to the closest boundary node")


plt.tight_layout()
plt.savefig("analysis/crypto_16/error_boxplots.png", dpi=150, bbox_inches="tight")



# print("Median values for all4":)
# for col, label in [
#     ("Distance_to_Boundary_Correct",          "Distance - Correct"),
#     ("Distance_to_Boundary_Incorrect",        "Distance - Incorrect"),
#     ("Neighborhood_Homogeneity_Correct",      "Homogeneity - Correct"),
#     ("Neighborhood_Homogeneity_Incorrect",    "Homogeneity - Incorrect"),
# ]:
#     print(f"{label}: {np.median(df[col].values):.4f}")




########################################
# structural features plots 

# 1: reach asymmetry (boundary vs internal)
df = pd.read_csv("analysis/crypto_16/results_2.csv")
df.columns = df.columns.str.strip()
df["Design"] = df["Design"].str.replace("Design: ", "", regex=False)
df["lift"] = df["PR_AUC"] - df["BR"]
df["success"] = df["lift"] >= 0.30


df_sorted = df.sort_values("PR_AUC").reset_index(drop=True)
df_sorted["Label"] = df_sorted["Design"] + " (" + df_sorted["Graph"] + ")"

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_sorted.index, df_sorted["Reach_Asym_I"], color="#76B7B2", linewidth=1.6, marker="o", markersize=4, alpha=0.88, zorder=3, label="Internal nodes")
ax.plot(df_sorted.index, df_sorted["Reach_Asym_B"], color="#E15759", linewidth=1.6, marker="o", markersize=4, alpha=0.88, zorder=3, label="Boundary nodes")
ax.axhline(y=0, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=1)
ax.set_xticks(df_sorted.index)
ax.set_xticklabels(df_sorted["Label"], rotation=45, ha="right", fontsize=5)
ax.set_ylabel("Reach asymmetry", fontsize=12, labelpad=8)
ax.set_xlabel("Designs (sorted by PR-AUC from low to high)", fontsize=11, labelpad=8)
ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
ax.grid(axis="x", color="#eeeeee", linewidth=0.3, linestyle=":")
ax.legend(fontsize=9, frameon=True, edgecolor="#cccccc", facecolor="white", framealpha=1, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
ax.set_title("Reach asymmetry for internal vs boundary nodes across crypto design families (sorted by PR-AUC)", fontsize=15, pad=35)
ax.set_xlim(-0.5, len(df_sorted) - 0.5)
plt.tight_layout()
plt.savefig("analysis/crypto_16/reach_asymmetry_b_i.png", dpi=150, bbox_inches="tight")
plt.close()






# plot 2: for all three strucutral featuers we are analysing here
features = {
    "Degree": ("Degree_B","Degree_I"),
    "2-Hop Density": ("2hop_ego_B","2hop_ego_I"),
    "Reach Asym.":("Reach_Asym_B", "Reach_Asym_I"),
}

success_df = df[df["success"]]
failure_df = df[~df["success"]]
colors = {"Boundary": "#E15759", "Internal": "#76B7B2"}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (feat_label, (col_b, col_i)) in zip(axes, features.items()):
    means = [success_df[col_b].mean(), success_df[col_i].mean(),
             failure_df[col_b].mean(), failure_df[col_i].mean()]
    bar_colors = [colors["Boundary"], colors["Internal"]] * 2
    ax.bar(np.arange(4), means, color=bar_colors, width=0.6, edgecolor="white", linewidth=0.8)
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(["Success\n(Lift ≥ 0.30)", "Failure\n(Lift < 0.30)"], fontsize=10)
    ax.axvline(x=1.5, color="#cccccc", linewidth=1, linestyle="--")
    ax.axhline(y=0, color="#999999", linewidth=0.8)
    ax.set_title(feat_label, fontsize=11, pad=8)
    ax.set_xlim(-0.5, 3.5)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Average value", fontsize=12, labelpad=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=colors["Boundary"]),
           plt.Rectangle((0, 0), 1, 1, color=colors["Internal"])]
fig.legend(handles, ["Boundary nodes", "Internal nodes"], fontsize=9, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2)
fig.suptitle("Comparison of mean feature values for successful vs failed signature inversions", fontsize=15, y=1.08)
plt.tight_layout()
plt.savefig("analysis/crypto_16/all_3_features_b_i.png", dpi=150, bbox_inches="tight")



# print("Mean values of the features for each boundary and internal:")
# for feat_label, (col_b, col_i) in features.items():
#     print(f"{feat_label}:")
#     print(f"  Success cases::: Boundary: {success_df[col_b].mean():.3f}, Internal: {success_df[col_i].mean():.3f}")
#     print(f"  Failure cases::: Boundary: {failure_df[col_b].mean():.3f}, Internal: {failure_df[col_i].mean():.3f}")
  
