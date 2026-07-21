import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analysis: all_51/data.csv")


print(f"avg nodes: {df['num_nodes'].mean():.2f}")
print(f"std nodes:{df['num_nodes'].std():.2f}")
print(f"avg edges: {df['num_edges'].mean():.2f}")
print(f"Std edges: {df['num_edges'].std():.2f}")

# Plot 1: Density and Graph Size (2 subplots side by side)
fig2, ax1 = plt.subplots(figsize=(5, 4))

ax1.hist(df["num_nodes"], bins=30, color="#4E79A7", edgecolor="white")
ax1.set_title("Graph size (nodes)")
ax1.set_xlabel("Number of nodes")
ax1.set_ylabel("Count")

# axes1[1].hist(df["graph_density"], bins=30, color="#59A14F", edgecolor="white")
# axes1[1].set_title("Graph density")
# axes1[1].set_xlabel("Density")
# axes1[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("analysis/all_51/graph_node_size.png", dpi=150)
plt.close()


##################
# Plot 2: Boundary ratio (single plot)
fig2, ax2 = plt.subplots(figsize=(5, 4))

ax2.hist(df["boundary_1_ratio_labeled"], bins=30, color="#E15759", edgecolor="white")
ax2.set_title("Boundary ratio")
ax2.set_xlabel("Boundary ratio (labeled nodes)")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("analysis/all_51/boundary_ratio.png", dpi=150)

######################
### plot 3: lift vs library
fig, ax = plt.subplots(figsize=(8, 5))

df["lift"] = df["PR-AUC"] - df["BR"] # calculating the lift

libraries = df["Library"].unique()
data = [df[df["Library"] == lib]["lift"].values for lib in libraries]
bp = ax.boxplot(data, labels=libraries, patch_artist=True, widths=0.5)
colors = ["#4E79A7", "#F28E2B", "#59A14F"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

ax.set_title("PR-AUC lift distribution by synthesis library", fontsize=12)
ax.set_xlabel("Library")
ax.set_ylabel("PR-AUC lift (PR-AUC − boundary ratio)")

plt.tight_layout()
plt.savefig("all_51/prauc_lift_by_library.png", dpi=150)
plt.close()

# print(df.groupby("Library")["lift"].median().round(3))
# print(df.groupby("Library")["lift"].mean().round(3))

