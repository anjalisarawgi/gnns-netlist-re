import json
import pandas as pd
import numpy as np

# load json
with open("data_statistics/graph_stats_m1/final_graph_stats_m1.json", "r") as f:
    data = json.load(f)

rows = []

for path, stats in data.items():

    parts = path.split("/")

    design = parts[2]
    library = parts[3]

    nodes = stats["num_nodes"]
    edges = stats["num_edges"]
    boundary_ratio = stats["boundary_1_ratio"]
    num_modules = stats["num_unique_partitions"]

    # graph density (directed assumption)
    density = edges / (nodes * (nodes - 1))

    rows.append({
        "design": design,
        "library": library,
        "nodes": nodes,
        "edges": edges,
        "boundary_ratio": boundary_ratio,
        "density": density,
        "num_modules": num_modules
    })

df = pd.DataFrame(rows)

# dataset level stats
summary = {
    "num_graphs": len(df),
    "num_designs": df["design"].nunique(),
    "num_libraries": df["library"].nunique(),

    "avg_nodes": df["nodes"].mean(),
    "std_nodes": df["nodes"].std(),

    "avg_edges": df["edges"].mean(),
    "std_edges": df["edges"].std(),

    "avg_density": df["density"].mean(),
    "std_density": df["density"].std(),

    "avg_boundary_ratio": df["boundary_ratio"].mean(),
    "std_boundary_ratio": df["boundary_ratio"].std(),
}

print("DATASET SUMMARY")
for k,v in summary.items():
    print(k, ":", v)

print("\nDesign counts")
print(df["design"].value_counts())
print("\nLibrary counts")
print(df["library"].value_counts())


import matplotlib.pyplot as plt

# Node count histogram
plt.figure()
plt.hist(df["nodes"], bins=20)
plt.xlabel("Number of nodes")
plt.ylabel("Number of designs")
plt.title("Node count distribution across all circuit designs")
plt.tight_layout()

plt.savefig("data_statistics/graph_stats_m1/node_distribution_histogram.png", dpi=300)



df_no_outlier = df[df["nodes"] < 300000]
import matplotlib.pyplot as plt

plt.figure()
plt.hist(df_no_outlier["nodes"], bins=20)
plt.xlabel("Number of nodes")
plt.ylabel("Number of designs")
plt.title("Node count distribution across all circuit designs (wo outlier)")
plt.tight_layout()

plt.savefig("data_statistics/graph_stats_m1/node_distribution_histogram_wo300k.png", dpi=300)


#### boundary ratio dist 
plt.figure()
plt.hist(df["boundary_ratio"], bins=20)
plt.xlabel("Boundary node ratio")
plt.ylabel("Number of graphs")
plt.title("Distribution of boundary node ratios across all circuit graphs")
plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_distribution.png",
    dpi=300
)

### boundary ratio vs graph density
plt.figure()
plt.scatter(df["density"], df["boundary_ratio"], alpha=0.6)

plt.xlabel("Graph density")
plt.ylabel("Boundary node ratio")
plt.title("Boundary ratio vs graph density")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_vs_density.png",
    dpi=300
)


### boundary ratio vs graph size
df["num_boundaries"] = df["boundary_ratio"] * df["nodes"]
plt.figure()
plt.scatter(df["nodes"], df["num_boundaries"], alpha=0.6)

plt.xlabel("Number of nodes")
plt.ylabel("Number of boundary nodes")
plt.title("Number of boundary nodes vs graph size")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_vs_nodes.png",
    dpi=300
)



# #test
# design_stats = df.groupby("design")["boundary_ratio"].agg(["mean","std","count"]).sort_values("mean")
# print(design_stats)


# design_boundary = df.groupby("design")["boundary_ratio"].mean().sort_values()

# plt.figure(figsize=(10,6))
# design_boundary.plot(kind="bar")

# plt.ylabel("Average boundary ratio")
# plt.title("Average boundary ratio per circuit design")
# plt.xticks(rotation=90)

# plt.tight_layout()

# plt.savefig(
#     "data_statistics/graph_stats_m1/boundary_ratio_per_design.png",
#     dpi=300
# )

# test
design_stats = df.groupby("design")["boundary_ratio"].agg(["mean","std","count"]).sort_values("mean")
print(design_stats)

means = design_stats["mean"]
stds = design_stats["std"]

plt.figure(figsize=(10,6))

plt.bar(
    means.index,
    means,
    yerr=stds,
    capsize=3,
    color="steelblue",
    error_kw=dict(
        ecolor="gray",
        lw=1,
        capthick=1,
        alpha=0.6
    )
)

plt.ylabel("Average boundary ratio")
plt.title("Average boundary ratio per circuit design")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_per_design.png",
    dpi=300
)





plt.figure(figsize=(12,6))

order = df.groupby("design")["boundary_ratio"].mean().sort_values().index
data = [df[df["design"] == d]["boundary_ratio"] for d in order]

plt.boxplot(
    data,
    labels=order,
    showfliers=False,          # remove black circles
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="steelblue", linewidth=1),
    whiskerprops=dict(color="steelblue", linewidth=1),
    capprops=dict(color="steelblue", linewidth=1),
    medianprops=dict(color="darkblue", linewidth=1.5)
)

plt.ylabel("Boundary ratio")
plt.title("Boundary ratio distribution per circuit design")
plt.xticks(rotation=90, fontsize=6)

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_per_design_boxplot.png",
    dpi=300
)



design_stats = (
    df.groupby("design")["boundary_ratio"]
    .agg(["mean", "std"])
    .sort_values("mean")
)

x = range(len(design_stats))

plt.figure(figsize=(12,6))

plt.errorbar(
    x,
    design_stats["mean"],
    yerr=design_stats["std"],
    fmt="o",
    color="steelblue",
    ecolor="gray",
    capsize=3
)

plt.xticks(x, design_stats.index, rotation=90, fontsize=6)

plt.ylabel("Boundary ratio")
plt.title("Average boundary ratio per circuit design")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_per_design_test.png",
    dpi=300
)


# compute mean and std per design
design_stats = df.groupby("design")["boundary_ratio"].agg(["mean", "std"]).sort_values("mean")

plt.figure(figsize=(10,6))

plt.bar(
    design_stats.index,
    design_stats["mean"],
    yerr=design_stats["std"],
    capsize=3
)

plt.ylabel("Average boundary ratio")
plt.title("Average boundary ratio per circuit design")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_per_design.png",
    dpi=300
)




#############
# boundary vs partition 

plt.figure()

plt.scatter(df["num_modules"], df["boundary_ratio"], alpha=0.6)

plt.xlabel("Number of modules")
plt.ylabel("Boundary node ratio")
plt.title("Boundary ratio vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_vs_modules.png",
    dpi=300
)




import numpy as np

corr = df["num_modules"].corr(df["boundary_ratio"])
print("Correlation between modules and boundary ratio:", corr)

plt.figure()

x = df["num_modules"]
y = df["num_boundaries"]

plt.scatter(x, y, alpha=0.6)

# fit linear regression line
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b

plt.plot(x_line, y_line, linestyle="--", color="black")

plt.xlabel("Number of modules")
plt.ylabel("Number of boundary nodes")
plt.title("Boundary nodes vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_nodes_vs_modules.png",
    dpi=300
)



# corr = df["num_modules"].corr(df["boundary_ratio"])
# print("Correlation between modules and boundary ratio:", corr)


# plt.figure()

# plt.scatter(df["num_modules"], df["num_boundaries"], alpha=0.6)

# plt.xlabel("Number of modules")
# plt.ylabel("Number of boundary nodes")
# plt.title("Boundary nodes vs number of modules")

# plt.tight_layout()

# plt.savefig(
#     "data_statistics/graph_stats_m1/boundary_nodes_vs_modules.png",
#     dpi=300
# )





plt.figure()

plt.scatter(df["num_modules"], df["boundary_ratio"], alpha=0.6)

plt.xlabel("Number of modules")
plt.ylabel("Boundary node ratio")
plt.title("Boundary ratio vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_vs_modules.png",
    dpi=300
)


plt.figure()

plt.scatter(df["num_modules"], df["nodes"], alpha=0.6)

plt.xlabel("Number of modules")
plt.ylabel("Number of nodes")
plt.title("Graph size vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/nodes_vs_modules.png",
    dpi=300
)


plt.figure()

df.boxplot(column="boundary_ratio", by="library")

plt.title("Boundary ratio across synthesis libraries")
plt.suptitle("")
plt.xlabel("Library")
plt.ylabel("Boundary ratio")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/boundary_ratio_by_library.png",
    dpi=300
)



### modules vs graph size
plt.figure()

plt.scatter(df_no_outlier["num_modules"], df_no_outlier["nodes"], alpha=0.6)

plt.xlabel("Number of modules")
plt.ylabel("Number of nodes (graph size)")
plt.title("Graph size vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/nodes_vs_modules.png",
    dpi=300
)
corr = df_no_outlier["num_modules"].corr(df_no_outlier["nodes"])
print("Correlation between modules and graph size:", corr)

df = df_no_outlier

import numpy as np

plt.figure()

x = df["num_modules"]
y = df["nodes"]

plt.scatter(x, y, alpha=0.6)

# linear fit
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = m * x_line + b

plt.plot(x_line, y_line, linestyle="--")

plt.xlabel("Number of modules")
plt.ylabel("Number of nodes (graph size)")
plt.title("Graph size vs number of modules")

plt.tight_layout()

plt.savefig(
    "data_statistics/graph_stats_m1/nodes_vs_modules_2.png",
    dpi=300
)

corr = x.corr(y)
print("Correlation between modules and graph size:", corr)
print("Slope (nodes per module):", m)


design_size_stats = df.groupby("design")["nodes"].agg(["mean", "std"]).round(2)
print(design_size_stats)
design_boundary_stats = df.groupby("design")["boundary_ratio"].mean().round(3)
design_full_stats = df.groupby("design").agg({
    "nodes": ["mean", "std"],
    "boundary_ratio": "mean"
}).round(3)

print(design_full_stats)


import matplotlib.pyplot as plt
import numpy as np

# --- Aggregate per design ---
design_stats = df.groupby("design").agg({
    "boundary_ratio": "mean",
    "nodes": "mean"
})

# sort by boundary ratio
design_stats = design_stats.sort_values("boundary_ratio")

# OPTIONAL: limit number of designs (uncomment if too crowded)
# design_stats = design_stats[-30:]

means = design_stats["boundary_ratio"]
sizes = design_stats["nodes"]

# --- Normalize sizes for color mapping ---
norm = plt.Normalize(sizes.min(), sizes.max())
colors = plt.cm.viridis(norm(sizes))

# --- Create plot ---
fig, ax = plt.subplots(figsize=(12,6))

bars = ax.bar(
    range(len(means)),
    means,
    color=colors
)

# --- Labels & title ---
ax.set_ylabel("Average boundary ratio")
ax.set_title("Boundary ratio per design (colored by graph size)")

# --- X ticks ---
ax.set_xticks(range(len(means)))
ax.set_xticklabels(means.index, rotation=90, fontsize=6)

# --- Colorbar ---
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Average number of nodes (graph size)")

# --- Layout ---
plt.tight_layout()
plt.savefig("data_statistics/graph_stats_m1/boundary_ratio_colored.png", dpi=300)
plt.close()




import matplotlib.pyplot as plt
import numpy as np

# --- compute stats ---
size_stats = df.groupby("design")["nodes"].agg(["mean", "std"]).sort_values("mean")

means = size_stats["mean"]
stds = size_stats["std"]
stds = stds.fillna(0)
x = np.arange(len(means))

# --- normalize colors based on std ---
norm = plt.Normalize(stds.min(), stds.max())
colors = plt.cm.viridis(norm(stds))

# --- plot ---
fig, ax = plt.subplots(figsize=(12,6))

bars = ax.bar(
    x,
    means,
    color=colors
)

# --- labels ---
ax.set_ylabel("Number of nodes (graph size)")
ax.set_title("Mean graph size per design (colored by standard deviation)")

# --- x ticks ---
ax.set_xticks(x)
ax.set_xticklabels(means.index, rotation=90, fontsize=6)

# --- optional: log scale (recommended) ---
# ax.set_yscale("log")

# --- colorbar ---
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Standard deviation of graph size")

# --- layout ---
plt.tight_layout()
plt.savefig("data_statistics/graph_stats_m1/nodes_per_design_std_colored.png", dpi=300)
plt.close()