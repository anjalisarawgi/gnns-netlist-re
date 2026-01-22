import json 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

with open("data_statistics/graph_stats_combined.json") as f:
    data = json.load(f)

rows = []

for path, stats in data.items():
    parts = path.split("/")

    design = parts[2] 
    library = parts[3] 
    module = parts[-1].replace(".gml", "")


    rows.append({
        "path": path, 
        "design": design, 
        "library": library, 
        "module": module, 
        "num_nodes": stats["num_nodes"], 
        "num_edges": stats["num_edges"],
        "boundary_ratio": stats["boundary_1_ratio"], 
        "avg_degree": 2 * stats["num_edges"] / stats["num_nodes"]  
    })

df = pd.DataFrame(rows)
print(df.head())

# add ing log 
df["log_nodes"] = np.log10(df["num_nodes"])
df["log_edges"] = np.log10(df["num_edges"])


# visualize 
sns.set(style = "whitegrid")

plt.figure(figsize=(6,4))

# a) boundary ratio vs size of the grpah 
plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x="log_nodes", y="boundary_ratio", hue="library",  alpha=0.6)
plt.title("Boundary ratio vs graph size")
plt.tight_layout()
plt.savefig("data_statistics/boundary_vs_graphSize.png")

# b) boundary ration by library 
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="library", y="boundary_ratio")
plt.title("Boundary ratio by library")
plt.tight_layout()
plt.savefig("data_statistics/boundary_vs_library.png")


# c) boundary distribution 
plt.figure(figsize=(6,4))
sns.histplot(df["boundary_ratio"], bins=40, kde=True)
plt.title("Boundary ratio distribution")
plt.tight_layout()
plt.savefig("data_statistics/boundary_dist.png")


# d) graph size vs boundary ratio
plt.figure(figsize=(6,4))
sns.regplot(
    data=df,
    x="log_nodes",
    y="boundary_ratio",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)
plt.title("Boundary ratio vs graph size (log nodes)")
plt.xlabel("log10(number of nodes)")
plt.ylabel("Boundary ratio")
plt.tight_layout()
plt.savefig("data_statistics/boundary_vs_log_nodes_reg.png")
corr = df["log_nodes"].corr(df["boundary_ratio"])
print("Pearson correlation:", corr)

# graph density vs boundary ratio
plt.figure(figsize=(6,4))
sns.regplot(
    data=df,
    x="avg_degree",
    y="boundary_ratio",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)
plt.title("Boundary ratio vs graph density (avg degree)")
plt.xlabel("Average degree")
plt.ylabel("Boundary ratio")
plt.tight_layout()
plt.savefig("data_statistics/boundary_vs_avg_degree_reg.png")
corr = df["avg_degree"].corr(df["boundary_ratio"])
print("Pearson correlation:", corr)

#### 
# # By design
# design_stats = df.groupby("design")["boundary_ratio"].agg(
#     ["mean","std","count"]
# ).sort_values("mean", ascending=False)
# print(design_stats.head(10))

# # By module (large variance!)
# module_stats = df.groupby("module")["boundary_ratio"].agg(
#     ["mean","std","count"]
# ).sort_values("std", ascending=False)
# print(module_stats.head(10))

# By library (large variance!)
module_stats = df.groupby("library")["boundary_ratio"].agg(
    ["mean","std","count"]
).sort_values("std", ascending=False)
print(module_stats.head(10))


# ######
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# features = df[[
#     "log_nodes",
#     "log_edges",
#     "avg_degree"
# ]].values

# X = StandardScaler().fit_transform(features)

# # PCA for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# df["pca1"] = X_pca[:,0]
# df["pca2"] = X_pca[:,1]

# plt.figure(figsize=(6,5))
# sns.scatterplot(
#     x="pca1",
#     y="pca2",
#     hue="boundary_ratio",
#     palette="viridis",
#     data=df,
#     alpha=0.8
# )
# plt.title("PCA of graph structure (colored by boundary ratio)")
# plt.tight_layout()
# plt.savefig("data_statistics/3_structural_analysis.png")


# kmeans = KMeans(n_clusters=4, random_state=42)
# df["cluster"] = kmeans.fit_predict(X)
# print("\n=== Cluster-level numeric summary ===")
# cluster_summary = df.groupby("cluster").agg(
#     count=("path", "count"),
#     median_nodes=("num_nodes", "median"),
#     mean_nodes=("num_nodes", "mean"),
#     median_degree=("avg_degree", "median"),
#     mean_degree=("avg_degree", "mean"),
#     median_boundary=("boundary_ratio", "median"),
#     mean_boundary=("boundary_ratio", "mean"),
# )
# print(cluster_summary)


# print("\n=== Library distribution per cluster ===")
# for c in sorted(df["cluster"].unique()):
#     print(f"\nCluster {c}")
#     print(
#         df[df["cluster"] == c]["library"]
#         .value_counts(normalize=True)
#         .head(5)
#     )


# # Discretize boundary ratio into semantic bands
# df["boundary_band"] = pd.cut(
#     df["boundary_ratio"],
#     bins=[0.0, 0.15, 0.35, 0.60, 1.0],
#     labels=["low", "mid", "high", "very_high"],
#     include_lowest=True
# )
# print("\n=== Boundary ratio bands per cluster ===")
# for c in sorted(df["cluster"].unique()):
#     print(f"\nCluster {c}")
#     print(
#         df[df["cluster"] == c]["boundary_band"]
#         .value_counts(normalize=True)
#     )

# plt.figure(figsize=(6,5))
# sns.scatterplot(
#     x="pca1",
#     y="pca2",
#     hue="cluster",
#     data=df,
#     palette="tab10"
# )
# plt.title("Structural clusters (no labels)")
# plt.tight_layout()
# plt.savefig("data_statistics/3_structural_analysis_2.png")

# print(
#     df.groupby("cluster")["boundary_ratio"]
#       .describe()[["mean","std","min","max"]]
# )


# import os, json
# import numpy as np

# out_dir = "data_statistics/clusters"
# os.makedirs(out_dir, exist_ok=True)

# # 1) Save per-graph listing (with stats) per cluster
# cols = [
#     "path", "design", "library", "module",
#     "num_nodes", "num_edges", "log_nodes", "log_edges",
#     "avg_degree", "boundary_ratio", "cluster"
# ]

# for c in sorted(df["cluster"].unique()):
#     df_c = df[df["cluster"] == c][cols].sort_values(["library", "design", "module"])
#     out_csv = os.path.join(out_dir, f"cluster{c}.csv")
#     df_c.to_csv(out_csv, index=False)
#     print(f"Saved {out_csv} ({len(df_c)} graphs)")

# # 2) Save cluster-level summary stats
# def q(x, p):  # percentile helper
#     return float(np.percentile(x, p))

# summary = {}
# for c in sorted(df["cluster"].unique()):
#     d = df[df["cluster"] == c]
#     summary[str(c)] = {
#         "count": int(len(d)),
#         "library_mix": (d["library"].value_counts(normalize=True).to_dict()),
#         "design_count": int(d["design"].nunique()),
#         "module_count": int(d["module"].nunique()),

#         "num_nodes": {
#             "min": int(d["num_nodes"].min()),
#             "median": float(d["num_nodes"].median()),
#             "mean": float(d["num_nodes"].mean()),
#             "p90": q(d["num_nodes"], 90),
#             "max": int(d["num_nodes"].max()),
#         },
#         "avg_degree": {
#             "min": float(d["avg_degree"].min()),
#             "median": float(d["avg_degree"].median()),
#             "mean": float(d["avg_degree"].mean()),
#             "p90": q(d["avg_degree"], 90),
#             "max": float(d["avg_degree"].max()),
#         },
#         "boundary_ratio": {
#             "min": float(d["boundary_ratio"].min()),
#             "median": float(d["boundary_ratio"].median()),
#             "mean": float(d["boundary_ratio"].mean()),
#             "p90": q(d["boundary_ratio"], 90),
#             "max": float(d["boundary_ratio"].max()),
#         },
#     }

# out_json = os.path.join(out_dir, "cluster_summary.json")
# with open(out_json, "w") as f:
#     json.dump(summary, f, indent=2)
# print(f"Saved {out_json}")


