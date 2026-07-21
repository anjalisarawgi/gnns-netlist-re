import json 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

with open("graph_stats_m1.json") as f:
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
plt.savefig("data_statistics/m1/boundary_vs_graphSize.png")

# b) boundary ration by library 
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="library", y="boundary_ratio")
plt.title("Boundary ratio by library")
plt.tight_layout()
plt.savefig("data_statistics/m1/boundary_vs_library.png")


# c) boundary distribution 
plt.figure(figsize=(6,4))
sns.histplot(df["boundary_ratio"], bins=40, kde=True)
plt.title("Boundary ratio distribution")
plt.tight_layout()
plt.savefig("data_statistics/m1/boundary_dist.png")


# d) graph size vs boundary ratio
plt.figure(figsize=(6,4))
sns.regplot(data=df, x="log_nodes", y="boundary_ratio", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Boundary ratio vs graph size (log nodes)")
plt.xlabel("log10(number of nodes)")
plt.ylabel("Boundary ratio")
plt.tight_layout()
plt.savefig("data_statistics/m1/boundary_vs_log_nodes_reg.png")
corr = df["log_nodes"].corr(df["boundary_ratio"])
print("Pearson correlation:", corr)

# graph density vs boundary ratio
plt.figure(figsize=(6,4))
sns.regplot(data=df, x="avg_degree", y="boundary_ratio", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Boundary ratio vs graph density (avg degree)")
plt.xlabel("Average degree")
plt.ylabel("Boundary ratio")
plt.tight_layout()
plt.savefig("data_statistics/m1/boundary_vs_avg_degree_reg.png")
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

# By library (large variance)
module_stats = df.groupby("library")["boundary_ratio"].agg(
    ["mean","std","count"]
).sort_values("std", ascending=False)
print(module_stats.head(10))

