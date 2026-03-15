import json
import pandas as pd
import numpy as np

# load json
with open("data_statistics/graph_stats_m1/graph_stats_m1.json", "r") as f:
    data = json.load(f)

rows = []

for path, stats in data.items():

    parts = path.split("/")

    design = parts[2]
    library = parts[3]

    nodes = stats["num_nodes"]
    edges = stats["num_edges"]
    boundary_ratio = stats["boundary_1_ratio"]

    # graph density (directed assumption)
    density = edges / (nodes * (nodes - 1))

    rows.append({
        "design": design,
        "library": library,
        "nodes": nodes,
        "edges": edges,
        "boundary_ratio": boundary_ratio,
        "density": density
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