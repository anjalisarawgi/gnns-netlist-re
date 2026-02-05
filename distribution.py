import json
import matplotlib.pyplot as plt

with open("graph_stats_m1.json", "r") as f:
    data = json.load(f)

# extract boundary_1_ratio values
ratios = [v["num_nodes"] for v in data.values()]

# plot distribution
plt.figure()
plt.hist(ratios, bins=10)
plt.xlabel("num_nodes")
plt.ylabel("count")
plt.title("Distribution of num_nodes across designs")
plt.savefig("distribution_m1_nodes.png")

small = [v["num_nodes"] for v in data.values() if v["num_nodes"] < 30000]
plt.figure()
plt.hist(small, bins=20)
plt.xlabel("num_nodes (<5000)")
plt.ylabel("count")
plt.title("Distribution of num_nodes below 5000")
plt.savefig("distribution_m1_nodes_small.png")

largest =  [v["num_nodes"] for v in data.values() if v["num_nodes"] > 200000]
print("largest:", largest)