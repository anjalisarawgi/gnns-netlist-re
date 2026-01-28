import json
import matplotlib.pyplot as plt

with open("graph_stats_m1.json", "r") as f:
    data = json.load(f)

# extract boundary_1_ratio values
ratios = [v["boundary_1_ratio"] for v in data.values()]

# plot distribution
plt.figure()
plt.hist(ratios, bins=20)
plt.xlabel("boundary_1_ratio")
plt.ylabel("count")
plt.title("Distribution of boundary_1_ratio across designs")
plt.savefig("distribution_m1.png")