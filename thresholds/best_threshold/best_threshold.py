import json
import matplotlib.pyplot as plt

with open("thresholds/best_threshold/best_thresholds_summary.json", "r") as f:
    data = json.load(f)

sorted_items = sorted(data.items(), key=lambda x: x[1]["threshold"])
design_names = [item[0] for item in sorted_items]
thresholds = [item[1]["threshold"] for item in sorted_items]

plt.figure(figsize=(12, 6))  

plt.barh(design_names, thresholds)

plt.xlabel("Best Threshold")
plt.ylabel("Design Name")
plt.title("Best Threshold per Design (Ascending)")

plt.tight_layout()
plt.savefig("thresholds/best_threshold/image.png")
plt.close()