import json
from pathlib import Path

labels_path = Path("graphs/processed_v2/usable_graphs.json")

with open(labels_path, "r") as f:
    paths = json.load(f)

m1 = [p for p in paths if "_m1.gml" in p]
m2 = [p for p in paths if "_m2.gml" in p]

with open("graphs/processed_v2/m1.json", "w") as f:
    json.dump(m1, f, indent=2)

with open("graphs/processed_v2/m2.json", "w") as f:
    json.dump(m2, f, indent=2)

print("done, it is split by method 1 and method2 (use method 1)")
