import yaml
import networkx as nx
from pathlib import Path

with open("config/train_m1_subset_AESSHAGOST.yml", "r") as f:
    config = yaml.safe_load(f)

all_graphs = (
    config.get("train_gml", []) +
    config.get("val_gml", []) +
    config.get("test_gml", [])
)

print(f"Checking {len(all_graphs)} graphs...\n")

any_issues = False
for gml_path in all_graphs:
    if gml_path is None:
        continue
    try:
        G = nx.read_gml(gml_path, label="id")
        total = G.number_of_nodes()
        missing = sum(1 for _, d in G.nodes(data=True) if "boundary" not in d)
        if missing > 0:
            print(f"⚠️  {gml_path}")
            print(f"     nodes={total:,}  missing={missing:,}  ({100*missing/total:.1f}%)")
            any_issues = True
        else:
            print(f"✅  {gml_path}")
    except FileNotFoundError:
        print(f"❌  NOT FOUND: {gml_path}")
        any_issues = True

print()
if any_issues:
    print("Some graphs have issues — see above.")
else:
    print("All graphs clean!")