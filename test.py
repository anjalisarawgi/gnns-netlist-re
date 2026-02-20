import yaml
from pathlib import Path
from collections import Counter

with open("config/train_m1_neighbours_hetero_onehot_all.yml", "r") as f:
    config = yaml.safe_load(f)

designs = [Path(p).parts[-3] for p in config["train_gml"]]

unique_designs = set(designs)
print(f"Number of unique designs: {len(unique_designs)}")
for d in sorted(unique_designs):
    print(f"  {d}")