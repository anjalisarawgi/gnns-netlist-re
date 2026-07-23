import json
import yaml
import os

# just for checking the configs sanity

YAML_PATH = "config/training_crypto/aes_core.yml"
COMPLEX_JSON_PATH = "config/file_paths/new/complex_graph.json"

with open(YAML_PATH) as f:
    config = yaml.safe_load(f)

with open(COMPLEX_JSON_PATH) as f:
    complex_graphs = json.load(f)


all_gml_paths = []
for split in ["train_gml", "val_gml", "test_gml"]:
    all_gml_paths.extend(config.get(split, []))


complex_paths = {os.path.normpath(k) for k in complex_graphs.keys()}

# check which are missing
missing = [p for p in all_gml_paths if os.path.normpath(p) not in complex_paths]
present = [p for p in all_gml_paths if os.path.normpath(p) in complex_paths]

print(f"Present: {len(present)} | Missing: {len(missing)}")

if missing:
    for m in missing:
        print(m)
    with open("missing_complex_graphs.txt", "w") as f:
        f.write("\n".join(missing))
else:
    print("All the files of yml are present in complex_graph.json") 
