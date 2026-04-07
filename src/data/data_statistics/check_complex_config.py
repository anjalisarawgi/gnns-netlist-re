import json
import yaml
import os

# =========================
# CONFIG PATHS (EDIT THESE)
# =========================
YAML_PATH = "config/training_crypto/aes_core.yml"
COMPLEX_JSON_PATH = "config/file_paths/new/complex_graph.json"


# =========================
# HELPERS
# =========================
def normalize(p):
    """Normalize paths to avoid mismatch issues"""
    return os.path.normpath(p)


# =========================
# LOAD FILES
# =========================
with open(YAML_PATH, "r") as f:
    config = yaml.safe_load(f)

with open(COMPLEX_JSON_PATH, "r") as f:
    complex_graphs = json.load(f)

print(f"Loaded YAML: {YAML_PATH}")
print(f"Loaded JSON: {COMPLEX_JSON_PATH}")
print(f"JSON type: {type(complex_graphs)}")


# =========================
# COLLECT ALL GML PATHS
# =========================
all_gml_paths = []

for split in ["train_gml", "val_gml", "test_gml"]:
    paths = config.get(split, [])
    all_gml_paths.extend(paths)

print(f"\nTotal GML paths found: {len(all_gml_paths)}")


# =========================
# EXTRACT COMPLEX PATHS
# =========================
complex_paths = set()

if isinstance(complex_graphs, dict):
    # Case 1: keys are paths
    sample_key = next(iter(complex_graphs.keys()))

    if isinstance(sample_key, str) and sample_key.endswith(".gml"):
        print("Detected: JSON keys are file paths")
        complex_paths = {normalize(k) for k in complex_graphs.keys()}

    else:
        print("Detected: JSON values contain file paths")
        for v in complex_graphs.values():
            if isinstance(v, dict):
                # Try common keys
                for key in ["file_path", "path", "gml_path"]:
                    if key in v:
                        complex_paths.add(normalize(v[key]))

elif isinstance(complex_graphs, list):
    print("Detected: JSON is a list")
    for item in complex_graphs:
        if isinstance(item, dict):
            for key in ["file_path", "path", "gml_path"]:
                if key in item:
                    complex_paths.add(normalize(item[key]))

print(f"Total complex graph entries: {len(complex_paths)}")


# =========================
# CHECK MISSING
# =========================
missing = []
present = []

for path in all_gml_paths:
    norm_path = normalize(path)
    if norm_path in complex_paths:
        present.append(path)
    else:
        missing.append(path)


# =========================
# RESULTS
# =========================
print("\n========== RESULTS ==========")
print(f"Present in complex_graph.json: {len(present)}")
print(f"Missing from complex_graph.json: {len(missing)}")


# =========================
# PRINT MISSING
# =========================
if missing:
    print("\n❌ Missing files:")
    for m in missing:
        print(m)

    # Save missing list
    with open("missing_complex_graphs.txt", "w") as f:
        for m in missing:
            f.write(m + "\n")

    print("\nSaved missing list to missing_complex_graphs.txt")

else:
    print("\n✅ All files are present in complex_graph.json")