import yaml
import networkx as nx
from collections import defaultdict, Counter
import os

def check_feature_dimensions(gml_paths, feature_key="features"):
    """
    Checks feature dimensionality consistency across multiple GML files.
    """
    file2dim = {}
    dim2files = defaultdict(list)

    for path in gml_paths:
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue

        try:
            G = nx.read_gml(path, label="id")
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            continue

        dims = set()
        bad_nodes = 0

        for n, attr in G.nodes(data=True):
            feat = attr.get(feature_key, None)

            if feat is None:
                dims.add(None)
                bad_nodes += 1
                continue

            if not isinstance(feat, (list, tuple)):
                dims.add("NON_LIST")
                bad_nodes += 1
                continue

            dims.add(len(feat))

        file2dim[path] = dims
        for d in dims:
            dim2files[d].append(path)

        print(
            f"[OK] {os.path.basename(path):60s} | "
            f"dims={sorted(dims)} | bad_nodes={bad_nodes}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for dim, files in sorted(dim2files.items(), key=lambda x: str(x[0])):
        print(f"Feature dim = {dim}: {len(files)} files")

    print("\nFiles with inconsistent feature dimensions:")
    for path, dims in file2dim.items():
        if len(dims) != 1:
            print(f"  [WARN] {path} -> {dims}")

    return file2dim, dim2files

with open("config/train_new_graphs_crypto_jan18.yml", "r") as f:
    cfg = yaml.safe_load(f)

all_paths = (
    cfg["train_gml"]
    + cfg["val_gml"]
    + cfg["test_gml"]
)

check_feature_dimensions(all_paths)