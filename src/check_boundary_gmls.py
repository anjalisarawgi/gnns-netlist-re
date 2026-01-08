from pathlib import Path
import yaml
import json
import networkx as nx  
from tqdm import tqdm  

cfg_path = "config/train_336_m2.yml"
cfg = yaml.safe_load(Path(cfg_path).read_text())
gml_paths = cfg.get("train_gml", []) + cfg.get("val_gml", []) + cfg.get("test_gml", [])

log_file = "gml_missing_boundary.log"
json_stats_file = "graph_stats.json"

missing_boundary = []
graph_stats = {}

for p in tqdm(gml_paths, desc="Processing GML files"):
    path = Path(p)
    stat = {}

    if not path.exists():
        missing_boundary.append(f"{path}  [FILE NOT FOUND]")
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "boundary" not in text:
            missing_boundary.append(str(path))

        G = nx.read_gml(path)
        boundary_vals = [G.nodes[n].get("boundary") for n in G.nodes]
        num_boundary_1 = sum(1 for v in boundary_vals if v == 1)
        num_total_nodes = len(boundary_vals)

        stat["num_nodes"] = num_total_nodes
        stat["num_edges"] = G.number_of_edges()
        stat["boundary_1_count"] = num_boundary_1
        stat["boundary_1_ratio"] = num_boundary_1 / num_total_nodes if num_total_nodes > 0 else 0

        graph_stats[str(path)] = stat

    except Exception as e:
        missing_boundary.append(f"{path}  [ERROR: {e}]")


Path(log_file).write_text("\n".join(missing_boundary) + ("\n" if missing_boundary else ""))
Path(json_stats_file).write_text(json.dumps(graph_stats, indent=2))
print(f"Checked {len(gml_paths)} files")
print(f"Files missing 'boundary': {len(missing_boundary)}")
print(f"Log written to: {log_file}")
print(f"Graph stats written to: {json_stats_file}")