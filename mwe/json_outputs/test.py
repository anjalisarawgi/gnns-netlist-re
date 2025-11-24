import json
from pathlib import Path

root = Path("tum-eisec-benchmarks-main")

netlist_root    = root / "netlist"
partition_root  = root / "partition"

netlist_dirs   = {p.name for p in netlist_root.iterdir() if p.is_dir()}
partition_dirs = {p.name for p in partition_root.iterdir() if p.is_dir()}

all_names = sorted(netlist_dirs | partition_dirs)

both = {}
netlist_only = {}
partition_only = {}

for name in all_names:
    if name in netlist_dirs and name in partition_dirs:
        both[name] = {
            "netlist_path":   str(netlist_root / name),
            "partition_path": str(partition_root / name)
        }
    elif name in netlist_dirs:
        netlist_only[name] = {
            "netlist_path": str(netlist_root / name)
        }
    else:
        partition_only[name] = {
            "partition_path": str(partition_root / name)
        }

combined = {
    "both": both,
    "netlist_only": netlist_only,
    "partition_only": partition_only
}

Path("json_outputs").mkdir(exist_ok=True)

with open("json_outputs/both.json", "w") as f:
    json.dump(both, f, indent=4)

# with open("json_outputs/netlist_only.json", "w") as f:
#     json.dump(netlist_only, f, indent=4)

# with open("json_outputs/partition_only.json", "w") as f:
#     json.dump(partition_only, f, indent=4)

# with open("json_outputs/combined.json", "w") as f:
#     json.dump(combined, f, indent=4)

