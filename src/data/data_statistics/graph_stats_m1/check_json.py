import json
from collections import defaultdict
from pathlib import Path

# output files
OUT_SIMPLE = Path("config/file_paths/simple_graph_paths_m1.txt")
OUT_COMPLEX = Path("config/file_paths/complex_graph_paths_m1.txt")

with open("data_statistics/graph_stats_m1/final_graph_stats_m1.json", "r") as f:
    data = json.load(f)

design_info = defaultdict(lambda: {
    "total": 0,
    "simple": 0,
    "complex": 0,
    "libraries": set(),
    "simple_paths": [],
    "complex_paths": []
})

# ------------------------------------------------------------------
# BUILD STRUCTURE
# ------------------------------------------------------------------
for path, meta in data.items():
    parts = path.split("/")
    
    if len(parts) >= 3:
        design = parts[-3]
        library = parts[-2]
        num_partitions = meta.get("num_unique_partitions", 0)
        
        design_info[design]["total"] += 1
        design_info[design]["libraries"].add(library)
        
        if num_partitions <= 2:
            design_info[design]["simple"] += 1
            design_info[design]["simple_paths"].append(path)
        else:
            design_info[design]["complex"] += 1
            design_info[design]["complex_paths"].append(path)

# ------------------------------------------------------------------
# PRINT TABLE
# ------------------------------------------------------------------
print(f"{'Design':<30} {'Total':<8} {'Simple':<8} {'Complex':<8} {'Libraries'}")
print("-" * 90)

for design, info in sorted(design_info.items()):
    libs = ", ".join(sorted(info["libraries"]))
    print(f"{design:<30} {info['total']:<8} {info['simple']:<8} {info['complex']:<8} {libs}")

print("-" * 90)
print(f"{'Total Designs':<30} {len(design_info)}")
print(f"{'Total Graphs':<30} {sum(info['total'] for info in design_info.values())}")

# ------------------------------------------------------------------
# SAVE ALL SIMPLE + COMPLEX PATHS
# ------------------------------------------------------------------
with open(OUT_SIMPLE, "w") as f_simple, open(OUT_COMPLEX, "w") as f_complex:
    
    for design, info in sorted(design_info.items()):
        
        # --- simple ---
        if info["simple_paths"]:
            f_simple.write(f"\n=== {design} ===\n")
            for p in info["simple_paths"]:
                f_simple.write(p + "\n")
        
        # --- complex ---
        if info["complex_paths"]:
            f_complex.write(f"\n=== {design} ===\n")
            for p in info["complex_paths"]:
                f_complex.write(p + "\n")

print(f"\nSaved simple graph paths → {OUT_SIMPLE}")
print(f"Saved complex graph paths → {OUT_COMPLEX}")