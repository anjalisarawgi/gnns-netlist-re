import json
import csv
import os
from collections import defaultdict

with open("complex_graph.json") as f:
    data = json.load(f)

rows = []

for filepath, stat in data.items():
    parts = filepath.split("/")
    
    if len(parts) < 5:
        continue

    family  = parts[2]
    library = parts[3]
    design  = parts[4].replace("_combined_m1.gml", "")

    row = {
        "family":                   family,
        "library":                  library,
        "design":                   design,
        "num_nodes":                stat.get("num_nodes"),
        "num_edges":                stat.get("num_edges"),
        "num_labeled_nodes":        stat.get("num_labeled_nodes"),
        "num_unlabeled_nodes":      stat.get("num_unlabeled_nodes"),
        "boundary_1_count":         stat.get("boundary_1_count"),
        "boundary_1_ratio_total":   stat.get("boundary_1_ratio_total"),
        "boundary_1_ratio_labeled": stat.get("boundary_1_ratio_labeled"),
        "num_unique_partitions":    stat.get("num_unique_partitions"),
    }
    rows.append(row)

rows.sort(key=lambda x: x["family"].lower())

# average boundary_1_ratio_labeled per family
family_ratios = defaultdict(list)
for row in rows:
    val = row["boundary_1_ratio_labeled"]
    if val is not None:
        family_ratios[row["family"]].append(val)

family_avg = {
    family: sum(vals) / len(vals)
    for family, vals in family_ratios.items()
}

print("Average boundary_1_ratio_labeled per family:")
print(f"{'Family':<50} {'Avg BR':>8}")
print("-" * 60)
for family in sorted(family_avg, key=lambda x: x.lower()):
    print(f"{family:<50} {family_avg[family]:>8.4f}")

# write per-design csv
output_path = "complex_graph_all.csv"
fieldnames = list(rows[0].keys())

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# write per-family avg csv
avg_output_path = "complex_graph_family_avg.csv"
avg_rows = [
    {"family": family, "avg_boundary_1_ratio_labeled": avg}
    for family, avg in sorted(family_avg.items(), key=lambda x: x[0].lower())
]

with open(avg_output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["family", "avg_boundary_1_ratio_labeled"])
    writer.writeheader()
    writer.writerows(avg_rows)

print(f"saved {len(rows)} rows to {output_path}")
print(f"saved {len(avg_rows)} family averages to {avg_output_path}")