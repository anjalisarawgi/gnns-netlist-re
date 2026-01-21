import json

INPUT_JSON = "data_statistics/graph_stats_combined.json"   # <-- change this
OUT_LOW  = "bins/combined/boundary_low_combined.txt"
OUT_MID  = "bins/combined/boundary_mid_combined.txt"
OUT_HIGH = "bins/combined/boundary_high_combined.txt"

LOW_THR = 0.30
HIGH_THR = 0.50

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

low, mid, high = [], [], []

for path, stats in data.items():
    r = stats.get("boundary_1_ratio", None)
    if r is None:
        continue

    if r < LOW_THR:
        low.append(path)
    elif r <= HIGH_THR:
        mid.append(path)
    else:
        high.append(path)

def write_list(fname, paths):
    with open(fname, "w") as f:
        for p in sorted(paths):
            f.write(p + "\n")

write_list(OUT_LOW, low)
write_list(OUT_MID, mid)
write_list(OUT_HIGH, high)

print("=== DONE ===")
print(f"Low  (<{LOW_THR}):   {len(low)} graphs → {OUT_LOW}")
print(f"Mid  ({LOW_THR}-{HIGH_THR}): {len(mid)} graphs → {OUT_MID}")
print(f"High (>{HIGH_THR}):  {len(high)} graphs → {OUT_HIGH}")