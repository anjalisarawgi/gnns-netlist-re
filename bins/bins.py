import json

INPUT_JSON = "graph_stats.json"   # <-- change this
OUT_LOW  = "bins/boundary_low.txt"
OUT_MID  = "bins/boundary_mid.txt"
OUT_HIGH = "bins/boundary_high.txt"

LOW_THR = 0.40
HIGH_THR = 0.550

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