import json

with open("final_graph_stats_m1_apr18.json", "r") as f:
    data = json.load(f)

simple_graph = {}
complex_graph = {}

for key, value in data.items():
    if value.get("num_unique_partitions", 0) <= 2:
        simple_graph[key] = value
    else:
        complex_graph[key] = value

with open("simple_graph_new.json", "w") as f:
    json.dump(simple_graph, f, indent=2)

with open("complex_graph_new.json", "w") as f:
    json.dump(complex_graph, f, indent=2)

print("Split complete ✅")