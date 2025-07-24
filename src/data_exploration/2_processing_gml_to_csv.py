import networkx as nx
import os
import pandas as pd
from tqdm import tqdm

gml_dir = "../../data/gscl45nm"
results = []

gml_files = [f for f in os.listdir(gml_dir) if f.endswith(".gml")]

for file in tqdm(gml_files):
    path = os.path.join(gml_dir, file)
    try:
        G = nx.read_gml(path)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        node_labels = list(G.nodes())
        label_upper = [str(l).upper() for l in node_labels]

        num_inputs = sum(1 for l in label_upper if l.startswith("INPUT"))
        num_outputs = sum(1 for l in label_upper if l.startswith("OUTPUT"))
        num_xor = sum(1 for l in label_upper if "XOR" in l)
        num_and = sum(1 for l in label_upper if "AND" in l)
        num_or = sum(1 for l in label_upper if "OR" in l)
        # num_buf = sum(1 for l in label_upper if "BUF" in l)

        num_known_gates = num_xor + num_and + num_or 
        # num_unknown = num_nodes - num_known_gates

        avg_degree = round(sum(dict(G.degree()).values()) / num_nodes, 2)
        is_connected = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)

        results.append({
            "file": file,
            "#nodes": num_nodes,
            "#edges": num_edges,
            "#inputs": num_inputs,
            "#outputs": num_outputs,
            "#XOR": num_xor,
            "#AND": num_and,
            "#OR": num_or,
            # "#unknown": num_unknown,
            "avg_degree": avg_degree,
            "is_connected": is_connected
        })

    except Exception as e:
        print(f"problem with reading {file}: {e}")

df = pd.DataFrame(results)
df.to_csv("data/csvs/gscl45nm.csv", index=False)
