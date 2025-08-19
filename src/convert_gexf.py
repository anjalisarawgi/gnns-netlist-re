import networkx as nx
import glob
import os

in_dir = "data/top_files"
out_dir = "data/top_files/gexf"
os.makedirs(out_dir, exist_ok=True)

for path in glob.glob(os.path.join(in_dir, "*.gml")):
    G = nx.read_gml(path)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".gexf")
    nx.write_gexf(G, out_path)
    print(f"Converted {path} -> {out_path}")