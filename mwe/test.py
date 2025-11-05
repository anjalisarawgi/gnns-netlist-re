import os
import glob
import networkx as nx
import pandas as pd


adjlist_path = "adjlist/des_latest/osu035/des.txt"   # path to top-level adjlist
partition_dir = "outputFiles/des_latest/osu035/partition_graph/des"  # directory of .pq partitions
out_path = "des_with_boundary.gml"

print(f"Loading main design graph from {adjlist_path} ...")
design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())
print(f"Loaded {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

boundary_dict = {}

for f in glob.glob(os.path.join(partition_dir, "*.pq")):
    if "@top" in f:
        print(f"Skipping top-level partition: {f}")
        continue

    print(f"Processing {os.path.basename(f)} ...")
    subgraph_df = pd.read_parquet(f, engine="pyarrow")
    subgraph = nx.from_pandas_edgelist(subgraph_df, create_using=nx.DiGraph())

    for node in subgraph.nodes():
        anc = nx.ancestors(subgraph, node)
        dec = nx.descendants(subgraph, node)
        boundary = "1" if len(anc) == 0 or len(dec) == 0 else "0"
        boundary_dict[node] = boundary

nx.set_node_attributes(design, boundary_dict, "boundary")

nx.write_gml(design, out_path)
print(f"Saved boundary-annotated graph → {os.path.abspath(out_path)}")
print(f"Total boundary nodes: {sum(int(v) for v in boundary_dict.values())}")