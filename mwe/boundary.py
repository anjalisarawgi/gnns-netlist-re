import os
import glob
import networkx as nx
import pandas as pd


adjlist_path = "adjlist/aes_cipher/osu035/aes_key_expand_128.txt"   # path to top-level adjlist
partition_dir = "outputFiles/aes_cipher/osu035/partition_graph/aes_key_expand_128"  # directory of .pq partitions
out_path = "boundary_graphs/aes-encryption_latest/osu035/aes_key_expand_128_v2.gml"

print(f"Loading main design graph from {adjlist_path} ...")
design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())

boundary_dict = {}
print(f"Loaded {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

for f in glob.glob(os.path.join(partition_dir, "*.pq")):
    # skip the top file
    if "@top" in f:
        print(f"Skipping top-level partition: {os.path.basename(f)}")
        continue
    
    # for each partition .pq file - this i think contains the edges of one partition 
    print(f"Processing {os.path.basename(f)} ...")
    subgraph_df = pd.read_parquet(f, engine="pyarrow")
    print(subgraph_df.columns) ## has  a 'source' and a 'target'
    # also these are possibly like edges from source node A to target node C (for example)

    # now: this extracts all the nodes in the parition 
    # combine nodes in source and target - take union - remove duplicates - - gives full sets of all nodes
    subgraph_nodes = set(subgraph_df["source"]).union(set(subgraph_df["target"])) 
    # now: to check if it is a boundary node
    for node in subgraph_nodes: 
        if node not in design:
            continue  
        
        # important = checking thsi from the FULL DESIGN
        parents = list(design.predecessors(node)) 
        children = list(design.successors(node))

        # boundary = 1 if it has atleast one input (coming from outside this parition)  /  (going outside this parition )
        # it has a parent outide its partition?
        # it has a child outside its partition?
        if any(p not in subgraph_nodes for p in parents) or any(c not in subgraph_nodes for c in children):
            boundary_dict[node] = 1
        else:
            boundary_dict[node] = 0

nx.set_node_attributes(design, boundary_dict, "boundary")
nx.write_gml(design, out_path)


print(f"Saved boundary-annotated graph → {os.path.abspath(out_path)}")
print(f"Total boundary nodes: {sum(int(v) for v in boundary_dict.values())}")


# design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())
# print(f"Loaded {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

# boundary_dict = {}

# for f in glob.glob(os.path.join(partition_dir, "*.pq")):
#     # if "@top" in f:
#     #     print(f"Skipping top-level partition: {f}")
#     #     continue

#     print(f"Processing {os.path.basename(f)} ...")
#     subgraph_df = pd.read_parquet(f, engine="pyarrow")
#     subgraph = nx.from_pandas_edgelist(subgraph_df, create_using=nx.DiGraph())

#     for node in subgraph.nodes():
#         anc = nx.ancestors(subgraph, node)
#         dec = nx.descendants(subgraph, node)
#         boundary = 1 if len(anc) == 0 or len(dec) == 0 else 0
#         boundary_dict[node] = boundary

     # Load the full design graph first


#####################

# ### green to green
# boundary_peer_only = {}

# for node, label in boundary_dict.items():
#     if label != 1:
#         continue

#     neighbors = set(design.successors(node)).union(design.predecessors(node))
#     neighbor_labels = [boundary_dict.get(n, -1) for n in neighbors]

#     if all(l == 1 for l in neighbor_labels):
#         boundary_peer_only[node] = 1

# # Set "boundary_peer_only" attribute to 1 for tagged nodes, 0 for others
# peer_only_attr = {node: 1 if node in boundary_peer_only else 0 for node in design.nodes()}
# nx.set_node_attributes(design, peer_only_attr, "boundary_peer_only")

###
