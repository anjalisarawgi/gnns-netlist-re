import os 
import subprocess
from pathlib import Path
import sys
import glob
import networkx as nx
import pandas as pd
from tqdm import tqdm

verliog_root = Path("tum-eisec-benchmarks-main/netlist")
partition_root = Path("tum-eisec-benchmarks-main/partition")

adj_root = Path("adjlist")
graph_root = Path("graphs/raw")
output_files_root = Path("")
LIB = "lib/osu035_stdcells.lib"
# LIB = "lib/NangateOpenCellLibrary_functional.lib"
# LIB = "gscl45nm_2.lib"


def run(cmd):
    print(">>", " ".join(map(str,cmd)))
    subprocess.run(cmd, check=True)


# def find_boundaries_method1(adjlist_path, partition_dir, out_gml):
#     print(f"[INFO] M1: Loading design from {adjlist_path}")
#     design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())
#     print(f"[INFO] M1: Design: {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

#     boundary_dict = {}
#     partition_files = glob.glob(os.path.join(partition_dir, "*.pq"))

#     for f in partition_files:
#         if "@top" in f:
#             continue

#         print(f"[INFO] M1: Processing {os.path.basename(f)} ...")
#         df = pd.read_parquet(f)
#         sub_nodes = set(df["source"]) | set(df["target"])

#         for node in sub_nodes:
#             if node not in design:
#                 continue

#             parents = set(design.predecessors(node))
#             children = set(design.successors(node))

#             # Boundary if ANY connection goes outside partition
#             is_boundary = (
#                 any(p not in sub_nodes for p in parents) or
#                 any(c not in sub_nodes for c in children)
#             )

#             boundary_dict[node] = int(is_boundary)

#     # for node in design.nodes():
#     #     if node not in boundary_dict:
#     #         boundary_dict[node] = -1

#     nx.set_node_attributes(design, boundary_dict, "boundary")
#     nx.write_gml(design, out_gml)
#     print(f"[INFO] M1: Saved to {out_gml}")


def find_boundaries_method1(adjlist_path, partition_dir, out_gml):
    print(f"[M1] Loading design from {adjlist_path}")
    design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())
    print(f"[M1] Design: {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

    boundary_dict = {}
    partition_files = glob.glob(os.path.join(partition_dir, "*.pq"))

    # for f in partition_files:
    for f in tqdm(partition_files, desc="[M1] Partitions", unit="file"):
        if "@top" in f:
            continue

        print(f"[M1] Processing {os.path.basename(f)} ...")
        df = pd.read_parquet(f)

        # Build subgraph with only internal edges
        subgraph = nx.from_pandas_edgelist(df, source="source", target="target", create_using=nx.DiGraph())
        nodes_list = list(subgraph.nodes())
        # for node in subgraph.nodes():
        for node in tqdm(nodes_list, desc=f"[M1] {os.path.basename(f)}", unit="node", leave=False):
            # anc = nx.ancestors(subgraph, node)
            # dec = nx.descendants(subgraph, node)
            # is_boundary = int(len(anc) == 0 or len(dec) == 0)

            # if slow?
            in_deg = subgraph.in_degree(node)
            out_deg = subgraph.out_degree(node)
            is_boundary = int(in_deg == 0 or out_deg == 0)
            boundary_dict[node] = is_boundary

    # for node in design.nodes():
    #     if node not in boundary_dict:
    #         boundary_dict[node] = -1

    nx.set_node_attributes(design, boundary_dict, "boundary")
    nx.write_gml(design, out_gml)
    print(f"[M1] Saved to {out_gml}")

def find_boundaries_method2(adjlist_path, partition_graph_dir, out_gml):
    design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())

    boundary_dict = {}
    print(f"[INFO] M2: Loaded {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

    for f in glob.glob(os.path.join(partition_graph_dir, "*.pq")):
        # skip the top file
        if "@top" in f:
            print(f"[INFO] M2: Skipping top-level partition: {os.path.basename(f)}")
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
    nx.write_gml(design, out_gml)


    print(f"[INFO] M2: Saved boundary-annotated graph → {os.path.abspath(out_gml)}")
    print(f"[INFO] M2: Total boundary nodes: {sum(int(v) for v in boundary_dict.values())}")


def merge_partition_and_boundary(gml_partition, gml_boundary, out_gml):
    """
    Merges partition GML with boundary GML.
    Boundary attributes override partition attributes except for 'label'.
    """
    import networkx as nx
    import os

    def clean_label(label):
        if isinstance(label, str):
            s = label.strip()
            if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
                return s[1:-1]
            return s
        return label

    print(f"[MERGE] Partition GML: {gml_partition}")
    print(f"[MERGE] Boundary  GML: {gml_boundary}")

    Gp = nx.read_gml(gml_partition, label="id")
    Gb = nx.read_gml(gml_boundary, label="id")

    # Clean labels
    for G in (Gp, Gb):
        for _, data in G.nodes(data=True):
            if "label" in data:
                data["label_copy"] = data["label"]
                data["label"] = clean_label(data["label"])

    directed = Gp.is_directed() or Gb.is_directed()
    Gm = nx.DiGraph() if directed else nx.Graph()

    all_nodes = set(Gp.nodes()) | set(Gb.nodes())

    for nid in all_nodes:
        merged = {}

        # Start with boundary attributes
        if nid in Gb:
            merged.update(Gb.nodes[nid])

        # Add partition attributes (do not override cleaned label)
        if nid in Gp:
            for k, v in Gp.nodes[nid].items():
                if k == "label":
                    continue
                merged[k] = v

        Gm.add_node(nid, **merged)

    # Combine edges
    Gm.add_edges_from(Gp.edges(data=True))
    Gm.add_edges_from(Gb.edges(data=True))

    os.makedirs(os.path.dirname(out_gml), exist_ok=True)
    nx.write_gml(Gm, out_gml)

    print(f"[MERGE] Saved merged graph → {out_gml}")

def process_verilog(verilog_file):
    # example : 
    #  tum-eisec-benchmarks-main/netlist/des_latest/verilog/osu035/des.v

    rel_path = verilog_file.relative_to(verliog_root) # --> des_latest/verilog/osu035/des.v
    prefix = rel_path.parts[0] # des_latest
    library = rel_path.parts[2] # osu035
    module = verilog_file.stem # des

    # so now the directories
    adj_out_dir = adj_root / prefix / library / module 
    partition_in_dir = partition_root / prefix / library 
    # graphs_out_dir = graph_root / prefix / library 
    # graphs_out_dir.mkdir(parents = True, exist_ok=True)

    # graphs our directory 
    graph_base = graph_root / prefix / library 
    partitions_dir = graph_base / "partitions"
    partitions_dir.mkdir(parents = True, exist_ok = True)

    # boundaries directory 
    boundary1_dir = graph_base / "boundary_1"
    boundary2_dir = graph_base / "boundary_2"
    boundary1_dir.mkdir(parents=True, exist_ok=True)
    boundary2_dir.mkdir(parents=True, exist_ok=True)

    adj_out_dir.mkdir(parents = True, exist_ok=True)
    # partition_out_dir.mkdir(parents = True, exist_ok=True)
    

    ######
    # processing the three steps now 
    print("[INFO] Processing:", {verilog_file})

    # step 1: 
    print("[INFO] Step 1: Verilog to adj list")
    run([
        "verilog2graph", 
        str(verilog_file), 
        "--output_format", "txt", 
        "--do-not-zip", 
        "-o", str(adj_out_dir), 
        "-l", LIB
    ])

    # step 2: 
    print("[INFO] Step 2: clustering")
    run([
        "run_clustering", 
        str(adj_out_dir), 
        "-i", str(partition_in_dir), 
        "-u", f"{prefix}/{library}/",
        "--task", "partition", 
        "--cores", "1"
    ])
            # "-u", partition_out_dir,  # check this

    # step 2: 
    print("[INFO] Step 3: partition to GML")
    run([
        "partition2gephi", 
        str(adj_out_dir/f"{module}.txt"), 
        "--do-not-zip", "-t", "gml", 
        "-o", str(partitions_dir),  
        "-p", f"outputFiles/{prefix}/{library}/partition_graph",
        "--cores", "1"

    ])  

    # step 4: Boundary extraction
    adjlist_path = adj_out_dir / f"{module}.txt"
    partition_graph_dir = Path(f"outputFiles/{prefix}/{library}/partition_graph/{module}")

    out_gml_1 = boundary1_dir / f"{module}_boundary_1.gml"
    out_gml_2 = boundary2_dir / f"{module}_boundary_2.gml"

    print("[INFO] Running Method 1...")
    find_boundaries_method1(adjlist_path, partition_graph_dir, out_gml_1)

    print("[INFO] Running Method 2...")
    find_boundaries_method2(adjlist_path, partition_graph_dir, out_gml_2)

    # Step 5: Merge partitions with boundary_1 and boundary_2

    # Find the partition GML (partition2gephi created multiple files, pick the module's one)
    partition_gml_candidates = list(partitions_dir.glob(f"{module}*_gephi.gml"))
    if len(partition_gml_candidates) == 0:
        raise FileNotFoundError(f"No partition GML found in {partitions_dir}")
    partition_gml = partition_gml_candidates[0]

    combined_m1 = graph_base / f"{module}_combined_m1.gml"
    combined_m2 = graph_base / f"{module}_combined_m2.gml"

    print("[INFO] Step 5: Merge partition + boundary_1")
    merge_partition_and_boundary(partition_gml, out_gml_1, combined_m1)

    print("[INFO] Step 6: Merge partition + boundary_2")
    merge_partition_and_boundary(partition_gml, out_gml_2, combined_m2)

    # final debugging / stats
    def print_gml_stats(name, gml_path):
        G = nx.read_gml(gml_path, label="id")
        print(f"{name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  →  {gml_path}")

    print_gml_stats("Partition GML", partition_gml)
    print_gml_stats("Boundary Method 1", out_gml_1)
    print_gml_stats("Boundary Method 2", out_gml_2)
    print_gml_stats("Combined M1", combined_m1)
    print_gml_stats("Combined M2", combined_m2)



if __name__ =="__main__":
    process_verilog(Path(sys.argv[1]))