import os 
import subprocess
from pathlib import Path
import sys
import glob
import networkx as nx
import pandas as pd

verliog_root = Path("tum-eisec-benchmarks-main/netlist")
partition_root = Path("tum-eisec-benchmarks-main/partition")

adj_root = Path("adjlist")
graph_root = Path("graphs/raw")
output_files_root = Path("")
LIB = "lib/osu035_stdcells.lib"


def run(cmd):
    print(">>", " ".join(map(str,cmd)))
    subprocess.run(cmd, check=True)

def find_boundaries(adjlist_path, partition_graph_dir, out_gml):
    design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())

    boundary_dict = {}
    print(f"Loaded {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

    for f in glob.glob(os.path.join(partition_graph_dir, "*.pq")):
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
    nx.write_gml(design, out_gml)


    print(f"Saved boundary-annotated graph → {os.path.abspath(out_gml)}")
    print(f"Total boundary nodes: {sum(int(v) for v in boundary_dict.values())}")


def merge_partition_and_boundary(
    gml_partition: Path,
    gml_boundary: Path,
    output_path: Path
):
    import networkx as nx
    import os

    def clean_label(label):
        if isinstance(label, str):
            s = label.strip()
            # strip outer quotes like "'foo'"
            if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
                return s[1:-1]
            return s
        return label

    print(f"[INFO] Loading partition GML: {gml_partition}")
    print(f"[INFO] Loading boundary  GML: {gml_boundary}")

    Gp = nx.read_gml(gml_partition, label="id")
    Gb = nx.read_gml(gml_boundary,  label="id")

    # Clean labels (but keep original)
    for G in (Gp, Gb):
        for _, data in G.nodes(data=True):
            if "label" in data:
                data["label_copy"] = data["label"]
                data["label"] = clean_label(data["label"])

    # Directed or not
    directed = Gp.is_directed() or Gb.is_directed()
    Gm = nx.DiGraph() if directed else nx.Graph()

    # Merge nodes
    all_nodes = set(Gp.nodes()) | set(Gb.nodes())
    for nid in all_nodes:
        merged_attrs = {}

        if nid in Gb:
            merged_attrs.update(Gb.nodes[nid])

        if nid in Gp:
            for k, v in Gp.nodes[nid].items():
                if k == "label":     # do not override cleaned label
                    continue
                merged_attrs[k] = v

        Gm.add_node(nid, **merged_attrs)

    # Merge edges
    Gm.add_edges_from(Gb.edges(data=True))
    Gm.add_edges_from(Gp.edges(data=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(Gm, output_path)

    print(f"[INFO] Combined GML written → {output_path}")

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
    boundaries_dir = graph_base / "boundary"
    partitions_dir.mkdir(parents = True, exist_ok = True)
    boundaries_dir.mkdir(parents = True, exist_ok = True)

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
    out_gml = boundaries_dir / f"{module}_with_boundaries.gml"

    find_boundaries(adjlist_path, partition_graph_dir, out_gml)

    # #  Step 5 – Merge partition GML + boundary GML
    # partition_gml = partitions_dir / f"{module}.gml"
    gml_candidates = list(partitions_dir.glob("*.gml"))
    partition_gml = gml_candidates[0]
    boundary_gml  = boundaries_dir / f"{module}_with_boundaries.gml"
    combined_gml  = (graph_base / f"{module}_combined.gml")

    print("[INFO] Step 5: Merge partition + boundary graphs")
    merge_partition_and_boundary(partition_gml, boundary_gml, combined_gml)



if __name__ =="__main__":
    process_verilog(Path(sys.argv[1]))