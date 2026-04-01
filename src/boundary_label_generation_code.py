import os 
import subprocess
from pathlib import Path
import sys
import glob
import networkx as nx
import pandas as pd
from tqdm import tqdm

verliog_root = Path("shared/netlist")
partition_root = Path("shared/partition")

adj_root = Path("shared/adjlist")
graph_root = Path("shared/graphs/rawv3")
output_files_root = Path("shared/")
partition_graph_root = Path("outputFiles")
# LIB = "lib/osu035_stdcells.lib"
# LIB = "lib/NangateOpenCellLibrary_functional.lib"
# LIB = "gscl45nm_2.lib"
LIB_MAP = {
    "osu035": "shared/osu035_stdcells.lib",
    "nangate": "shared/NangateOpenCellLibrary_functional.lib",
    "gscl45nm": "shared/gscl45nm_2.lib"
}

def run(cmd):
    print(">>", " ".join(map(str,cmd)))
    subprocess.run(cmd, check=True)

def analyze_missing_boundary(gml_path, tag="", G=None):
    if G is None:
        G = nx.read_gml(gml_path, label="id")

    total = G.number_of_nodes()
    missing_nodes = []
    for nid, d in G.nodes(data=True):
        if "boundary" not in d:
            missing_nodes.append((nid, d))

    missing = len(missing_nodes)
    with_boundary = total - missing

    # categorize missing nodes by gate type
    io_count = 0
    dff_count = 0
    other_count = 0
    gate_type_dist = {}

    for nid, d in missing_nodes:
        # label = str(d.get("label", d.get("label_copy", ""))).strip("'\"").upper()
        label = str(d.get("label_copy", d.get("label", ""))).strip("'\"").upper()
        
        # extract gate type prefix (e.g. "DFFPOSX1_31" -> "DFFPOSX1")
        gate_prefix = label.rsplit("_", 1)[0] if "_" in label else label
        gate_type_dist[gate_prefix] = gate_type_dist.get(gate_prefix, 0) + 1

        if "INPUT" in label or "OUTPUT" in label:
            io_count += 1
        elif "DFF" in label or "LATCH" in label:
            dff_count += 1
        else:
            other_count += 1

    print(f"\n[{tag}] Boundary Analysis")
    print(f"  Total nodes        : {total}")
    print(f"  Nodes with boundary: {with_boundary}")
    print(f"  Missing boundary   : {missing}")
    print(f"    - INPUT/OUTPUT   : {io_count}")
    print(f"    - DFF/LATCH      : {dff_count}")
    print(f"    - Other gates    : {other_count}")
    
    if gate_type_dist:
        print(f"  Gate type breakdown:")
        for gt, cnt in sorted(gate_type_dist.items(), key=lambda x: -x[1])[:15]:
            print(f"    {gt}: {cnt}")

    return missing

def find_boundaries_method1(adjlist_path, partition_dir, out_gml):
    print(f"[M1] Loading design from {adjlist_path}")
    design = nx.read_adjlist(adjlist_path, create_using=nx.DiGraph())
    print(f"[M1] Design: {design.number_of_nodes()} nodes, {design.number_of_edges()} edges")

    boundary_dict = {}
    partition_files = glob.glob(os.path.join(partition_dir, "*.pq"))

    # for f in partition_files:
    for f in tqdm(partition_files, desc="[M1] Partitions", unit="file"):
        # if "@top" in f:
        #     continue

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
            # boundary_dict[node] = is_boundary

            # # # # if slow?
            in_deg = subgraph.in_degree(node)
            out_deg = subgraph.out_degree(node)
            is_boundary = int(in_deg == 0 or out_deg == 0)
            boundary_dict[node] = is_boundary

    # for node in design.nodes():
    #     if node not in boundary_dict:
    #         boundary_dict[node] = -1

   
    # label INPUT/OUTPUT nodes that weren't in any partition as boundary=0
    for node in design.nodes():
        if node not in boundary_dict:
            label = str(node).upper()
            if "INPUT" in label or "OUTPUT" in label:
                boundary_dict[node] = 0
    nx.set_node_attributes(design, boundary_dict, "boundary")
    nx.write_gml(design, out_gml)
    print(f"[M1] Saved to {out_gml}")

def report_boundary_coverage(design, boundary_dict, tag=""):
    total_nodes = design.number_of_nodes()
    labeled_nodes = len(boundary_dict)
    missing_nodes = total_nodes - labeled_nodes

    print(f"[{tag}] Boundary coverage:")
    print(f"  Total nodes     : {total_nodes}")
    print(f"  Labeled nodes   : {labeled_nodes}")
    print(f"  Missing nodes   : {missing_nodes}")
    if missing_nodes == 0:
        print(f"yes! All nodes have boundary labels.")
    else:
        print(f"nope ::: {missing_nodes} nodes are missing boundary labels.")

def check_fully_connected(gml_path, name="", G=None):
    if G is None:
        G = nx.read_gml(gml_path, label="id")

    if isinstance(G, nx.DiGraph):
        connected = nx.is_weakly_connected(G)
    else:
        connected = nx.is_connected(G)

    if connected:
        print(f"[CHECK] {name}: yes — fully connected")
    else:
        print(f"[CHECK] {name}: no — not fully connected")


def merge_partition_and_boundary(gml_partition, gml_boundary, out_gml):
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
    
    # LIB 
    LIB = LIB_MAP[library]
    print(f"[INFO] Using library: {LIB}")

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
    print("[DEBUG] Checking boundary GML BEFORE merge...")
    analyze_missing_boundary(out_gml_1, tag="BEFORE MERGE (Boundary GML)")

    # debugging
    print("[DEBUG] Checking boundary GML BEFORE merge...")
    Gb = nx.read_gml(out_gml_1, label="id")

    total_nodes = Gb.number_of_nodes()
    boundary_nodes = sum(1 for _, d in Gb.nodes(data=True) if "boundary" in d)
    missing_nodes = total_nodes - boundary_nodes

    print(f"[DEBUG] Boundary GML stats:::::::::::::")
    print(f"  Total nodes     : {total_nodes}")
    print(f"  Nodes w/ label  : {boundary_nodes}")
    print(f"  Missing labels  : {missing_nodes}")



    # Step 5: Merge partitions with boundary_1 and boundary_2

    # Find the partition GML (partition2gephi created multiple files, pick the module's one)
    partition_gml_candidates = list(partitions_dir.glob(f"{module}*_gephi.gml"))
    if len(partition_gml_candidates) == 0:
        raise FileNotFoundError(f"No partition GML found in {partitions_dir}")
    partition_gml = partition_gml_candidates[0]

    combined_m1 = graph_base / f"{module}_combined_m1.gml"

    print("[INFO] Step 5: Merge partition + boundary_1")
    merge_partition_and_boundary(partition_gml, out_gml_1, combined_m1)
    
    Gm = nx.read_gml(combined_m1, label="id")

    print("[DEBUG] Checking AFTER merge...")
    analyze_missing_boundary(combined_m1, tag="AFTER MERGE (Combined GML)", G=Gm)

    print(f"Combined M1: {Gm.number_of_nodes():,} nodes, {Gm.number_of_edges():,} edges  →  {combined_m1}")

    check_fully_connected(combined_m1, "Combined M1", G=Gm)


    import json
    from datetime import datetime

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verilog_file": str(verilog_file),
        "design_family": prefix,
        "library": library,
        "module": module,
        "before_merge": {
            "gml_path": str(out_gml_1),
            "total_nodes": total_nodes,
            "nodes_with_boundary": boundary_nodes,
            "missing_boundary": missing_nodes,
        },
        "after_merge": {
            "gml_path": str(combined_m1),
        },
        "partition_gml": str(partition_gml),
        "combined_gml": str(combined_m1),
    }

    # recompute after-merge stats for JSON
    # Gm = nx.read_gml(combined_m1, label="id")
    after_missing = []
    after_io = 0
    after_dff = 0
    after_other = 0
    gate_dist = {}

    for nid, d in Gm.nodes(data=True):
        if "boundary" not in d:
            after_missing.append(nid)
            label = str(d.get("label_copy", d.get("label", ""))).strip("'\"").upper()
            gate_prefix = label.rsplit("_", 1)[0] if "_" in label else label
            gate_dist[gate_prefix] = gate_dist.get(gate_prefix, 0) + 1

            if "INPUT" in label or "OUTPUT" in label:
                after_io += 1
            elif "DFF" in label or "LATCH" in label:
                after_dff += 1
            else:
                after_other += 1

    meta["after_merge"].update({
        "total_nodes": Gm.number_of_nodes(),
        "total_edges": Gm.number_of_edges(),
        "nodes_with_boundary": Gm.number_of_nodes() - len(after_missing),
        "missing_boundary": len(after_missing),
        "missing_input_output": after_io,
        "missing_dff_latch": after_dff,
        "missing_other": after_other,
        "missing_gate_type_breakdown": gate_dist,
        "pct_missing": round(len(after_missing) / Gm.number_of_nodes() * 100, 4),
    })

    # boundary label distribution
    boundary_dist = {}
    for _, d in Gm.nodes(data=True):
        val = d.get("boundary", "missing")
        boundary_dist[str(val)] = boundary_dist.get(str(val), 0) + 1
    meta["after_merge"]["boundary_distribution"] = boundary_dist

    all_labeled = len(after_missing) == 0
    meta["after_merge"]["all_nodes_labeled"] = all_labeled

    if all_labeled:
        print(f"[ALLOK] {module}: all {Gm.number_of_nodes()} nodes have boundary labels")
    else:
        print(f"[!NOTOKAY] {module}: {len(after_missing)} nodes still missing boundary labels")
        
    meta_path = combined_m1.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved metadata → {meta_path}")

    

if __name__ =="__main__":
    process_verilog(Path(sys.argv[1]))