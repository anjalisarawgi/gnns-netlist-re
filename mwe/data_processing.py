import os 
import subprocess
from pathlib import Path
import sys

verliog_root = Path("tum-eisec-benchmarks-main/netlist")
partition_root = Path("tum-eisec-benchmarks-main/partition")

adj_root = Path("adjlist")
graph_root = Path("graphs/raw")
output_files_root = Path("")
LIB = "lib/osu035_stdcells.lib"


def run(cmd):
    print(">>", " ".join(map(str,cmd)))
    subprocess.run(cmd, check=True)

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
    partition_out_dir = output_files_root / prefix / library 
    graphs_out_dir = graph_root / prefix / library 

    adj_out_dir.mkdir(parents = True, exist_ok=True)
    # partition_out_dir.mkdir(parents = True, exist_ok=True)
    graphs_out_dir.mkdir(parents = True, exist_ok=True)

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
        "-o", str(graphs_out_dir),  
        "-p", f"outputFiles/{prefix}/{library}/partition_graph",
        "--cores", "1"

    ])



   
if __name__ =="__main__":
    process_verilog(Path(sys.argv[1]))