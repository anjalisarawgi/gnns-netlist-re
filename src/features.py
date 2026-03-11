import os
import csv
import networkx as nx
import numpy as np

# ------------------------------------------------------------
# Libraries
# ------------------------------------------------------------

LIBS = ["osu035", "nangate", "gscl45nm"]
# LIBS = ["osu035"]

BASE_RAW = "graphs/raw_v2/raw/sha1-master"
BASE_OUT = "new_graphs_crypto/gnn-re-processed/sha1-master"

FILENAME = "sha1_core_combined_m1.gml"

# ------------------------------------------------------------
# Gate types
# ------------------------------------------------------------

gate_types = [
    "INPUT","OUTPUT",
    "XOR","XNOR",
    "AND","OR",
    "NAND","NOR",
    "INV","BUF",
    "AOI","OAI",
    "DFF","MUX"
]

gate2idx = {g:i for i,g in enumerate(gate_types)}

def extract_gate_type(label):
    base = label.split("_")[0]
    for gate in gate_types:
        if gate in base:
            return gate
    return "UNKNOWN"

# ------------------------------------------------------------
# AES subcircuit mapping
# ------------------------------------------------------------

PARTITION_TO_NAME = {

    "@top": "sha1_core",
    "top": "sha1_core",

    "top+w_mem_inst": "sha1_w_mem",
    # "top+u_CRYPTO_PATH": "CRYPTO_PATH",

    # "@top+u_KEY_SCHED": "KEY_SCHED",  
    # "top+u_CONTROL": "CONTROL",
    # "top+u_CRYPTO_PATH+u_RF":"RF",
    # "top+u_CRYPTO_PATH+u_WF": "WF", 
    # "top+u_KEY_SCHED": "KEY_SCHED",
    # "top+u_KEY_SCHED+u_SKG":"SKG",
    # "top+u_KEY_SCHED+u_WKG": "WKG",



    # "@top+u0": "aes_key_expand_128",
    # "top+u0": "aes_key_expand_128", 

    # "top+r0":"aes_rcon",

    # "top+u0+u0": "aes_sbox",
    # "top+u0+u1": "aes_sbox",
    # "top+u0+u2": "aes_sbox",
    # "top+u0+u3": "aes_sbox",
}

INPUT_OUTPUT_PARTITIONS = {"UNKNOWN", "unknown", "", "input", "output"}

# def assign_subcircuit_name(partition, label):
#     # known partition → use mapped name
#     if partition in PARTITION_TO_NAME:
#         return PARTITION_TO_NAME[partition]
    
#     # INPUT/OUTPUT gates → unknown
#     gate_type = extract_gate_type(label)
#     if gate_type in ("INPUT", "OUTPUT"):
#         return "unknown"
    
#     # everything else → aes_sbox
#     return "aes_sbox"

def assign_subcircuit_name(partition, label):
    # INPUT/OUTPUT gates → unknown
    gate_type = extract_gate_type(label)
    if gate_type in ("INPUT", "OUTPUT"):
        return "unknown"

    last = partition.split("+")[-1]  # get the deepest level

    # aes_sbox: S_0..S_3 (key expansion + final round sboxes)
    #           s0, s4   (round sboxes — plain S and xS variant)
    if last in ("S_0", "S_1", "S_2", "S_3", "s0", "s4"):
        return "aes_sbox"

    # S4: groups of 4 sboxes
    if last.startswith("S4"):
        return "aes_sbox"  # or "aes_S4" if you want a separate class

    # T: table lookup leaf (contains S + xS)
    if last in ("t0", "t1", "t2", "t3"):
        return "aes_table_lookup"

    # one_round / final_round
    if last.startswith("r") and last[1:].isdigit():
        return "aes_one_round"
    if last == "rf":
        return "aes_final_round"

    # expand_key_128 submodules (a1..a10)
    if last.startswith("a") and last[1:].isdigit():
        return "aes_key_expand_128"

    # top level
    if partition in ("top", "@top"):
        return "aes_key_expand_128"

    # everything else
    return "aes_key_expand_128"
# ------------------------------------------------------------
# Process each library
# ------------------------------------------------------------

for lib in LIBS:

    INPUT_GML = os.path.join(BASE_RAW, lib, FILENAME)
    OUT_DIR = os.path.join(BASE_OUT, lib)

    os.makedirs(OUT_DIR, exist_ok=True)

    OUTPUT_GML = os.path.join(OUT_DIR, FILENAME)

    print("Processing:", INPUT_GML)

    G = nx.read_gml(INPUT_GML)
    G = G.to_directed()

    unknown_gates = []
    log_lines = []

    # --------------------------------------------------------
    # Node feature construction
    # --------------------------------------------------------

    for node in G.nodes():

        raw_label = G.nodes[node].get("label_copy", "")
        label = str(raw_label).strip("'")

        raw_partition = G.nodes[node].get("partition","")
        partition_cleaned = str(raw_partition).strip("'")

        gate_type = extract_gate_type(label)

        # gate one-hot
        onehot = [0]*len(gate_types)

        if gate_type in gate2idx:
            onehot[gate2idx[gate_type]] = 1
        else:
            if label != "":
                base = label.split("_")[0]
                unknown_gates.append(base)

        # 2-hop gate counts
        gate_counts_2hop = [0]*len(gate_types)

        neighbors_1 = set(G.predecessors(node)) | set(G.successors(node))

        neighbors_2 = set()
        for n in neighbors_1:
            neighbors_2 |= set(G.predecessors(n))
            neighbors_2 |= set(G.successors(n))

        neighbors = neighbors_1 | neighbors_2

        for n in neighbors:

            nlabel = str(G.nodes[n].get("label",""))
            ngate = extract_gate_type(nlabel)

            if ngate in gate2idx:
                gate_counts_2hop[gate2idx[ngate]] += 1

        # PI / PO connections
        pi_connections = 0
        po_connections = 0

        for n in G.predecessors(node):
            if "INPUT" in str(G.nodes[n].get("label","")):
                pi_connections += 1

        for n in G.successors(node):
            if "OUTPUT" in str(G.nodes[n].get("label","")):
                po_connections += 1

        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)

        is_sequential = int(gate_type == "DFF")

        G.nodes[node]['features'] = (
            onehot +
            gate_counts_2hop +
            [pi_connections, po_connections, indeg, outdeg, is_sequential]
        )

        # G.nodes[node]['subcircuit_name'] = assign_subcircuit_name(partition_cleaned)
        G.nodes[node]['subcircuit_name'] = assign_subcircuit_name(partition_cleaned, label)


    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    nx.write_gml(G, OUTPUT_GML)
    print("Saved:", OUTPUT_GML)

    # --------------------------------------------------------
    # Unknown gate log
    # --------------------------------------------------------

    if unknown_gates:

        counts = {}

        for g in unknown_gates:
            counts[g] = counts.get(g,0) + 1

        log_path = os.path.join(OUT_DIR,"unknown_gate_types.csv")

        with open(log_path,"w",newline="") as f:

            writer = csv.writer(f)
            writer.writerow(["gate_type","count"])

            for g,c in sorted(counts.items(), key=lambda x:-x[1]):
                writer.writerow([g,c])

        print("Unknown gate types saved:", log_path)

    else:
        print("No unknown gate types found")