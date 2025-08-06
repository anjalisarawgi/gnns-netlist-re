import networkx as nx
from networkx.algorithms import isomorphism
from tqdm import tqdm

def extract_gate_type(label):
    if not label:
        return "UNKNOWN"
    label = label.strip()
    if label.startswith("INPUT"):
        return "INPUT"
    elif label.startswith("OUTPUT"):
        return "OUTPUT"
    part = label.split('_')[0]
    core = ''.join([c for c in part if not c.isdigit()]).replace('X', '')
    return core.upper()

def get_gate_category(gate_type):
    if gate_type == "INPUT":
        return "INPUT"
    elif gate_type == "OUTPUT":
        return "OUTPUT"
    else:
        return "GATE"


def load_graph(path):
    G = nx.read_gml(path)
    for node in G.nodes:
        raw_label = str(node)
        label_attr = G.nodes[node].get("label", raw_label)
        clean_label = label_attr.strip().replace("\\", "")  # clean the backslashes
        gate_type = extract_gate_type(clean_label)

        label = clean_label

        # input nodes
        if clean_label == "INPUT_clk":
            gate_category = "INPUT_clk"
        elif clean_label ==  "INPUT_kld": 
            gate_category = "INPUT_kld"
        elif clean_label == "INPUT_ld":
            gate_category = "INPUT_ld"
        elif clean_label == "INPUT_rst":
            gate_category = "INPUT_rst"
        elif clean_label.startswith("INPUT_"):
            lowered = clean_label.lower()
            if "key" in lowered:
                gate_category = "INPUT_key"
            elif "text" in lowered:
                gate_category = "INPUT_text"
            else:
                gate_category = "INPUT"
        # output nodes
        elif clean_label == "OUTPUT_done":
            gate_category = "OUTPUT_done"
        elif clean_label.startswith("INPUT_"):
            gate_category = "INPUT"
        elif clean_label.startswith("OUTPUT_"):
            gate_category = "OUTPUT"
        else:
            gate_category = get_gate_category(gate_type)

        G.nodes[node]['label'] = label
        G.nodes[node]['gate_type'] = gate_type
        G.nodes[node]['gate_category'] = gate_category
    return G

def save_graph(G, path):
    nx.write_gml(G, path)

if __name__ == "__main__":
    big_graph_path = "data/aes_core/automated_gscl45nm_aes_core_latest_final_aes_key_expand_128.gml"
    output_path = "data/aes_core/annotated/key_expand_128_annotated.gml"
    G = load_graph(big_graph_path)
    save_graph(G, output_path)