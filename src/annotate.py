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
        label = str(node)
        gate_type = extract_gate_type(label)
        G.nodes[node]['label'] = label
        G.nodes[node]['gate_type'] = gate_type
        G.nodes[node]['gate_category'] = get_gate_category(gate_type)
    return G

def save_graph(G, path):
    nx.write_gml(G, path)

if __name__ == "__main__":
    big_graph_path = "data/aes_core/automated_gscl45nm_aes_core_latest_final_aes_sbox.gml"
    output_path = "data/aes_core/aes_cipher_sbox_annotated.gml"
    G = load_graph(big_graph_path)
    save_graph(G, output_path)