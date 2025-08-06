import networkx as nx 
import numpy as np 
import os 



def parse_aes_gml(file_path, output_path, output_path_gml):
    G = nx.read_gml(file_path, label='id') # label id

    nodes = sorted(G.nodes(), key=int) # key int
    num_nodes = len(nodes)
    print(f"Number of nodes: {num_nodes}")

    # making feature matrix - [in_degree, out_degree]
    features = np.zeros((num_nodes, 13), dtype=np.int32) # num_nodes x 2 matrix 

    ##### pi and po 
    primary_inputs = set(n for n in G.nodes() if G.in_degree(n) == 0)
    primary_outputs = set(n for n in G.nodes() if G.out_degree(n) == 0)

    GATE_TYPES = ['XOR', 'XNOR', 'AND', 'OR', 'NAND', 'NOR', 'INV', 'BUF']


    for i, node in enumerate(nodes):
        attr = G.nodes[node]
        label = attr.get('label', '').upper()

        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # Base features
        is_pi = int(in_deg == 0)
        is_po = int(out_deg == 0)
        is_key = int('KEY' in label)

        # Gate type features
        gate_type_features = [int(gate in label) for gate in GATE_TYPES]

        # Final vector: [PI, PO, KEY, is_XOR...is_BUF, in_deg, out_deg]
        all_feats = [is_pi, is_po, is_key] + gate_type_features + [in_deg, out_deg]

        features[i] = all_feats
        G.nodes[node]['features'] = [str(f) for f in all_feats]
        
    np.save(output_path, features)
    np.savetxt(output_path, features, fmt='%d', delimiter=',')
    print("saved")

    nx.write_gml(G, output_path_gml)
    print("saved gml")


def load_all_gmls(folder_path):
    gml_files = [f for f in os.listdir(folder_path) if f.endswith('.gml')]

    for gml_file in gml_files:
        file_path = os.path.join(folder_path, gml_file)
        output_path = os.path.join(folder_path, gml_file.replace('.gml', '_features.txt'))
        output_path_gml = os.path.join(folder_path, gml_file.replace('.gml', '_parsed.gml'))
        parse_aes_gml(file_path, output_path, output_path_gml)

if __name__ == "__main__":
    folder_path = "data/aes_core/"
    load_all_gmls(folder_path)


############## TODO : need to have class labels - then parse the gml with class labels --- then train the model 