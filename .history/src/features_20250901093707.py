import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

G = nx.read_gml(gml_path)


# should we make it directed?
# should we create an encoding just for the type?

gate_types = ["XOR", "XNOR", "AND", "OR",  "NAND", "NOR", "INV", "BUF"]
gate2idx = {gate: idx for idx, gate in enumerate(gate_types)}

def extract_gate_type(label):
    base = label.strip("'").split("_")[0]



for node in G.nodes():



    # in degree and out degree
    indeg = G.in_degree(node)
    outdeg = G.out_degree(node)
    G.nodes[node]['features'] = [indeg, outdeg]
    

nx.write_gml(G, "aes_key_expand_128_gephi_wfeatures.gml")