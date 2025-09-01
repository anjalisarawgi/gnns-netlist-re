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
# should we create an encoding just for the type


for node in G.nodes():



    # in degree and out degree
    indeg = G.in_degree(node)
    outdeg = G.out_degree(node)
    G.nodes[node]['features'] = [indeg, outdeg]
    

nx.write_gml(G, "aes_key_expand_128_gephi_wfeatures.gml")