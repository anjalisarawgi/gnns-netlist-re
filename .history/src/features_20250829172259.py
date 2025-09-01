import os 
import networkx as nx
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import json
import numpy as np


gml_path = "mwe/aes_key_expand_128_gephi.gml"

def make_features(gml_path):
    G = ns.