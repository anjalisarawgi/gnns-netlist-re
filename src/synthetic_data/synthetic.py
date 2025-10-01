import networkx as nx 
import matplotlib.pyplot as plt 
import copy 
import random
import math
import os 


set_seed = 42


def _get_rng(seed=None):
    return random.Random(seed) if seed is not None else random

gml_file_path = "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml"

G = nx.read_gml(gml_file_path)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("number of nodes:", num_nodes)
print("number of edges:", num_edges)

def random_netIDError(G, error_rate = 0.01, seed=None): # 0.01 = 1%
    rng = _get_rng(seed)
    G_error = copy.deepcopy(G)
    edges = list(G_error.edges())
    nodes = list(G_error.nodes())
    num_errors = max(1, int(len(edges) * error_rate))
    edges_to_modify = rng.sample(edges, num_errors)

    for u,v in edges_to_modify:
        # step 1: remove the edge (u,v)
        if G_error.has_edge(u,v):
            G_error.remove_edge(u,v)

            # step 2: assign it to a new random destination node
            new_v = rng.choice(nodes)
            while new_v == u or G_error.has_edge(u, new_v): # -- repeat until the condition is met
                new_v = rng.choice(nodes)
            G_error.add_edge(u, new_v, is_error = 1)

    # annotate error for gephi visualization 
    for u,v in G_error.edges():
        if 'is_error' not in G_error[u][v]:
            G_error[u][v]['is_error'] = 0

    print(f"Removed {num_errors} edges as errors")
    return G_error

def layout_netIDError(G, error_rate = 0.01, layout_radius = 5, seed=None):
    rng = _get_rng(seed)
    G_error = copy.deepcopy(G)
    edges = list(G_error.edges())
    nodes = list(G_error.nodes())
    node_ids = {n: i for i, n in enumerate(nodes)}
    num_errors = max(1, int(len(edges)* error_rate))

    edges_to_modify = rng.sample(edges, num_errors)
    

    for u,v in edges_to_modify:
        if G_error.has_edge(u,v):
            G_error.remove_edge(u,v)

            # find candidates for nearby nodes
            nearby = [n for n in nodes if abs(node_ids[n] - node_ids[u]) <= layout_radius and n != u] ###

            if nearby:
                new_v = rng.choice(nearby)
            else:
                print("No nearby nodes found, selecting random node")
                new_v = rng.choice(nodes)
            
            while new_v == u or G_error.has_edge(u, new_v): # -- repeat until the condition is met
                new_v = rng.choice(nodes)
            G_error.add_edge(u, new_v, is_error = 1)    

    # annotate error for gephi visualization
    for u,v in G_error.edges():
        if 'is_error' not in G_error[u][v]:
            G_error[u][v]['is_error'] = 0

    print(f"Removed {num_errors} edges as errors")
    return G_error

# nodes removed from random number of locations and of random sizes (very confusing!! pls chekc once)
def speckled_gateIDError(G, error_rate=0.01, min_area=2, max_area=5, seed=None):
    rng = _get_rng(seed)
    G_annotated = copy.deepcopy(G)
    nodes = list(G_annotated.nodes())
    total_to_mark = max(1, int(len(nodes) * error_rate))

    marked = set()
    attempts = 0
    max_attempts = 10 * total_to_mark

    while len(marked) < total_to_mark and attempts < max_attempts:
        seed_node = rng.choice(nodes)
        if seed_node in marked:
            attempts += 1
            continue

        cluster_size = rng.randint(min_area, max_area)
        cluster = list(nx.bfs_tree(G_annotated, source=seed_node, depth_limit=2).nodes())[:cluster_size]

        for n in cluster:
            if n not in marked and G_annotated.has_node(n):
                G_annotated.nodes[n]['is_error'] = 1
                marked.add(n)
                if len(marked) >= total_to_mark:
                    break

        attempts += 1

    for n in G_annotated.nodes():
        if 'is_error' not in G_annotated.nodes[n]:
            G_annotated.nodes[n]['is_error'] = 0

    print(f"Removed {len(marked)} nodes as errors")
    return G_annotated    
    





####### each file is saved with a different name 
G_error_random_netIDError = random_netIDError(G, error_rate=0.01, seed=set_seed)
G_error_layout_netIDError = layout_netIDError(G, error_rate=0.01, layout_radius=3, seed=set_seed + 1)
G_error_speckeled_gateIDError = speckled_gateIDError(G, error_rate=0.01, min_area=2, max_area=5, seed=set_seed + 2)

nx.write_gml(G_error_random_netIDError, "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi_randomNetIDError.gml")
nx.write_gml(G_error_layout_netIDError, "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi_layoutNetIDError.gml")
nx.write_gml(G_error_speckeled_gateIDError, "graphs/processed/aes_encryption_latest/osu035/aes_cipher_top_gephi_speckledGateIDError.gml")

print("total nodes modified in G_error_random_netIDError :", G_error_random_netIDError.number_of_nodes())
print("total edges modified in G_error_layout_netIDError", G_error_random_netIDError.number_of_edges())
print("total nodes modified in G_error_layout_netIDError :", G_error_layout_netIDError.number_of_nodes())
