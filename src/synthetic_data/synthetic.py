import networkx as nx 
import matplotlib.pyplot as plt 
import copy 
import random
import math
import os 
import argparse




def random_netIDError(G, error_rate = 0.01, seed=None): # 0.01 = 1%
    random.seed(set_seed)
    G_error = copy.deepcopy(G)
    edges = list(G_error.edges())
    nodes = list(G_error.nodes())
    num_errors = max(1, int(len(edges) * error_rate))
    edges_to_modify = random.sample(edges, num_errors)

    for u,v in edges_to_modify:
        # step 1: remove the edge (u,v)
        if G_error.has_edge(u,v):
            G_error.remove_edge(u,v)

            # step 2: assign it to a new random destination node
            new_v = random.choice(nodes)
            while new_v == u or G_error.has_edge(u, new_v): # -- repeat until the condition is met
                new_v = random.choice(nodes)
            G_error.add_edge(u, new_v, is_error = 1)

    # annotate error for gephi visualization 
    for u,v in G_error.edges():
        if 'is_error' not in G_error[u][v]:
            G_error[u][v]['is_error'] = 0

    print(f"Removed {num_errors} edges as errors")
    return G_error

def layout_netIDError(G, error_rate = 0.01, layout_radius = 5, seed=None):
    random.seed(set_seed)
    G_error = copy.deepcopy(G)
    edges = list(G_error.edges())
    nodes = list(G_error.nodes())
    node_ids = {n: i for i, n in enumerate(nodes)}
    num_errors = max(1, int(len(edges)* error_rate))

    edges_to_modify = random.sample(edges, num_errors)
    

    for u,v in edges_to_modify:
        if G_error.has_edge(u,v):
            G_error.remove_edge(u,v)

            # find candidates for nearby nodes
            nearby = [n for n in nodes if abs(node_ids[n] - node_ids[u]) <= layout_radius and n != u] ###

            if nearby:
                new_v = random.choice(nearby)
            else:
                print("No nearby nodes found, selecting random node")
                new_v = random.choice(nodes)
            
            while new_v == u or G_error.has_edge(u, new_v): # -- repeat until the condition is met
                new_v = random.choice(nodes)
            G_error.add_edge(u, new_v, is_error = 1)    

    # annotate error for gephi visualization
    for u,v in G_error.edges():
        if 'is_error' not in G_error[u][v]:
            G_error[u][v]['is_error'] = 0

    print(f"Removed {num_errors} edges as errors")
    return G_error

# nodes removed from random number of locations and of random sizes (very confusing!! pls chekc once)
def speckled_gateIDError(G, error_rate=0.01, min_area=2, max_area=5, seed=None):
    random.seed(set_seed)
    G_annotated = copy.deepcopy(G)
    nodes = list(G_annotated.nodes())
    total_to_mark = max(1, int(len(nodes) * error_rate))

    marked = set()
    attempts = 0
    max_attempts = 10 * total_to_mark

    while len(marked) < total_to_mark and attempts < max_attempts:
        seed_node = random.choice(nodes)
        if seed_node in marked:
            attempts += 1
            continue

        cluster_size = random.randint(min_area, max_area)
        cluster = list(nx.bfs_tree(G_annotated, source=seed_node, depth_limit=2).nodes())[:cluster_size]

        for n in cluster:
            if n not in marked and G_annotated.has_node(n):
                #G_annotated.nodes[n]['is_error'] = 1
                G_annotated.remove_node(n)
                marked.add(n)
                if len(marked) >= total_to_mark:
                    break

        attempts += 1


    print(f"Removed {len(marked)} nodes as errors")
    return G_annotated    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply multiple graph error models with random error rates.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graphs/synthetic/raw/aes_encryption_latest/osu035/1",
        help="Directory to save output .gml files"
    )

    parser.add_argument(
        "--error_rate",
        type=float,
    )

    args = parser.parse_args()

    set_seed = 42
    random.seed(set_seed)

    gml_file_path = "graphs/raw/aes_encryption_latest/osu035/aes_cipher_top_gephi.gml"
    G = nx.read_gml(gml_file_path)

    print("number of nodes:", G.number_of_nodes())
    print("number of edges:", G.number_of_edges())


    error_models = {
        "random": random_netIDError,
        "layout": layout_netIDError,
        "speckled": speckled_gateIDError
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, model_fn in error_models.items():
        for i in range(1):  # no. runs per model
            seed = set_seed + i

            if model_name == "random":
                G_error = model_fn(G, error_rate=args.error_rate, seed=seed)
            elif model_name == "layout":
                G_error = model_fn(G, error_rate=args.error_rate, layout_radius=3, seed=seed)
            elif model_name == "speckled":
                G_error = model_fn(G, error_rate=args.error_rate, min_area=2, max_area=5, seed=seed)

            out_file = os.path.join(
                args.output_dir,
                f"aes_cipher_top_gephi_{model_name}Error_{i+1}.gml"
            )

            nx.write_gml(G_error, out_file)
            print(f"Saved: {out_file}")


    # ---- Generate 2 mixed-error graphs ----
    print("\nGenerating mixed-error graphs with shared error rate...\n")

    mixed_error_rate = args.error_rate / 3  # divide total error budget across 3 models

    for i in range(2):
        G_mixed = copy.deepcopy(G)
        seed = set_seed + 100 + i  # offset seed to avoid overlap

        print(f"Run {i+1}: Applying each model with error_rate = {mixed_error_rate:.4f}")

        G_mixed = random_netIDError(G_mixed, error_rate=mixed_error_rate, seed=seed)
        G_mixed = layout_netIDError(G_mixed, error_rate=mixed_error_rate, layout_radius=3, seed=seed)
        G_mixed = speckled_gateIDError(G_mixed, error_rate=mixed_error_rate, min_area=2, max_area=5, seed=seed)

        out_file = os.path.join(
            args.output_dir,
            f"aes_cipher_top_gephi_mixedError_{i+1}.gml"
        )

        nx.write_gml(G_mixed, out_file)
        print(f"Saved mixed error graph to: {out_file}")