import networkx as nx

def check_gml_dag(gml_path):
    G = nx.read_gml(gml_path)
    
    if not G.is_directed():
        G = G.to_directed()
        print("Note: graph was undirected, converted to directed")

    print(f"Nodes    : {G.number_of_nodes()}")
    print(f"Edges    : {G.number_of_edges()}")
    
    is_dag = nx.is_directed_acyclic_graph(G)  # stops at first cycle, no list
    
    if is_dag:
        print("IS a DAG — no cycles found")
    else:
        print("NOT a DAG — has cycles")
        # find just ONE cycle cheaply instead of all of them
        try:
            cycle = nx.find_cycle(G)
            print(f"Example cycle: {cycle}")
        except:
            pass
    
    return is_dag

check_gml_dag("new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_key_expand_128_combined_m1.gml")