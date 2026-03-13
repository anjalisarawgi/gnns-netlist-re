import networkx as nx

def check_gml_dag(gml_path):
    G = nx.read_gml(gml_path)
    
    print(f"Graph type: {'Directed' if G.is_directed() else 'Undirected'}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    is_dag = nx.is_directed_acyclic_graph(G) # this is networkx \method to check if a grpah is  a dag or not
    
    if is_dag:
        print("it is a DAG — no cycles found")
    else:
        print("it is not a DAG — has cycles")
        cycle = nx.find_cycle(G, orientation="original")
        print(f"  Example cycle: {cycle}")
    
    return is_dag

check_gml_dag("new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_key_expand_128_combined_m1.gml") # aes_sbox_combined_m1   aes_key_expand_128_combined_m1