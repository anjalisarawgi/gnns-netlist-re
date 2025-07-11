import json

import networkx as nx
import numpy as np

import utils.deprecated


def convert_GNN_RE_graphs():
    cell_file = np.load('A:/IDP/GNN-RE-converted/GNN-RE-main/graph_np/npy_files/cell.npy')
    print(cell_file)

    feat_file = np.load('A:/IDP/GNN-RE-converted/GNN-RE-main/Graphs_datasets/Interconnected-Modules/feats.npy').astype(
        int)
    print(feat_file)

    class_map = json.load(
        open('A:/IDP/GNN-RE-converted/GNN-RE-main/Graphs_datasets/Interconnected-Modules/class_map.json'))

    row_ids = np.loadtxt('A:/IDP/GNN-RE-converted/GNN-RE-main/Graphs_datasets/Interconnected-Modules/row.txt',
                         dtype=int)
    col_ids = np.loadtxt('A:/IDP/GNN-RE-converted/GNN-RE-main/Graphs_datasets/Interconnected-Modules/col.txt',
                         dtype=int)

    edges = zip(row_ids, col_ids)

    graphs = {}

    for node in cell_file:
        graphname = utils.deprecated.split(".")[0]
        if graphname not in graphs:
            graphs[graphname] = nx.Graph()
        node_id = node[0]
        graphs[graphname].add_node(node_id, x=feat_file[int(node_id)], y=int(class_map[node_id]))
        if graphname == "Train_add_mul_1_bit_Syn_65nm":
            print(f"{node_id}: {graphs[graphname].nodes[node_id]}")

    print(graphs)

    for (u, v) in edges:
        for graph in graphs.values():
            if str(u) in graph.nodes and str(v) in graph.nodes:
                graph.add_edge(str(u), str(v))
                break

    return graphs


if __name__ == '__main__':
    graphs = convert_GNN_RE_graphs()
    for name, graph in graphs.items():
        print(f"{name}; nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")
        nx.write_gml(graph, "A:/IDP/2024-idp-lassmann-braun-gnn-re/data/" + name + '.gml')
