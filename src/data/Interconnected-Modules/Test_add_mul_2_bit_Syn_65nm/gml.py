#!/usr/bin/env python3
import os
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp

def load_txt(path, dtype=str):
    with open(path) as f:
        return [dtype(line.strip()) for line in f if line.strip()]

def main(folder):
    # Load adjacency matrix (full graph)
    adj_path = os.path.join(folder, 'adj_full.npz')
    adj = sp.load_npz(adj_path).tocoo()  # Convert to COO for easy iteration

    feat = np.loadtxt(os.path.join(folder, 'feat.txt'))
    labels = load_txt(os.path.join(folder, 'label.txt'), int)
    cell = load_txt(os.path.join(folder, 'cell.txt'), str)

    G = nx.DiGraph()

    # Add nodes with attributes
    for i, vec in enumerate(feat):
        attrs = {
            "name": cell[i] if i < len(cell) else f"cell_{i}",
            "class_label": labels[i] if labels else -1,
            "features": "[" + ",".join(str(int(v)) for v in vec) + "]"
        }
        G.add_node(i, **attrs)

    # Add edges from adjacency matrix
    for src, dst in zip(adj.row, adj.col):
        G.add_edge(src, dst)

    out_path = os.path.join(folder, 'graph_2.gml')
    nx.write_gml(G, out_path)
    print(f"GML saved to {out_path}")

if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(folder)