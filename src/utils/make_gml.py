#!/usr/bin/env python3
import os
import argparse
import networkx as nx
import numpy as np

def load_txt(path, dtype=str):
    with open(path) as f:
        return [dtype(line.strip()) for line in f if line.strip()]

def main(folder):
    row = load_txt(os.path.join(folder, 'row.txt'), int)
    col = load_txt(os.path.join(folder, 'col.txt'), int)
    feat = np.loadtxt(os.path.join(folder, 'feat.txt'))
    labels = load_txt(os.path.join(folder, 'label.txt'), int)
    cell = load_txt(os.path.join(folder, 'cell.txt'), str)

    G = nx.DiGraph()

    for i, vec in enumerate(feat):
        attrs = {
            "name": cell[i] if i < len(cell) else f"cell_{i}",
            "class_label": labels[i] if labels else -1,
            "features": "[" + ",".join(str(int(v)) for v in vec) + "]"
        }
        # for j, val in enumerate(vec):
        #     attrs[f"feat_{j:02d}"] = int(val)
        G.add_node(i, **attrs)

    for src, dst in zip(row, col):
        G.add_edge(src, dst)

    out_path = os.path.join(folder, 'graph.gml')
    nx.write_gml(G, out_path)
    print(f"GML saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    args = parser.parse_args()
    main(args.folder)