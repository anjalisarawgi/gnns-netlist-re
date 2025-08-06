import networkx as nx
import matplotlib.pyplot as plt
import os
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout

# just to undetstand the structure of the GML file

gx = nx.read_gml('data/gscl45nm/automated_gscl45nm_8bit_vedic_multiplier_latest_final_full_adder.gml')
print("Number of nodes:", gx.number_of_nodes())
print("Number of edges:", gx.number_of_edges())

for node in gx.nodes(data=True):
    print("Sample node:", node)
    break

color_map = []
for node in gx.nodes():
    label = node.upper() 
    if label.startswith('INPUT'):
        color_map.append('blue')
    elif label.startswith('OUTPUT'):
        color_map.append('green')
    elif "XOR" in label:
        color_map.append('orange')
    elif "AND" in label:
        color_map.append('yellow')
    elif "OR" in label:
        color_map.append('pink')
    elif "BUF" in label:
        color_map.append('purple')
    else:
        color_map.append('gray')


pos = graphviz_layout(gx, prog='dot')  # for topdown layout
plt.figure(figsize=(12, 12))
nx.draw_networkx_edges(gx, pos, alpha=0.5)
nx.draw_networkx_nodes(gx, pos, node_color=color_map, node_size=300, alpha=0.9)
nx.draw_networkx_labels(
    gx, pos,
    labels={n: n for n in gx.nodes()},
    font_size=8
)

plt.axis('off')
plt.tight_layout()
plt.savefig('src/visualizations/automated_gscl45nm_8bit_vedic_multiplier_latest_final_full_adder.png')
