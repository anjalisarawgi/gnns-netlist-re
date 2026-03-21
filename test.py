import networkx as nx
from collections import Counter


G = nx.read_gml("new_graphs_crypto_scp/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_inv_cipher_top_combined_m1.gml")
print("Weakly connected:", nx.is_weakly_connected(G))
print("Number of components:", nx.number_weakly_connected_components(G))
isolated = list(nx.isolates(G))
print("Number of isolated nodes:", len(isolated))


def extract_gate(label):
    if label is None:
        return "UNKNOWN"
    
    label = label.strip("'\"")

    return label.split("_")[0]  # e.g. DFFSR_1 → DFFSR

def normalize_gate(g):
    if g.startswith("INPUT"):
        return "INPUT"
    if g.startswith("OUTPUT"):
        return "OUTPUT"
    if g.startswith("DFF"):
        return "DFF"
    if g.startswith("AND"):
        return "AND"
    if g.startswith("OR"):
        return "OR"
    if g.startswith("NAND"):
        return "NAND"
    if g.startswith("NOR"):
        return "NOR"
    if g.startswith("XOR"):
        return "XOR"
    if g.startswith("XNOR"):
        return "XNOR"
    if g.startswith("INV"):
        return "INV"
    if g.startswith("AOI"):
        return "AOI"
    if g.startswith("OAI"):
        return "OAI"
    if g.startswith("MUX"):
        return "MUX"
    if g.startswith("UNKNOWN"):
        return "UNKNOWN"
    return g

gate_types = []
total_gates = G.number_of_nodes()
for _, data in G.nodes(data=True):
    gate = extract_gate(data.get("label_copy"))
    gate = normalize_gate(gate)
    gate_types.append(gate)

counts = Counter(gate_types)

print("\nTop gate types:")
for g, c in counts.most_common(15):
    print(f"{g}: {c} -- {(c/total_gates)*100}")


import matplotlib.pyplot as plt
# sort alphab
sorted_gates = sorted(counts.items(), key=lambda x: x[0])

labels = [g for g, _ in sorted_gates]
values = [c for _, c in sorted_gates]

plt.figure()
plt.bar(labels, values)

plt.xlabel("Gate Type")
plt.ylabel("Frequency")
plt.title("Gate type distribution (gscl45nm)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("misc/inv_top_gate_type_dist_gscl45nm.png")
