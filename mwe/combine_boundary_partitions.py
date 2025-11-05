import networkx as nx
import os

# === Paths ===
gml_with_partition = "../graphs/raw/des_latest/osu035/des_gephi.gml"
gml_with_boundary = "boundary_graphs/des_latest/osu035/des.gml"
output_merged = "boundary_partitions_graphs/des_latestt/osu035/des_com.gml"

# --- Helper: clean labels like "'FOO'" -> "FOO"
def clean_label(label):
    if isinstance(label, str):
        s = label.strip()
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            return s[1:-1]
        return s
    return label

# === Load keyed by node ID ===
G_boundary  = nx.read_gml(gml_with_boundary,  label="id", destringizer=int)
G_partition = nx.read_gml(gml_with_partition, label="id", destringizer=int)

# === Clean labels but keep original copies ===
for G in (G_boundary, G_partition):
    for _, data in G.nodes(data=True):
        if "label" in data:
            data["label_copy"] = data["label"]  # preserve original
            data["label"] = clean_label(data["label"])  # clean for display/merge

# === Create merged graph (preserve direction) ===
directed = G_boundary.is_directed() or G_partition.is_directed()
G_merged = nx.DiGraph() if directed else nx.Graph()

# === Merge nodes by ID ===
all_ids = set(G_boundary.nodes()) | set(G_partition.nodes())
for nid in all_ids:
    attrs = {}

    # Start with boundary attrs (keeps boundary label)
    if nid in G_boundary:
        attrs.update(G_boundary.nodes[nid])

    # Add partition attrs but don't override label
    if nid in G_partition:
        for k, v in G_partition.nodes[nid].items():
            if k == "label":
                continue
            attrs[k] = v

    G_merged.add_node(nid, **attrs)

# === Merge edges (union) ===
G_merged.add_edges_from(G_boundary.edges(data=True))
G_merged.add_edges_from(G_partition.edges(data=True))

# === Save ===
os.makedirs(os.path.dirname(output_merged), exist_ok=True)
nx.write_gml(G_merged, output_merged)

print(f"✅ Merged graph saved to: {output_merged}")