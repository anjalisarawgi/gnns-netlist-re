import networkx as nx
import os

# === Paths ===
gml_with_partition = "test/test_partition.gml"
gml_with_boundary = "test/test_boundary.gml"
output_merged = "test/combined.gml"

# --- Helper: clean labels like "'FOO'" -> "FOO"
def clean_label(label: str):
    if isinstance(label, str):
        s = label.strip()
        # strip exactly one leading/trailing single-quote if present
        if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
            s = s[1:-1]
        return s
    return label

# === Load graphs, KEY BY NODE ID ===
# destringizer=int ensures node ids are ints even if read as strings
G_boundary = nx.read_gml(gml_with_boundary, label="id", destringizer=int)
G_partition = nx.read_gml(gml_with_partition, label="id", destringizer=int)

# Keep direction consistent
directed = G_boundary.is_directed() or G_partition.is_directed()
G_merged = nx.DiGraph() if directed else nx.Graph()

# === Merge nodes by numeric id ===
all_ids = set(G_boundary.nodes()) | set(G_partition.nodes())

for nid in all_ids:
    attrs = {}
    if nid in G_boundary:
        attrs.update(G_boundary.nodes[nid])     # boundary attrs
    if nid in G_partition:
        attrs.update(G_partition.nodes[nid])    # partition attrs

    # normalize label if present
    if "label" in attrs:
        attrs["label"] = clean_label(attrs["label"])

    G_merged.add_node(nid, **attrs)

# === Merge edges (union) ===
G_merged.add_edges_from(G_boundary.edges(data=True))
G_merged.add_edges_from(G_partition.edges(data=True))

# === Save ===
os.makedirs(os.path.dirname(output_merged), exist_ok=True)
nx.write_gml(G_merged, output_merged)

print(f"✅ Merged graph saved to: {output_merged}")
