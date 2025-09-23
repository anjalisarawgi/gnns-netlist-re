import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------- GATE TYPE SETUP -----------
GATE_TYPES = [
    "INPUT", "OUTPUT", "XOR", "XNOR", "AND", "OR", "NAND", "NOR",
    "INV", "BUF", "AOI", "OAI", "DFF", "MUX"
]
gate2idx = {gate: i for i, gate in enumerate(GATE_TYPES)}

def extract_gate_type(label):
    base = label.strip("'").split("_")[0]
    for gate in GATE_TYPES:
        if gate in base:
            return gate
    return "UNKNOWN"

# ----------- GML to PyG Data -----------
def load_gml_as_pyg(gml_path):
    G = nx.read_gml(gml_path, label="id")
    node_features = []
    node_map = {n: i for i, n in enumerate(G.nodes())}

    for node in G.nodes():
        attr = G.nodes[node]
        label = attr.get("label", "")
        gate_type = extract_gate_type(label)
        
        # One-hot gate type
        onehot = [0] * len(GATE_TYPES)
        if gate_type in gate2idx:
            onehot[gate2idx[gate_type]] = 1
        
        # Structural features
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        fan_io_ratio = indeg / (outdeg + 1)
        clustering_coeff = nx.clustering(G.to_undirected(), node)
        logic_cone_size = len(nx.ancestors(G, node))
        transitive_fanout = len(nx.descendants(G, node))
        partition = attr.get("partition", "")
        hierarchy_depth = partition.count('+')

        # Final feature vector
        feature = onehot + [indeg, outdeg, fan_io_ratio, clustering_coeff,
                            logic_cone_size, transitive_fanout, hierarchy_depth]
        node_features.append(feature)

    edge_index = []
    for src, dst in G.edges():
        edge_index.append([node_map[src], node_map[dst]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data, list(G.nodes()), G

# ----------- GNN Encoder -----------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return self.prelu(x)

# ----------- DGI Training -----------
def train_dgi(data, hidden_dim=64, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = GCNEncoder(data.num_features, hidden_dim)
    dgi = DeepGraphInfomax(
        hidden_channels=hidden_dim,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(dgi.parameters(), lr=0.001)

    for epoch in range(epochs):
        dgi.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = dgi(data.x, data.edge_index)
        loss = dgi.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:04d}, Loss: {loss.item():.4f}")

    dgi.eval()
    z, _, _ = dgi(data.x, data.edge_index)
    return z.cpu().detach().numpy()

# ----------- Clustering & PCA -----------

def cluster_embeddings(embeddings, node_names, G, n_clusters=10):
    print("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    clusters = {i: [] for i in range(n_clusters)}
    for idx, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(node_names[idx])

    for c in clusters:
        print(f"\nCluster {c}: {len(clusters[c])} nodes")
        for n in clusters[c][:10]:  # Show top-10 per cluster
            print("  ", G.nodes[n].get("label", n))

    # 2D plot
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap="tab10", s=10)
    plt.title("Node Embeddings Clustered")
    plt.savefig("clustered_embeddings.png", dpi=300)
    plt.close()

    return cluster_labels

# ----------- Save Annotated GML -----------

def annotate_gml_with_clusters(G, node_names, cluster_labels, output_path):
    print("Annotating GML with cluster labels...")
    for idx, node in enumerate(node_names):
        G.nodes[node]["dgi_cluster"] = int(cluster_labels[idx])
    nx.write_gml(G, output_path)
    print(f"Saved annotated GML with cluster labels to: {output_path}")

# ----------- Main Pipeline -----------

if __name__ == "__main__":
    GML_PATH = "graphs/raw/mips_16_latest/osu035/mips_16_core_top_gephi.gml"

    print("Loading GML...")
    data, node_names, G = load_gml_as_pyg(GML_PATH)

    print("Training DGI encoder...")
    embeddings = train_dgi(data, hidden_dim=64, epochs=2000)

    print("Clustering embeddings...")
    cluster_labels = cluster_embeddings(embeddings, node_names, G, n_clusters=10)

    print("Saving...")
    annotate_gml_with_clusters(G, node_names, cluster_labels, "annotated_clustered.gml")