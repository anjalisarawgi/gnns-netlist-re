import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
import random
from sklearn.model_selection import train_test_split
import os
import numpy as np

class SubgraphDataset(InMemoryDataset):
    def __init__(self, root, subgraphs=None, transform=None, pre_transform=None):
        self.subgraphs = subgraphs
        super().__init__(root, transform, pre_transform)
        if subgraphs is not None:
            self.data, self.slices = self.collate(subgraphs)
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data, slices = self.collate(self.subgraphs)
        torch.save((data, slices), self.processed_paths[0])

def extract_subgraphs_from_gml(gml_path):
    G = nx.read_gml(gml_path, label='id')
    node_subcircuit = {}

    for node in G.nodes:
        attr = G.nodes[node]
        if 'subcircuit_id' in attr:
            node_subcircuit[node] = attr['subcircuit_id']

    # Group nodes by subcircuit_id
    subcircuit_to_nodes = {}
    for node, sub_id in node_subcircuit.items():
        if sub_id not in subcircuit_to_nodes:
            subcircuit_to_nodes[sub_id] = []
        subcircuit_to_nodes[sub_id].append(node)

    subgraphs = []
    label_map = {name: idx for idx, name in enumerate(sorted(subcircuit_to_nodes))}

    for sub_id, nodes in subcircuit_to_nodes.items():
        H = G.subgraph(nodes).copy()
        node_map = {n: i for i, n in enumerate(H.nodes())}

        x = []
        for n in H.nodes():
            feats = H.nodes[n]['features']
            if isinstance(feats, list):
                x.append(feats)
            else:
                x.append(list(map(int, feats.strip('[]').split(','))))

        edge_index = []
        for u, v in H.edges():
            edge_index.append([node_map[u], node_map[v]])

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor([label_map[sub_id]], dtype=torch.long)

        subgraphs.append(Data(x=x, edge_index=edge_index, y=y))

    return subgraphs

def sample_random_background_subgraphs(G, num_samples=10, size=20):
    nodes = list(G.nodes())
    subgraphs = []
    for _ in range(num_samples):
        start_node = random.choice(nodes)
        sampled_nodes = list(nx.bfs_tree(G, start_node, depth_limit=2).nodes)[:size]
        H = G.subgraph(sampled_nodes).copy()
        node_map = {n: i for i, n in enumerate(H.nodes())}

        x = []
        for n in H.nodes():
            feats = H.nodes[n]['features']
            if isinstance(feats, list):
                x.append(feats)
            else:
                x.append(list(map(int, feats.strip('[]').split(','))))

        edge_index = []
        for u, v in H.edges():
            edge_index.append([node_map[u], node_map[v]])

        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor([0], dtype=torch.long)  # Label as "not subcircuit"

        subgraphs.append(Data(x=x, edge_index=edge_index, y=y))

    return subgraphs

def create_binary_dataset(gml_path):
    G = nx.read_gml(gml_path, label='id')

    positive_subgraphs = extract_subgraphs_from_gml(gml_path)
    negative_subgraphs = sample_random_background_subgraphs(G, num_samples=len(positive_subgraphs))

    # Label positive as 1
    for sg in positive_subgraphs:
        sg.y = torch.tensor([1], dtype=torch.long)

    all_subgraphs = positive_subgraphs + negative_subgraphs
    random.shuffle(all_subgraphs)

    return all_subgraphs

# Example usage:
if __name__ == "__main__":
    gml_file = "aes_key_expand_128_modified_wfeatures.gml"
    subgraphs = create_binary_dataset(gml_file)

    train_set, test_set = train_test_split(subgraphs, test_size=0.2, random_state=42)

    dataset = SubgraphDataset(root="subgraph_cache", subgraphs=train_set)
    dataset.process()  # Save processed data to disk

    print(f"Prepared {len(dataset)} training subgraphs and saved to disk.")