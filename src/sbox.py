import networkx as nx
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.model_selection import train_test_split


G = nx.read_gml("data/aes_core/automated_gscl45nm_aes_core_latest_final_aes_key_expand_128.gml") 

# labelling original nodes as (label = 1) 
for node in list(G.nodes):
    G.nodes[node]['label'] = 1

total_nodes = len(G.nodes)
print(f"Total nodes in the graph: {total_nodes}")

# adding random nodes to the graph for noise (label = 0)
num_noise_node = 1000
for i in range(num_noise_node):
    new_id = f"noise_{i}"
    G.add_node(new_id, label=0)
    for _ in range(random.randint(1, 3)): # CHANGE THIS BASED ON AVERAGE DEGREE 
        target_node = random.choice(list(G.nodes))
        G.add_edge(new_id, target_node) # adding random edges to noise nodes



# adding feautres to the nodes 
for node in G.nodes:
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    G.nodes[node]['x'] = [in_degree, out_degree]

# print average degree in and degree out 
in_degrees = [G.in_degree(n) for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]

print("Average in-degree:", sum(in_degrees) / len(in_degrees))
print("Average out-degree:", sum(out_degrees) / len(out_degrees))

# convert to pyg
data = from_networkx(G)
labels = [G.nodes[node]['label'] for node in G.nodes] 
features = [G.nodes[node]['x'] for node in G.nodes]

data.y = torch.tensor(labels, dtype = torch.long)
data.x = torch.tensor(features, dtype=torch.float)

# splitting the datasets into train and test sets
num_total_nodes = data.num_nodes
train_idx, test_idx = train_test_split(range(num_total_nodes), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

train_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
val_mask[val_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print("Train nodes:", train_mask.sum().item())
print("Test nodes:", test_mask.sum().item())
print("Validation nodes:", val_mask.sum().item())
# print("total noise nodes:", num_noise_node)

# debugging for checks
labels = [G.nodes[n]['label'] for n in G.nodes]
num_real = sum(1 for n in G.nodes if G.nodes[n]['label'] == 1)
num_noise = sum(1 for n in G.nodes if G.nodes[n]['label'] == 0)
print(f"Real nodes: {num_real}, Noise nodes: {num_noise}, Total: {G.number_of_nodes()}")


###### GCN model  - a simple one 
class gcn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 2) 

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)
    
##### model configs 
model = gcn()
optimizer = torch.optim.Adam (model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
epochs = 100

# trainign the model 
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    # loss = criterion(model(data.x, data.edge_index)[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad(): 
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
            val_total = data.val_mask.sum().item()
            val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch}, Loss: {loss.item()}, Val Accuracy:{val_acc:.4f}")
        

# evaluating the model
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct =(pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = correct / test_mask.sum().item()
print(f"Test Accuracy: {accuracy:.4f}")

pos = nx.spring_layout(G, seed=42)
for n in G.nodes():
    G.nodes[n]['graphics'] = {
        'x': float(pos[n][0]) * 1000,
        'y': float(pos[n][1]) * 1000,
        'w': 10.0,
        'h': 10.0,
        'type': 'ellipse',
        'fill': '#000000'
    }

for n in G.nodes():
    G.nodes[n]['class_label'] = G.nodes[n]['label']


for i, node in enumerate(G.nodes()):
    if data.train_mask[i]:
        G.nodes[node]['split'] = 'train'
    elif data.val_mask[i]:
        G.nodes[node]['split'] = 'val'
    elif data.test_mask[i]:
        G.nodes[node]['split'] = 'test'

for i, node in enumerate(G.nodes()):
    true_label = data.y[i].item()
    predicted_label = pred[i].item()
    G.nodes[node]['predictions'] = predicted_label
    G.nodes[node]['is_correct'] = (true_label == predicted_label) # green = wrong pred

nx.write_gml(G, "automated_gscl45nm_aes_core_latest_final_aes_key_expand_128.gml" )