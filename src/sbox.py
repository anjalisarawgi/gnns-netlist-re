import networkx as nx
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.model_selection import train_test_split


# Load the GML (or simulate small S-box graph here)
G = nx.read_gml("data/aes/automated_nangate_aes_core_latest_final_aes_sbox.gml")  # <-- Replace with your GML filename

# # STEP 1: Label original nodes as subcircuit (label = 1)
# for node in G.nodes():
#     G.nodes[node]['label'] = 1
#     G.nodes[node]['x'] = [1.0]  # dummy node feature (same for all subcircuit nodes)

for node in list(G.nodes):
    if not str(node).startswith("noise_"):
        G.nodes[node]['label'] = 1
        G.nodes[node]['x'] = [1.0] 


# STEP 2: Add 10 synthetic "noise" nodes (label = 0)
existing_ids = set(G.nodes)
i = 0
while f"noise_{i}" in existing_ids:
    i += 1
start_id = i

print(f"Total number of nodes: {G.number_of_nodes()}")

for i in range(500):
    new_id = f"noise_{start_id + i}"
    G.add_node(new_id, label=0, x=[0.0])
    for _ in range(random.randint(1, 3)):
        target = random.choice(list(G.nodes))
        G.add_edge(str(new_id), target)
        # G.add_edge(target, str(new_id))

# STEP 3: Convert to PyTorch Geometric
data = from_networkx(G)
data.x = torch.tensor([G.nodes[n]['x'] for n in G.nodes], dtype=torch.float)
data.y = torch.tensor([G.nodes[n]['label'] for n in G.nodes], dtype=torch.long)
labels_np = data.y.numpy()


num_nodes = data.num_nodes


indices = np.arange(num_nodes)

train_idx, test_idx = train_test_split(
    indices, stratify=labels_np, test_size=0.2, random_state=42
)


# Create boolean masks
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

# STEP 4: Define GCN Model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# STEP 5: Train the GCN
losses = []
for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # loss = criterion(out, data.y)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# STEP 6: Evaluate
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = correct.item() / data.test_mask.sum().item()
print(f"\nTest Accuracy: {acc:.4f}")

labels = [G.nodes[n]['label'] for n in G.nodes]
num_real = sum(1 for n in G.nodes if G.nodes[n]['label'] == 1)
num_noise = sum(1 for n in G.nodes if G.nodes[n]['label'] == 0)
print(f"Real nodes: {num_real}, Noise nodes: {num_noise}, Total: {G.number_of_nodes()}")

# # STEP 7: Plot training loss
# plt.plot(losses)
# plt.title("Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.show()


# # STEP 1: Load and label original nodes
# for node in G.nodes():
#     G.nodes[node]['label'] = 1
#     G.nodes[node]['x'] = [1.0]

# STEP 2: Add noise nodes
# for i in range(1):
#     new_id = f"noise_{i}"
#     G.add_node(new_id, label=0, x=[0.0])
#     for _ in range(random.randint(1, 3)):
#         target = random.choice(list(G.nodes))
#         G.add_edge(new_id, target)
#         G.add_edge(target, new_id)

# ✅ STEP 3: Compute layout *after all nodes exist*
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
    G.nodes[n]['class_label'] = G.nodes[n]['label']  # 0 = noise, 1 = real


# for n in G.nodes():
#     lbl = G.nodes[n]['label']
#     G.nodes[n]['graphics']['fill'] = '#FF0000' if lbl == 0 else '#000000'  # Red = noise, black = real
#     G.nodes[n]['label'] = f"noise_{n}" if lbl == 0 else str(n)  # Use label attribute for Gephi text

nx.write_gml(G, "modified_sbox_with_noise.gml", stringizer=lambda x: x)

print("Noise nodes:")
for n in G.nodes():
    if G.nodes[n]['label'] == 0:
        print(f"  {n}, degree: {G.degree[n]}")



for n in list(G.nodes)[:5]:
    print(n, G.nodes[n])