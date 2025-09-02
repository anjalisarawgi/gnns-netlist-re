import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
from subgnn import SubgraphDataset
from model import SubgraphClassifier
from tqdm import tqdm

# === Load saved subgraph dataset ===
dataset = SubgraphDataset(root="subgraph_cache")
print(f"Loaded {len(dataset)} subgraphs.")

# === Data loader ===
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# === Model ===
input_dim = dataset[0].x.size(1)
model = SubgraphClassifier(input_dim=input_dim, hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === Training ===
model.train()
for epoch in range(1, 101):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch:03d}"):
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}")

# === Evaluation ===
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1)
        y_pred.extend(preds.tolist())
        y_true.extend(batch.y.tolist())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["background", "subcircuit"]))