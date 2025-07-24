import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("csv/osu035.csv")  

# for nodes and edges, we create bins to categorize  counts
bins = [0, 1000, 5000, 100000, float('inf')]
labels = ['1–1000', '1001–5000', '5001–10000', '10000+']
df['node_bin'] = pd.cut(df['#nodes'], bins=bins, labels=labels, right=False)
df['edge_bin'] = pd.cut(df['#edges'], bins=bins, labels=labels, right=False)
node_bin_counts = df['node_bin'].value_counts().sort_index()
edge_bin_counts = df['edge_bin'].value_counts().sort_index()

print("Node Count Bins:", node_bin_counts)
print("Edge Count Bins:", edge_bin_counts)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# node histogram
node_bin_counts.plot(kind='bar', ax=axs[0], color='skyblue', edgecolor='black')
axs[0].set_title('Node Count Distribution')
axs[0].set_xlabel('Node Count Bin')
axs[0].set_ylabel('Number of Files')

# edge histogram
edge_bin_counts.plot(kind='bar', ax=axs[1], color='lightgreen', edgecolor='black')
axs[1].set_title('Edge Count Distribution')
axs[1].set_xlabel('Edge Count Bin')
axs[1].set_ylabel('Number of Files')

plt.tight_layout()
plt.savefig('visualizations/osu035/node_edge_distribution.png')





###################################
## correlation matrix

numeric_cols = df[[
    '#nodes', '#edges', '#inputs', '#outputs', '#XOR', '#AND',
    '#OR', '#BUF', 'avg_degree'
]]

corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Circuit Features")
plt.tight_layout()
plt.savefig("visualizations/osu035/correlation_heatmap.png")
plt.close()
