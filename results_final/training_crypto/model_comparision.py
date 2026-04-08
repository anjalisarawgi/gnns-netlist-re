import matplotlib.pyplot as plt
import numpy as np


circuits = [
    'aes_core', 'apbtoes', 'aes_enc', 'aes_master', 'gcm_aes',
    'chacha', 'cmac', 'csa', 'des', 'gost',
    'hight', 'md5', 'sha1', 'sha3', 'sha256', 'simon'
]

scores = {
    'Boundary-Ratio':      [0.09, 0.14, 0.06 , 0.08, 0.12, 0.08, 0.06, 0.20, 0.15 , 0.24,0.21 ,0.05, 0.05, 0.19, 0.07, 0.02],
    'RandomForest':      [0.75, 0.26, 0.30, 0.44,0.72 ,0.21,    0.23 ,0.47 ,0.55 , 0.67,0.59 ,0.23 ,0.39,     0.40,0.72,0.32],
    'GraphSAGE':      [0.64, 0.22, 0.77, 0.43, 0.71, 0.17, 0.20, 0.54, 0.49, 0.60, 0.48, 0.10, 0.29, 0.45, 0.51, 0.32],
    'Bi-Directed GraphSAGE':         [0.72, 0.24, 0.77, 0.47, 0.76, 0.22, 0.24, 0.59, 0.50, 0.65, 0.49, 0.15, 0.27, 0.37, 0.50, 0.17],
    'JumpingKnowledge GraphSAGE':   [0.76, 0.24, 0.71, 0.45, 0.75, 0.17, 0.26, 0.57, 0.52, 0.59, 0.51, 0.13, 0.34, 0.43, 0.47, 0.32],
}

# colors = {
#     'GraphSAGE':      '#3266ad',
#     'BiSAGE':         '#1D9E75',
#     'JK-GraphSAGE':   '#BA7517',
    # 'BiJK-GraphSAGE': '#993556',
# }

colors = {
    'Boundary-Ratio':             '#4E79A7',  # blue
    'RandomForest':               '#F28E2B',  # orange
    'GraphSAGE':                  '#E15759',  # red
    'Bi-Directed GraphSAGE':      '#76B7B2',  # teal
    'JumpingKnowledge GraphSAGE': '#59A14F',  # green
}

n_circuits = len(circuits)
n_models = len(scores)
x = np.arange(n_circuits)
bar_width = 0.15
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model, vals) in enumerate(scores.items()):
    ax.bar(x + offsets[i], vals, width=bar_width,
           label=model, color=colors[model], edgecolor='white', linewidth=0.5)

# average lines
for model, vals in scores.items():
    avg = np.mean(vals)
    ax.axhline(avg, color=colors[model], linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(circuits, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('PR-AUC score')
ax.set_xlabel('Design families')
ax.set_ylim(0, 1.05)
ax.set_title('Model performance comparition across design families', fontsize=13, fontweight='normal')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('results_final/training_crypto/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()