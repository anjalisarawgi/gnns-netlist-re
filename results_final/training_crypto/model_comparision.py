import matplotlib.pyplot as plt
import numpy as np


circuits = [
    'aes_core',  'aes_enc', 'aes_master', 'apbtoes',
    
    'chacha', 'cmac', 'csa', 'des', 'gcm_aes', 'gost',
    'hight', 'md5', 'sha1', 'sha3', 'sha256', 'simon'
]
# ### 1
# scores = {
#     'Boundary-Ratio':      [0.09,  0.07 , 0.09,  0.15,       0.09, 0.06, 0.21, 0.16 , 0.12, 0.29,           0.24 ,0.05, 0.06, 0.20, 0.07, 0.02],
#     'GraphSAGE':      [0.64, 0.77, 0.43, 0.22,  0.17, 0.20, 0.54, 0.49,  0.71, 0.60, 0.48, 0.10, 0.29, 0.45, 0.51, 0.32],
#     'Bi-Directed GraphSAGE':         [0.78,  0.77, 0.47, 0.24, 0.23, 0.26, 0.58, 0.50, 0.76, 0.65, 0.49, 0.15, 0.27, 0.37, 0.50, 0.17],
#     'JumpingKnowledge GraphSAGE':   [0.76,  0.71, 0.45, 0.24, 0.17, 0.26, 0.57, 0.52, 0.75, 0.59, 0.51, 0.13, 0.34, 0.43, 0.47, 0.32],
# }

# colors = {
#     'Boundary-Ratio':             '#4E79A7',  # blue
#     'GraphSAGE':               '#F28E2B',  # orange
#     'Bi-Directed GraphSAGE':                  '#E15759',  # red
#     'JumpingKnowledge GraphSAGE':      '#76B7B2',  # teal
# }


# ### 2
scores = {
    'Boundary-Ratio':      [0.09,  0.07 , 0.09,  0.15,       0.09, 0.06, 0.21, 0.16 , 0.12, 0.29,           0.24 ,0.05, 0.06, 0.20, 0.07, 0.02], #0.121
    'RandomForest':      [0.76, 0.31, 0.44,    0.26,      0.22,0.23 ,0.47 ,0.56 ,  0.68 ,  0.73,0.59 ,0.22 ,0.39,     0.40,0.72,0.32], # 0.444
    'Bi-Directed GraphSAGE':         [0.78,  0.77, 0.47, 0.24, 0.23, 0.26, 0.58, 0.50, 0.76, 0.65, 0.49, 0.15, 0.27, 0.37, 0.50, 0.17], # 0.468
}

colors = {
    'Boundary-Ratio':             '#4E79A7',  # blue
    'RandomForest':               '#F28E2B',  # orange
    'Bi-Directed GraphSAGE':                  '#E15759',  # red
}



# ## 3 (gnn feature ablation)
# scores = {
#     'Boundary-Ratio':      [0.09,  0.07 , 0.09,  0.15,       0.09, 0.06, 0.21, 0.16 , 0.12, 0.29,           0.24 ,0.05, 0.06, 0.20, 0.07, 0.02], # 0.121
#     'Features = 14':    [0.75, 0.78, 0.48, 0.23, 0.19, 0.24, 0.58, 0.51, 0.75, 0.65, 0.50, 0.20, 0.27, 0.33, 0.53, 0.15 ], #   0.494
#     'Features = 31':   [0.75, 0.82, 0.50, 0.28,  0.22, 0.25, 0.54, 0.65, 0.78, 0.67, 0.51, 0.19, 0.30, 0.41, 0.57, 0.14], # 0.510
#     'Features = 42':   [0.78,  0.77, 0.47, 0.24, 0.23, 0.26, 0.58, 0.50, 0.76, 0.65, 0.49, 0.15, 0.27, 0.37, 0.50, 0.17], # 0.468

# }

# colors = {
#     'Boundary-Ratio':             '#4E79A7',  # blue
#     'Features = 14':               '#F28E2B',  # orange
#     'Features = 31':                  '#E15759',  # red
#     'Features = 42':      '#76B7B2',  # teal
#     # 'JumpingKnowledge GraphSAGE': '#59A14F',  # green
# }


# ## 3 (rf feature ablation)
# scores = {
#     'Boundary-Ratio':      [0.09,  0.07 , 0.09,  0.15,       0.09, 0.06, 0.21, 0.16 , 0.12, 0.29,           0.24 ,0.05, 0.06, 0.20, 0.07, 0.02], # 0.121
#     'Features = 14':   [0.27, 0.34, 0.15, 0.15, 0.12, 0.10, 0.38, 0.30, 0.31, 0.63, 0.41, 0.08,  0.13, 0.31,  0.15, 0.08], # 0.264
#     'Features = 31':   [0.45, 0.31, 0.26, 0.20, 0.15, 0.13, 0.38, 0.38, 0.45, 0.60, 0.49 , 0.10, 0.29, 0.31, 0.43, 0.12], # 0.319
#     'Features = 42':    [0.76, 0.31, 0.44,    0.26,      0.22,0.23 ,0.47 ,0.56 ,  0.68 ,  0.73,0.59 ,0.22 ,0.39,     0.40,0.72,0.32], # 0.444
# }

# colors = {
#     'Boundary-Ratio':             '#4E79A7',  # blue
#     'Features = 14':               '#F28E2B',  # orange
#     'Features = 31':                  '#E15759',  # red
#     'Features = 42':      '#76B7B2',  # teal
# }



n_circuits = len(circuits)
n_models = len(scores)
x = np.arange(n_circuits)
bar_width = 0.22
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
# ax.set_title('Feature ablations across design families (Model: Bi-Directed GraphSAGE)', fontsize=13, fontweight='normal')
# ax.set_title('Feature ablations across design families (Model: Random Forest', fontsize=13, fontweight='normal')
ax.set_title('GNN architecture vs Random Forest across design families', fontsize=13, fontweight='normal')
# ax.set_title('GNN architecture comparision across design families', fontsize=13, fontweight='normal')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.grid(axis='y', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('results_final/training_crypto/rf_vs_gnns.png', dpi=150, bbox_inches='tight')
plt.show()
