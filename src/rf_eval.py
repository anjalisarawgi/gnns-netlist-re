import joblib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

rf = joblib.load("models/rf_afterBus/fullgraph_per_design_ce_soft_300ep_for_aes_128_combined_m1+aes_inv_cipher_top_combined_m1+more/rf_model_march3allCrypto.joblib")

# check feature dim from the model itself
print("num features RF expects:", rf.n_features_in_)

feature_names = (
    [f"gate_self_{i}" for i in range(14)] +
    [f"gate_1hop_{i}" for i in range(14)] +
    ["in_deg", "out_deg", "fan_ratio",  # 28, 29, 30
     "in_deg_mean", "in_deg _std",#  31, 32
     "out_neigh_mean", "out_neigh_std", # 33, 34
     "ego_density", # 35
     "kcore", "pagerank", "dist_io", # 36, 37, 38 
     "f_reach", "b_reach", # 39, 40
     "reach_asym", "deg_contrast", # 41, 42
    # "same_deg_count", "avg_neighbor_overlap", "node_fingerprint", "bus_width", "log_bus_width"
     ]
)

print("num names:", len(feature_names))
assert rf.n_features_in_ == len(feature_names), \
    f"Mismatch! RF expects {rf.n_features_in_} features but got {len(feature_names)} names"

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n[GINI] Top 20 features:")
for rank, i in enumerate(indices[:50]):
    print(f"  {rank+1:2d}. {feature_names[i]:30s} {importances[i]:.4f}")

top_n = 50
plt.figure(figsize=(10, 6))
plt.barh(
    [feature_names[i] for i in indices[:top_n]][::-1],
    importances[indices[:top_n]][::-1]
)
plt.xlabel("Gini Importance")
plt.title("RF Feature Importance (Gini)")
plt.tight_layout()
plt.savefig("gini_importance.png", bbox_inches="tight")

