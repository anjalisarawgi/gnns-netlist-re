# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

# # --- load data ---
# with open("complex_graph.json") as f:
#     data = json.load(f)

# # --- mapping ---
# folder_to_label = {
#     "aes_core":               "aes_core",
#     "apbtoaes128_latest":     "apbtoes",
#     "aes-encryption_latest":  "aes_enc",
#     "aes-master":             "aes_master",
#     "gcm-aes_latest":         "gcm_aes",
#     "chacha-master":          "chacha",
#     "cmac-master":            "cmac",
#     "csa_latest":             "csa",
#     "des_latest":             "des",
#     "gost28147-89_latest":    "gost",
#     "hight_latest":           "hight",
#     "md5_core-master":        "md5",
#     "sha1-master":            "sha1",
#     "sha3_latest":            "sha3",
#     "sha256-master":          "sha256",
#     "simon_core_latest":      "simon",
# }

# # --- aggregate ---
# aggregated = {
#     label: {"ratio": [], "nodes": [], "partitions": []}
#     for label in folder_to_label.values()
# }

# for filepath, stat in data.items():
#     parts = filepath.split("/")
#     if len(parts) < 3:
#         continue
#     folder = parts[2]
#     if folder not in folder_to_label:
#         continue
#     label = folder_to_label[folder]
#     aggregated[label]["ratio"].append(stat["boundary_1_ratio_labeled"])
#     aggregated[label]["nodes"].append(stat["num_nodes"])
#     aggregated[label]["partitions"].append(stat["num_unique_partitions"])

# # --- compute means ---
# circuits = list(aggregated.keys())
# mean_nodes      = np.array([np.mean(aggregated[c]["nodes"])      for c in circuits])
# mean_boundary   = np.array([np.mean(aggregated[c]["ratio"])      for c in circuits])
# mean_partitions = np.array([np.mean(aggregated[c]["partitions"]) for c in circuits])

# # --- PR-AUC (Bi-Directed GraphSAGE) ---
# prauc_dict = {
#     "aes_core":0.78, "aes_enc":0.77, "aes_master":0.47, "apbtoes":0.24,
#     "chacha":0.23,   "cmac":0.26,    "csa":0.58,         "des":0.50,
#     "gcm_aes":0.76,  "gost":0.65,    "hight":0.49,       "md5":0.15,
#     "sha1":0.27,     "sha3":0.37,    "sha256":0.50,       "simon":0.17,
# }

# # --- boundary-ratio baseline (from your results) ---
# baseline_dict = {
#     "aes_core":0.09, "aes_enc":0.07, "aes_master":0.09, "apbtoes":0.15,
#     "chacha":0.09,   "cmac":0.06,    "csa":0.21,         "des":0.16,
#     "gcm_aes":0.12,  "gost":0.29,    "hight":0.24,       "md5":0.05,
#     "sha1":0.06,     "sha3":0.20,    "sha256":0.07,       "simon":0.02,
# }

# prauc    = np.array([prauc_dict[c]    for c in circuits])
# baseline = np.array([baseline_dict[c] for c in circuits])
# lift     = prauc - baseline

# # --- styling ---
# COLOR     = "#3266ad"
# COLOR_POS = "#3266ad"
# COLOR_NEG = "#dc2626"
# FS_LABEL  = 8
# FS_TITLE  = 11
# FS_AXIS   = 10

# def add_labels(ax, xs, ys, names, offset=(4, -4)):
#     for x, y, name in zip(xs, ys, names):
#         ax.annotate(name, (x, y), xytext=offset,
#                     textcoords="offset points", fontsize=FS_LABEL,
#                     color="#444", clip_on=True)

# def add_corrline(ax, xs, ys):
#     r, p = stats.pearsonr(xs, ys)
#     m, b = np.polyfit(xs, ys, 1)
#     x_line = np.linspace(xs.min(), xs.max(), 100)
#     ax.plot(x_line, m * x_line + b, color="gray",
#             linewidth=0.8, linestyle="--", alpha=0.6)
#     sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
#     ax.set_title(f"r = {r:.2f}{sig}  (p = {p:.3f})", fontsize=FS_TITLE - 1,
#                  color="#555", pad=4)
#     return r

# fig, axes = plt.subplots(2, 2, figsize=(11, 9))
# fig.subplots_adjust(hspace=0.38, wspace=0.32)

# # --- 6.3.1  size vs PR-AUC ---
# ax = axes[0, 0]
# ax.scatter(mean_nodes / 1000, prauc, color=COLOR, s=55, zorder=3, edgecolors="white", linewidths=0.5)
# add_labels(ax, mean_nodes / 1000, prauc, circuits)
# add_corrline(ax, mean_nodes / 1000, prauc)
# ax.set_xlabel("mean num_nodes (thousands)", fontsize=FS_AXIS)
# ax.set_ylabel("PR-AUC", fontsize=FS_AXIS)
# ax.set_ylim(0, 1.05)
# ax.set_title("", fontsize=FS_TITLE)
# ax.text(0.04, 0.96, "6.3.1  graph size vs PR-AUC",
#         transform=ax.transAxes, fontsize=FS_TITLE, fontweight="normal",
#         va="top", color="#333")
# ax.spines[["top", "right"]].set_visible(False)
# ax.grid(axis="both", alpha=0.2, linewidth=0.5)

# # --- 6.3.2  boundary ratio vs PR-AUC ---
# ax = axes[0, 1]
# ax.scatter(mean_boundary, prauc, color=COLOR, s=55, zorder=3, edgecolors="white", linewidths=0.5)
# add_labels(ax, mean_boundary, prauc, circuits)
# add_corrline(ax, mean_boundary, prauc)
# ax.set_xlabel("mean boundary ratio (labeled)", fontsize=FS_AXIS)
# ax.set_ylabel("PR-AUC", fontsize=FS_AXIS)
# ax.set_ylim(0, 1.05)
# ax.text(0.04, 0.96, "6.3.2  boundary ratio vs PR-AUC",
#         transform=ax.transAxes, fontsize=FS_TITLE, fontweight="normal",
#         va="top", color="#333")
# ax.spines[["top", "right"]].set_visible(False)
# ax.grid(axis="both", alpha=0.2, linewidth=0.5)

# # --- 6.3.3  partitions vs PR-AUC ---
# ax = axes[1, 0]
# ax.scatter(mean_partitions, prauc, color=COLOR, s=55, zorder=3, edgecolors="white", linewidths=0.5)
# add_labels(ax, mean_partitions, prauc, circuits)
# add_corrline(ax, mean_partitions, prauc)
# ax.set_xscale("log")
# ax.set_xlabel("mean unique partitions (log scale)", fontsize=FS_AXIS)
# ax.set_ylabel("PR-AUC", fontsize=FS_AXIS)
# ax.set_ylim(0, 1.05)
# ax.text(0.04, 0.96, "6.3.3  subcomponents vs PR-AUC",
#         transform=ax.transAxes, fontsize=FS_TITLE, fontweight="normal",
#         va="top", color="#333")
# ax.spines[["top", "right"]].set_visible(False)
# ax.grid(axis="both", alpha=0.2, linewidth=0.5)

# # --- 6.3.2 lift  boundary ratio vs GNN lift ---
# ax = axes[1, 1]
# point_colors = [COLOR_POS if l >= 0 else COLOR_NEG for l in lift]
# ax.scatter(mean_boundary, lift, color=point_colors, s=55, zorder=3, edgecolors="white", linewidths=0.5)
# add_labels(ax, mean_boundary, lift, circuits)
# add_corrline(ax, mean_boundary, lift)
# ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
# ax.set_xlabel("mean boundary ratio (labeled)", fontsize=FS_AXIS)
# ax.set_ylabel("GNN lift  (GNN − baseline)", fontsize=FS_AXIS)
# ax.text(0.04, 0.96, "6.3.2  GNN lift vs boundary ratio",
#         transform=ax.transAxes, fontsize=FS_TITLE, fontweight="normal",
#         va="top", color="#333")
# ax.spines[["top", "right"]].set_visible(False)
# ax.grid(axis="both", alpha=0.2, linewidth=0.5)

# plt.suptitle("Graph characteristics vs Bi-Directed GraphSAGE PR-AUC",
#              fontsize=13, fontweight="normal", y=1.01, color="#222")

# plt.savefig("results_final/graph_characteristics_vs_prauc.png",
#             dpi=150, bbox_inches="tight")
# plt.show()