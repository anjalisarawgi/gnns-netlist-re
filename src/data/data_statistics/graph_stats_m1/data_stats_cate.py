# """
# Graph statistics plots for circuit designs, split into two category groups:
#   Group A: Crypto Core, Processor, Arithmetic Core
#   Group B: Communication Controller, ECC Core, Coprocessor, Other

# For each group, four plots are generated:
#   1. Mean nodes per design, all graphs
#   2. Mean nodes per design, complex only (num_modules > 2)
#   3. Mean boundary ratio per design, all graphs
#   4. Mean boundary ratio per design, complex only

# Boundary ratio = mean(boundary_1_ratio) across all graphs of that design
# """

# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# import pandas as pd


# # ---------------------------------------------------------------------------
# # CONFIG
# # ---------------------------------------------------------------------------

# DATA_PATH = Path("data_statistics/graph_stats_m1/final_graph_stats_m1.json")
# OUT_DIR   = Path("data_statistics/graph_stats_m1/by_category")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# CATEGORY_MAP: dict[str, str] = {
#     # ---------------- Crypto Core ----------------
#     "aes_core":                                         "Crypto Core",
#     "aes-encryption_latest":                            "Crypto Core",
#     "aes-master":                                       "Crypto Core",
#     "apbtoaes128_latest":                               "Crypto Core",
#     "csa_latest":                                       "Crypto Core",
#     "des_latest":                                       "Crypto Core",
#     "gcm-aes_latest":                                   "Crypto Core",
#     "gost28147-89_latest":                              "Crypto Core",
#     "hight_latest":                                     "Crypto Core",
#     "md5_core-master":                                  "Crypto Core",
#     "sha3_latest":                                      "Crypto Core",
#     "sha1-master":                                      "Crypto Core",
#     "sha256-master":                                    "Crypto Core",
#     "simon_core_latest":                                "Crypto Core",
#     "tiny_aes_latest":                                  "Crypto Core",

#     # ---------------- Processor ----------------
#     "8051_latest":                                      "Processor",
#     "cpu8080_latest":                                   "Processor",
#     "edge_latest":                                      "Processor",
#     "mips32r1_latest":                                  "Processor",
#     "mips_16_latest":                                   "Processor",
#     "nextz80_latest":                                   "Processor",
#     "openmsp430_latest":                                "Processor",
#     "aemb_latest":                                      "Processor",
#     "Aquarius_latest":                                  "Processor",

#     # ---------------- Arithmetic Core ----------------
#     "8bit_vedic_multiplier_latest":                     "Arithmetic Core",
#     "cavlc_latest":                                     "Arithmetic Core",
#     "double_fpu_latest":                                "Arithmetic Core",
#     "fixed_point_arithmetic_parameterized_latest":      "Arithmetic Core",
#     "ft816float_latest":                                "Arithmetic Core",
#     "mod3_calc_latest":                                 "Arithmetic Core",
#     "signed_integer_divider_latest":                    "Arithmetic Core",
#     "suslik_latest":                                 "Arithmetic Core",
#     "trigonometric_functions_in_double_fpu_latest":     "Arithmetic Core",

#     # ---------------- ECC Core ----------------
#     "rs_5_3_gf256_latest":                              "ECC Core",
#     "rs_decoder_31_19_6_latest":                        "ECC Core",
#     "rs_encoder_decoder_latest":                        "ECC Core",
#     "rsencoder_latest":                                 "ECC Core",

#     # ---------------- Communication Controller ----------------
#     "cxd9731_latest":                                   "Communication Controller",
#     "mac_layer_switch_latest":                          "Communication Controller",
#     "ssp_uart_latest":                                  "Communication Controller",
#     "uart2spi_latest":                                  "Communication Controller",
#     "usb_device_core_latest":                           "Communication Controller",
#     "usbhostslave_latest":                              "Communication Controller",
#     "sdcard_mass_storage_controller_latest":            "Communication Controller",

#     # ---------------- Coprocessor ----------------
#     "orsoc_graphics_accelerator":                       "Coprocessor",

#     # ---------------- Other ----------------
#     "ecg_latest":                                       "Other",
#     "chacha-master":                                    "Other",
#     "cmac-master":                                      "Other",
#     "dpll-isdn_latest":                                 "Other",
#     "blue_latest":                                      "Other",
#     "verilog-axi-librecores":                           "Other",
# }

# CATEGORY_COLORS: dict[str, str] = {
#     "Crypto Core":               "#1f77b4",
#     "Processor":                 "#ff7f0e",
#     "Arithmetic Core":           "#2ca02c",
#     "Communication Controller":  "#d62728",
#     "ECC Core":                  "#17becf",
#     "Coprocessor":               "#8c564b",
#     "Other":                     "#9467bd",
# }

# CATEGORY_ORDER = [
#     "Crypto Core",
#     "Processor",
#     "Arithmetic Core",
#     "Communication Controller",
#     "ECC Core",
#     "Coprocessor",
#     "Other",
# ]

# GROUP_A = ["Crypto Core", "Processor", "Arithmetic Core"]
# GROUP_B = ["Communication Controller", "ECC Core", "Coprocessor", "Other"]
 
# NOISY_DESIGNS = {"tiny_aes_latest"}
 
# GAP = 0  # extra x-space between category groups
 
 
# # ---------------------------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------------------------
 
# def legend_patches(category_subset: list[str] | None = None) -> list[mpatches.Patch]:
#     cats = category_subset if category_subset else CATEGORY_ORDER
#     return [
#         mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat)
#         for cat in cats
#         if cat in CATEGORY_COLORS
#     ]
 
 
# def save(fig: plt.Figure, name: str) -> None:
#     path = OUT_DIR / name
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  saved → {path}")
 
 
# def _grouped_positions(
#     series: pd.Series,
#     category_col: pd.Series,
#     category_subset: list[str] | None = None,
# ) -> tuple[list[int], list[str], list[str]]:
#     """
#     Return (x_positions, ordered_labels, ordered_colors) grouping designs by
#     category with a gap of GAP between groups, sorted by value within each group.
#     """
#     order = category_subset if category_subset else CATEGORY_ORDER
#     tmp = pd.DataFrame({"value": series, "category": category_col})
#     x_positions, labels, colors = [], [], []
#     current_pos = 0
#     for cat in order:
#         subset = tmp[tmp["category"] == cat].sort_values("value")
#         for design in subset.index:
#             x_positions.append(current_pos)
#             labels.append(design)
#             colors.append(CATEGORY_COLORS[cat])
#             current_pos += 1
#         current_pos += GAP
#     return x_positions, labels, colors
 
 
# def max_x_for_group(category_col: pd.Series, group_cats: list[str]) -> int:
#     """Compute the maximum x extent for a given category group."""
#     n_designs = int(category_col.isin(group_cats).sum())
#     n_gaps    = len(group_cats) * GAP
#     return n_designs + n_gaps
 
 
# def grouped_bar(
#     series: pd.Series,
#     category_col: pd.Series,
#     ylabel: str,
#     title: str,
#     filename: str,
#     category_subset: list[str] | None = None,
#     ylim: float | None = None,
#     xlim: float | None = None,
# ) -> None:
#     """Generic grouped bar chart: one bar per design, grouped by category."""
#     # Filter to subset of categories if requested
#     if category_subset is not None:
#         mask = category_col.isin(category_subset)
#         series = series[mask]
#         category_col = category_col[mask]
 
#     x_positions, labels, colors = _grouped_positions(series, category_col, category_subset)
#     values = series[labels].values
 
#     fig, ax = plt.subplots(figsize=(14, 6))
#     ax.bar(x_positions, values, color=colors)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.set_xticks(x_positions)
#     ax.set_xticklabels(labels, rotation=90, fontsize=8)
#     ax.legend(handles=legend_patches(category_subset), title="Category")
 
#     if ylim is not None:
#         ax.set_ylim(0, ylim)
#     if xlim is not None:
#         ax.set_xlim(-0.5, xlim)
 
#     # vertical separators between categories
#     active_order = category_subset if category_subset else CATEGORY_ORDER
#     current_pos = 0
#     for i, cat in enumerate(active_order):
#         n = (category_col == cat).sum()
#         current_pos += n
#         ax.axvline(current_pos - 0.5 + i * GAP, color="black", linewidth=0.5)
#         current_pos += GAP
 
#     fig.tight_layout()
#     save(fig, filename)
 
 
# # ---------------------------------------------------------------------------
# # LOAD DATA
# # ---------------------------------------------------------------------------
 
# print("Loading data …")
# with open(DATA_PATH) as f:
#     data: dict = json.load(f)
 
# rows = []
# for path, stats in data.items():
#     parts   = path.split("/")
#     design  = parts[2]
 
#     nodes   = stats["num_nodes"]
#     b_count = stats["boundary_1_count"]
#     b_ratio = stats["boundary_1_ratio"]
#     n_parts = stats.get("num_unique_partitions", None)
 
#     rows.append({
#         "design":           design,
#         "nodes":            nodes,
#         "boundary_1_count": b_count,
#         "boundary_1_ratio": b_ratio,
#         "num_modules":      n_parts,
#     })
 
# df = pd.DataFrame(rows)
# df = df[~df["design"].isin(NOISY_DESIGNS)]
# df["category"] = df["design"].map(CATEGORY_MAP)
# df = df.dropna(subset=["category"])
 
# print(f"  {len(df):,} graphs across {df['design'].nunique()} designs loaded.")
 
# # Complex-only subset (num_modules > 2)
# df_complex = df[df["num_modules"] > 2]
# print(f"  {len(df_complex):,} complex graphs (modules > 2).")
 
 
# # ---------------------------------------------------------------------------
# # BUILD PER-DESIGN AGGREGATES
# # ---------------------------------------------------------------------------
 
# # -- Nodes: mean per design --------------------------------------------------
# nodes_all     = df.groupby("design")["nodes"].mean()
# nodes_all_cat = nodes_all.index.map(CATEGORY_MAP)
 
# nodes_cpx     = df_complex.groupby("design")["nodes"].mean()
# nodes_cpx_cat = nodes_cpx.index.map(CATEGORY_MAP)
 
# # -- Boundary ratio: mean per design -----------------------------------------
# def mean_boundary(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
#     """Returns (mean boundary_ratio series, category series), indexed by design."""
#     agg = frame.groupby("design").agg(
#         boundary_ratio=("boundary_1_ratio", "mean"),
#         category=("category", "first"),
#     )
#     return agg["boundary_ratio"], agg["category"]
 
# br_all, br_all_cat = mean_boundary(df)
# br_cpx, br_cpx_cat = mean_boundary(df_complex)
 
 
# # ---------------------------------------------------------------------------
# # COMPUTE GLOBAL AXIS LIMITS FOR COMPARABILITY
# # ---------------------------------------------------------------------------
 
# # Shared y limits across all nodes plots and all boundary plots
# nodes_ymax = max(nodes_all.max(), nodes_cpx.max()) * 1.05
# br_ymax    = max(br_all.max(), br_cpx.max()) * 1.05
 
# # Shared x limit: widest group across both GROUP_A and GROUP_B
# nodes_xmax = max(
#     max_x_for_group(nodes_all_cat, GROUP_A),
#     max_x_for_group(nodes_all_cat, GROUP_B),
# )
# br_xmax = max(
#     max_x_for_group(br_all_cat, GROUP_A),
#     max_x_for_group(br_all_cat, GROUP_B),
# )
 
# print(f"\n  nodes_ymax={nodes_ymax:.1f},  br_ymax={br_ymax:.4f}")
# print(f"  nodes_xmax={nodes_xmax},      br_xmax={br_xmax}")
 
 
#  # ---------------------------------------------------------------------------
# # GENERATE PLOTS — all categories in one plot
# # ---------------------------------------------------------------------------
# print("\nGenerating plots …")


# grouped_bar(
#     nodes_all, nodes_all_cat,
#     ylabel   = "Mean number of nodes",
#     title    = "Mean graph size per design — all graphs (grouped by category)",
#     filename = "1_nodes_all.png",
#     # ylim     = nodes_ymax,
# )

# grouped_bar(
#     nodes_cpx, nodes_cpx_cat,
#     ylabel   = "No. of nodes",
#     title    = "Mean nodes per graph design across all circuit designs",
#     filename = "2_nodes_complex.png",
#     # ylim     = nodes_ymax,
# )

# grouped_bar(
#     br_all, br_all_cat,
#     ylabel   = "Mean boundary ratio",
#     title    = "Mean boundary ratio per design — all graphs (grouped by category)",
#     filename = "3_boundary_all.png",
#     # ylim     = br_ymax,
# )

# grouped_bar(
#     br_cpx, br_cpx_cat,
#     ylabel   = "Boundary Ratio",
#     title    = "Mean boundary ratio per graph design across all circuit designs",
#     filename = "4_boundary_complex.png",
#     # ylim     = br_ymax,
# )

# print("\nAll done.")