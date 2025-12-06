# import networkx as nx
# import csv

# ###############################################################################
# # 1. Extract boundary nodes from a GML
# ###############################################################################
# def extract_boundary_nodes(gml_path):
#     """Return set of node IDs where boundary = 1."""
#     G = nx.read_gml(gml_path, label="id")
#     boundary_nodes = set()

#     for node, attr in G.nodes(data=True):
#         raw = attr.get("boundary", 0)

#         # Convert to int safely
#         try:
#             raw = int(raw)
#         except:
#             raw = 0

#         if raw == 1:
#             boundary_nodes.add(node)

#     return boundary_nodes


# ###############################################################################
# # 2. Compare method-1 and method-2 boundary sets
# ###############################################################################
# def compare_boundaries(method1_gml, method2_gml, csv_out=None):
#     m1 = extract_boundary_nodes(method1_gml)
#     m2 = extract_boundary_nodes(method2_gml)

#     print("====================================================")
#     print("Boundary Label Comparison")
#     print("====================================================")
#     print(f"Method 1 boundary count: {len(m1)}")
#     print(f"Method 2 boundary count: {len(m2)}")
#     print()

#     is_subset = m1.issubset(m2)
#     print("Is Method-1 a subset of Method-2 ?", is_subset)
#     print()

#     missing = m2 - m1
#     print(f"Nodes in Method-2 missing from Method-1 ({len(missing)}):")
#     print(sorted(list(missing)))
#     print("====================================================")

#     # Optional: Write to CSV
#     if csv_out:
#         with open(csv_out, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["metric", "value"])
#             writer.writerow(["method1_boundary_count", len(m1)])
#             writer.writerow(["method2_boundary_count", len(m2)])
#             writer.writerow(["is_subset", is_subset])
#             writer.writerow(["missing_count", len(missing)])
#             writer.writerow(["missing_nodes", list(missing)])

#         print(f"[INFO] CSV report saved to: {csv_out}")

#     return {
#         "method1": m1,
#         "method2": m2,
#         "is_subset": is_subset,
#         "missing": missing
#     }


# ###############################################################################
# # 3. Example usage (modify paths as needed)
# ###############################################################################
# if __name__ == "__main__":
#     method1_gml = "graphs/processed_v2/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml"   
#     method2_gml = "graphs/processed_v2/aes-encryption_latest/osu035/aes_cipher_top_combined_m2.gml"   

#     # Optional CSV output
#     csv_report = "boundary_comparison_report.csv"

#     result = compare_boundaries(method1_gml, method2_gml, csv_out=csv_report)