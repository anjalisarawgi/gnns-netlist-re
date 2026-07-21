import pandas as pd
import networkx as nx
from sklearn.metrics import average_precision_score


### gives pr-auc for each paritition within a gml file
GML_PATH = "processed/annotated/aes_master/trainingAEs_graphsage_ce_weighted_fullgraph_per_design_aes_master/aes-master__osu035__aes_core_combined_m1_predictions.gml"
G = nx.read_gml(GML_PATH, label="id")

rows = []
for nid, data in G.nodes(data=True):
    rows.append({
        "gt"       : int(data.get("boundary", -1)),
        "partition": str(data.get("partition_label", "unknown")),
        "prob"     : float(data.get("prob", 0.5)),
    })
df = pd.DataFrame(rows)
df_valid = df[df["gt"].isin([0, 1])].copy()

# calculate prauc for parition based on the probability scores
def partition_pr_auc(sub):
    if sub["gt"].sum() == 0 or sub["gt"].sum() == len(sub):
        return float("nan")
    return average_precision_score(sub["gt"], sub["prob"])

part_prauc = (
    df_valid.groupby("partition")
    .apply(lambda s: pd.Series({
        "n_nodes"        : len(s),
        "n_boundary"     : s["gt"].sum(),
        "boundary_ratio" : s["gt"].mean(),
        "pr_auc"         : partition_pr_auc(s),
    }))
    .reset_index()
    .sort_values("pr_auc", ascending=True)
)
print(part_prauc.to_string(index=False, float_format="{:.3f}".format))
