
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

## the lodo plot 
labels = [
    "gost28147-89_latest\nall designs of gscl45nm",
    "hight_latest\nall designs of osu035",
    "md5_core\nall designs of osu035",
    "sha1-master\nall designs of nangate",
]

lodo   = [0.45, 0.54, 0.14, 0.23]
added  = [0.65, 0.83, 0.58, 0.19]



x = np.arange(len(labels))
width = 0.32
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - width/2, lodo,  width, color="#E15759", label="Before adding designs from other libraries")
ax.bar(x + width/2, added, width, color="#76B7B2", label="After adding designs from other libraries")

for xi, (l, a) in enumerate(zip(lodo, added)):
    delta = a - l
    ax.annotate(f"{'+' if delta > 0 else ''}{delta:.2f}", xy=(xi + width/2, a),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, fontweight="bold",
                color="#1a6648" if delta > 0 else "#8B0000")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("PR-AUC", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("Effect of designs from two libraries and testing it on the third library", fontsize=12, pad=12)
ax.legend(fontsize=9, framealpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("analysis/ablation_lodo_crypto.png", dpi=150, bbox_inches="tight")
