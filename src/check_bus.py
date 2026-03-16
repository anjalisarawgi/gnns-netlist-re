"""
Bus Hypothesis Test for WL Pattern Analysis
============================================
Tests whether nodes with frequency=8 WL patterns represent bus structures
by comparing clustering, gate composition, and boundary proximity
against all other frequency groups.
"""

import networkx as nx
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter, defaultdict
import csv
import os

# ─── Config ───────────────────────────────────────────────────────────────────
WL_ITERS      = 3
TARGET_FREQ   = 8          # hypothesis: these are bus nodes
COMPARE_FREQS = [64, 32, 24, 12, 4]
OUTPUT_DIR    = "bus_hypothesis_outputs"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_gate(label_copy):
    if label_copy is None:
        return "UNKNOWN"
    name = str(label_copy).strip("'").upper()
    for key in ["INPUT", "OUTPUT", "DFF", "INV", "AND", "OR",
                "NAND", "NOR", "XOR", "XNOR", "MUX"]:
        if key in name:
            return key
    return "UNKNOWN"


def wl_refinement(G, labels, iters):
    for _ in range(iters):
        new_labels = {}
        for node in G.nodes():
            neigh = sorted(labels[n] for n in G.neighbors(node))
            sig   = labels[node] + "|" + "|".join(neigh)
            new_labels[node] = hashlib.md5(sig.encode()).hexdigest()
        labels = new_labels
    return labels


def boundary_ratio(nodes, G):
    """Returns (boundary_count, total, ratio) for a set of nodes."""
    b = sum(1 for n in nodes if int(G.nodes[n].get("boundary", 0)) == 1)
    t = len(nodes)
    return b, t, (b / t if t else 0.0)


def nodes_for_freq(freq, wl_labels, counts, valid_nodes):
    patterns = {p for p, c in counts.items() if c == freq}
    return [n for n in valid_nodes if wl_labels[n] in patterns]


def one_hop_neighbors(node_set, G):
    out = set()
    for n in node_set:
        for nb in G.neighbors(n):
            if nb not in node_set:
                out.add(nb)
    return out


def two_hop_neighbors(node_set, G):
    out = set()
    for n in node_set:
        for nb in G.neighbors(n):
            for nb2 in G.neighbors(nb):
                if nb2 not in node_set:
                    out.add(nb2)
    return out


# ── clustering metrics ────────────────────────────────────────────────────────

def cluster_stats(node_set, G):
    """
    For a set of nodes, compute:
      - number of connected components when we look at the induced subgraph
      - average component size
      - fraction of inter-component edges (edges leaving the subgraph)
    """
    if not node_set:
        return {}

    sub = G.subgraph(node_set)
    comps = list(nx.connected_components(sub))
    sizes = [len(c) for c in comps]

    # edges leaving the subgraph  (bus wires connect to outside logic)
    internal_edges = sub.number_of_edges()
    total_edges    = sum(G.degree(n) for n in node_set)
    external_edges = total_edges - 2 * internal_edges   # each internal edge counted twice

    return {
        "num_components"      : len(comps),
        "avg_component_size"  : np.mean(sizes),
        "max_component_size"  : max(sizes),
        "node_count"          : len(node_set),
        "internal_edges"      : internal_edges,
        "external_edges"      : external_edges,
        "external_edge_ratio" : external_edges / (total_edges or 1),
    }


# ── gate composition ──────────────────────────────────────────────────────────

def gate_composition(node_set, G):
    c = Counter(G.nodes[n].get("gate_type", "UNKNOWN") for n in node_set)
    total = len(node_set)
    return {g: (cnt, cnt / total) for g, cnt in c.most_common()}


# ── main ──────────────────────────────────────────────────────────────────────

def run(input_gml):
    print("=" * 60)
    print("BUS HYPOTHESIS TEST")
    print("=" * 60)

    G = nx.read_gml(input_gml)
    if G.is_directed():
        G = G.to_undirected()

    # label nodes with gate type
    labels      = {}
    valid_nodes = set()
    for n, data in G.nodes(data=True):
        gate = parse_gate(data.get("label_copy", ""))
        G.nodes[n]["gate_type"] = gate
        labels[n] = gate
        if gate not in ["INPUT", "OUTPUT"]:
            valid_nodes.add(n)

    wl_labels = wl_refinement(G, labels, WL_ITERS)
    counts    = Counter(wl_labels[n] for n in valid_nodes)

    all_freqs = sorted({TARGET_FREQ} | set(COMPARE_FREQS))

    # ── 1. Summary stats table ────────────────────────────────────────────────
    print(f"\n{'Freq':>6}  {'Nodes':>6}  {'BndryRatio':>10}  "
          f"{'1hopBndry':>10}  {'2hopBndry':>10}  "
          f"{'#Comps':>7}  {'AvgCompSz':>10}  {'ExtEdgeRatio':>13}")
    print("-" * 80)

    summary_rows = []

    for freq in all_freqs:
        ns = nodes_for_freq(freq, wl_labels, counts, valid_nodes)
        if not ns:
            continue

        b, t, br     = boundary_ratio(ns, G)
        hop1         = one_hop_neighbors(set(ns), G)
        hop2         = two_hop_neighbors(set(ns), G)
        _, _, h1r    = boundary_ratio(hop1, G)
        _, _, h2r    = boundary_ratio(hop2, G)
        cs           = cluster_stats(set(ns), G)

        tag = " ← TARGET" if freq == TARGET_FREQ else ""
        print(f"{freq:>6}  {t:>6}  {br:>10.3f}  {h1r:>10.3f}  {h2r:>10.3f}  "
              f"{cs['num_components']:>7}  {cs['avg_component_size']:>10.2f}  "
              f"{cs['external_edge_ratio']:>13.3f}{tag}")

        summary_rows.append({
            "freq": freq,
            "nodes": t,
            "boundary_ratio": round(br, 4),
            "1hop_boundary_ratio": round(h1r, 4),
            "2hop_boundary_ratio": round(h2r, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in cs.items()},
        })

    # ── 2. Gate composition for target vs rest ────────────────────────────────
    print(f"\n── Gate composition: freq={TARGET_FREQ} (bus candidates) ──")
    target_nodes = nodes_for_freq(TARGET_FREQ, wl_labels, counts, valid_nodes)
    gc_target    = gate_composition(target_nodes, G)
    for g, (cnt, pct) in gc_target.items():
        print(f"  {g:<10}  {cnt:>5}  ({pct:.3f})")

    all_non_target = [n for n in valid_nodes
                      if counts[wl_labels[n]] != TARGET_FREQ]
    print(f"\n── Gate composition: all other nodes ──")
    gc_other = gate_composition(all_non_target, G)
    for g, (cnt, pct) in gc_other.items():
        print(f"  {g:<10}  {cnt:>5}  ({pct:.3f})")

    # ── 3. Clustering: are freq=8 nodes physically close to each other? ───────
    print(f"\n── Cluster stats for freq={TARGET_FREQ} ──")
    cs = cluster_stats(set(target_nodes), G)
    for k, v in cs.items():
        print(f"  {k:<25}  {v}")

    # ── 4. Export CSV for manual inspection ──────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f"bus_candidates_freq{TARGET_FREQ}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "gate_type", "boundary",
                         "wl_label", "degree", "wl_pattern_freq"])
        for n in target_nodes:
            d    = G.nodes[n]
            freq = counts[wl_labels[n]]
            writer.writerow([
                n,
                d.get("gate_type", ""),
                d.get("boundary", 0),
                wl_labels[n][:12] + "...",
                G.degree(n),
                freq,
            ])
    print(f"\nExported {len(target_nodes)} bus candidates → {csv_path}")

    # ── 5. Save annotated GML ─────────────────────────────────────────────────
    for n in G.nodes():
        freq = counts.get(wl_labels[n], 0)
        G.nodes[n]["wl_freq"]         = freq
        G.nodes[n]["is_bus_candidate"] = 1 if freq == TARGET_FREQ else 0

    gml_path = os.path.join(OUTPUT_DIR, "graph_bus_annotated.gml")
    nx.write_gml(G, gml_path)
    print(f"Annotated GML → {gml_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    _plot_boundary_bars(summary_rows)
    _plot_subgraph(G, target_nodes, wl_labels, counts)
    _plot_gate_pie(gc_target, gc_other)
    _plot_component_size_hist(G, target_nodes)

    print(f"\nAll plots saved to ./{OUTPUT_DIR}/")
    return G, wl_labels, counts, summary_rows


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_boundary_bars(rows):
    freqs  = [r["freq"] for r in rows]
    br     = [r["boundary_ratio"] for r in rows]
    h1     = [r["1hop_boundary_ratio"] for r in rows]
    h2     = [r["2hop_boundary_ratio"] for r in rows]
    colors = ["#e63946" if r["freq"] == TARGET_FREQ else "#457b9d" for r in rows]

    x  = np.arange(len(freqs))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, br, w, label="Direct",  color=colors, alpha=0.9)
    ax.bar(x,     h1, w, label="1-hop",   color=colors, alpha=0.6)
    ax.bar(x + w, h2, w, label="2-hop",   color=colors, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([f"freq={f}" for f in freqs], rotation=30)
    ax.set_ylabel("Boundary node ratio")
    ax.set_title("Boundary Proximity by WL Pattern Frequency\n(red = bus hypothesis target)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    red_patch  = mpatches.Patch(color="#e63946", label=f"freq={TARGET_FREQ} (bus candidate)")
    blue_patch = mpatches.Patch(color="#457b9d", label="other frequencies")
    ax.legend(handles=[red_patch, blue_patch,
                        mpatches.Patch(color="gray", alpha=0.9, label="Direct"),
                        mpatches.Patch(color="gray", alpha=0.6, label="1-hop"),
                        mpatches.Patch(color="gray", alpha=0.35, label="2-hop")])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boundary_proximity_bars.png"), dpi=150)
    plt.close()


def _plot_subgraph(G, target_nodes, wl_labels, counts):
    """
    Visualise the induced subgraph of bus candidates.
    Node colour = boundary status. Size = degree.
    """
    if len(target_nodes) > 300:
        # sample for readability
        rng   = np.random.default_rng(42)
        sample = list(rng.choice(target_nodes, 300, replace=False))
    else:
        sample = target_nodes

    sub    = G.subgraph(sample)
    pos    = nx.spring_layout(sub, seed=42, k=0.4)
    colors = ["#e63946" if int(G.nodes[n].get("boundary", 0)) == 1
              else "#a8dadc" for n in sub.nodes()]
    sizes  = [20 + 8 * sub.degree(n) for n in sub.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx(sub, pos, ax=ax,
                     node_color=colors, node_size=sizes,
                     with_labels=False, edge_color="#cccccc",
                     width=0.5, alpha=0.85)

    red_p  = mpatches.Patch(color="#e63946", label="Boundary node")
    blue_p = mpatches.Patch(color="#a8dadc", label="Non-boundary node")
    ax.legend(handles=[red_p, blue_p], loc="upper left")
    ax.set_title(f"Induced subgraph of freq={TARGET_FREQ} bus candidates\n"
                 f"({len(sample)} nodes shown, size ∝ degree)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bus_subgraph.png"), dpi=150)
    plt.close()


def _plot_gate_pie(gc_target, gc_other):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, gc, title in zip(
        axes,
        [gc_target, gc_other],
        [f"freq={TARGET_FREQ} (bus candidates)", "All other nodes"]
    ):
        labels = list(gc.keys())
        sizes  = [v[0] for v in gc.values()]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.set_title(title)

    plt.suptitle("Gate Type Composition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gate_composition_pie.png"), dpi=150)
    plt.close()


def _plot_component_size_hist(G, target_nodes):
    """Distribution of connected-component sizes within bus candidates."""
    sub   = G.subgraph(target_nodes)
    sizes = sorted([len(c) for c in nx.connected_components(sub)], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=range(1, max(sizes) + 2), color="#e63946",
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Component size (# nodes)")
    ax.set_ylabel("Count")
    ax.set_title(f"Connected component size distribution\nwithin freq={TARGET_FREQ} subgraph")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # annotate bus-width multiples
    for bw in [8, 16, 32]:
        if bw <= max(sizes):
            ax.axvline(bw, color="#457b9d", linestyle="--", linewidth=1.2,
                       label=f"{bw}-bit bus width")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "component_size_hist.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_GML = (
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml"
    )
    run(INPUT_GML)

    import networkx as nx

    G = nx.read_gml("new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml")
    total = G.number_of_nodes()
    boundary = sum(1 for n, d in G.nodes(data=True) if int(d.get("boundary", 0)) == 1)
    print(f"Total nodes:    {total}")
    print(f"Boundary nodes: {boundary}")
    print(f"Baseline rate:  {boundary/total:.4f}  ({boundary/total*100:.1f}%)")