import os
import json
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from analyse_thresholds import load_single_gml, load_model


@torch.no_grad()
def get_embeddings(model, data):
    logits, embeddings = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        return_embeddings=True
    )

    probs = torch.softmax(logits, dim=1)[:, 1]

    return (
        embeddings.cpu().numpy(),
        data.y.cpu().numpy(),
        probs.cpu().numpy()
    )


def process_graph(gml_path, model, scaler, output_dir, max_per_class=5000):
    print(f"\nProcessing: {gml_path}")

    data, _ = load_single_gml(
        gml_path,
        use_partition_features=False,
        use_graph_features=False
    )

    data.x = torch.tensor(
        scaler.transform(data.x.cpu().numpy()),
        dtype=torch.float32
    )

    embeddings, labels, probs = get_embeddings(model, data)

    # Balanced sampling
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    if len(idx0) > max_per_class:
        idx0 = np.random.choice(idx0, max_per_class, replace=False)

    if len(idx1) > max_per_class:
        idx1 = np.random.choice(idx1, max_per_class, replace=False)

    idx = np.concatenate([idx0, idx1])

    embeddings = embeddings[idx]
    labels = labels[idx]
    probs = probs[idx]

    print("After sampling:", embeddings.shape)

    # Silhouette raw
    if len(np.unique(labels)) > 1:
        sil_raw = silhouette_score(embeddings, labels)
        print("Silhouette (raw):", sil_raw)
    else:
        print("Only one class present.")

    # PCA
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    emb_pca = pca.fit_transform(embeddings)

    if len(np.unique(labels)) > 1:
        sil_pca = silhouette_score(emb_pca, labels)
        print("Silhouette (PCA):", sil_pca)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )

    emb_2d = tsne.fit_transform(emb_pca)


    parts = gml_path.split(os.sep)
    module = os.path.splitext(parts[-1])[0]
    library = parts[-2]
    design = parts[-3]
    base_name = f"{design}_{library}_{module}"

    # Plot 1: label-colored
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=6
    )

    plt.title(f"{base_name}")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()

    label_path = os.path.join(output_dir, f"{base_name}_tsne_labels.png")
    plt.savefig(label_path)
    plt.close()

    # Plot 2: probability-colored
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=probs,
        cmap="viridis",
        alpha=0.6,
        s=6
    )

    plt.title(f"{base_name)")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()

    prob_path = os.path.join(output_dir, f"{base_name}_tsne_probs.png")
    plt.savefig(prob_path)
    plt.close()

    print("Saved:", label_path)
    print("Saved:", prob_path)


if __name__ == "__main__":

    test_graphs = [
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/tiny_aes_latest/osu035/aes_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/gscl45nm/aes_inv_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes_core/nangate/aes_key_expand_128_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/sha1-master/osu035/sha1_core_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/des_latest/gscl45nm/des_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_key_expand_128_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/aes-encryption_latest/osu035/aes_cipher_top_combined_m1.gml",
        "new_graphs_crypto/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/trigonometric_functions_in_double_fpu_latest/gscl45nm/top_combined_m1.gml",
        "graphs/processed_partitions_boundaryM1_oneHotAdd_graphF_partitionF_feb10/mips32r1_latest/gscl45nm/ALU_combined_m1.gml",
    ]

    model_dir = "models/fullgraph_per_design_ce_soft_200ep_for_aes_128_combined_m1+aes_inv_cipher_top_combined_m1+more"
    model_path = os.path.join(model_dir, "model.pt")
    metadata_path = os.path.join(model_dir, "metadata.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    output_dir = "plots/woPartitions"
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, metadata_path)
    scaler = joblib.load(scaler_path)

    for gml_path in test_graphs:
        process_graph(gml_path, model, scaler, output_dir)