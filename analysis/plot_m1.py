import matplotlib.pyplot as plt
import numpy as np

# Walk lengths
walk_lengths = np.array([3, 6, 10, 20, 50, 100])

# Metrics
accuracy = np.array([0.69, 0.82, 0.85, 0.91, 0.89, 0.89])
f1 = np.array([0.83, 0.86, 0.66, 0.70, 0.53, 0.62])

recall = np.array([0.28, 0.42, 0.51, 0.61, 0.64, 0.65])
precision = np.array([0.83, 0.86, 0.66, 0.70, 0.53, 0.62])  # corrected typo from your numbers

# Create 1×3 subplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Accuracy ---
axs[0].plot(walk_lengths, accuracy, marker='o', linewidth=2)
axs[0].set_title("Accuracy vs. Walk Length")
axs[0].set_xlabel("Walk Length")
axs[0].set_ylabel("Accuracy")
axs[0].grid(True)

# --- Plot 2: F1 ---
axs[1].plot(walk_lengths, f1, marker='o', color='purple', linewidth=2)
axs[1].set_title("F1 Score vs. Walk Length")
axs[1].set_xlabel("Walk Length")
axs[1].set_ylabel("F1 Score")
axs[1].grid(True)

# --- Plot 3: Precision + Recall ---
axs[2].plot(walk_lengths, recall, marker='o', linewidth=2, label="Recall")
axs[2].plot(walk_lengths, precision, marker='s', linewidth=2, label="Precision")
axs[2].set_title("Precision & Recall vs. Walk Length")
axs[2].set_xlabel("Walk Length")
axs[2].set_ylabel("Score")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()

# Save the combined figure
plt.savefig("analysis/plots/walk_length_metrics_1x3.png", dpi=300, bbox_inches='tight')
