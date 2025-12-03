import matplotlib.pyplot as plt
import numpy as np

# Walk lengths
walk_lengths = np.array([3, 6, 10, 20, 50, 100])

# Method 2 Metrics
accuracy = np.array([0.75, 0.84, 0.91, 0.87, 0.82, 0.83])
recall =   np.array([0.61, 0.73, 0.85, 0.76, 0.63, 0.65])
precision = np.array([0.53, 0.67, 0.83, 0.73, 0.68, 0.64])

# Create 1x3 subplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Accuracy ---
axs[0].plot(walk_lengths, accuracy, marker='o', linewidth=2)
axs[0].set_title("Accuracy vs. Walk Length (Method 2)")
axs[0].set_xlabel("Walk Length")
axs[0].set_ylabel("Accuracy")
axs[0].grid(True)

# --- Plot 2: Recall ---
axs[1].plot(walk_lengths, recall, marker='o', color='green', linewidth=2)
axs[1].set_title("Recall vs. Walk Length (Method 2)")
axs[1].set_xlabel("Walk Length")
axs[1].set_ylabel("Recall")
axs[1].grid(True)

# --- Plot 3: Precision + Recall ---
axs[2].plot(walk_lengths, recall, marker='o', linewidth=2, label="Recall")
axs[2].plot(walk_lengths, precision, marker='s', linewidth=2, label="Precision")
axs[2].set_title("Precision & Recall vs. Walk Length (Method 2)")
axs[2].set_xlabel("Walk Length")
axs[2].set_ylabel("Score")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()

# Save figure
plt.savefig("analysis/plots/method2_walk_length_metrics_1x3.png", dpi=300, bbox_inches='tight')
