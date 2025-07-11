import networkx as nx
import webcolors
from matplotlib import pyplot as plt


# Function to find the closest color name for an RGB tuple
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


# Function to get color name from RGB
def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        return closest_color(rgb_tuple)


def draw_checkgraph(G, predictions, true_labels, pos, filename=None):
    node_colors = []
    for node in G.nodes:
        if predictions[node] == true_labels[node]:
            node_colors.append('green')  # Correct prediction
        else:
            node_colors.append('red')  # Incorrect prediction

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500)
    plt.title('Graph Node Classification Predictions (Green: Correct, Red: Incorrect)')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def print_closest_colors(colors, labels):
    labellist = list(labels)
    for i in range(len(labellist)):
        color_255 = tuple(int(255 * c) for c in colors[i])
        color_name = get_color_name(color_255)
        print(f"{labellist[i]}: {color_name}")


def draw_classgraph(G, predictions, pos, colors, filename=None):
    node_colors = [colors[pred] for pred in predictions]

    # Draw the graph with node colors
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, cmap=plt.cm.rainbow)
    plt.title('Graph Node Classification Predictions')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_acc_and_val_loss(training_accuracies, validation_losses, training_losses, num_epochs, filename=None):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot training and validation loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, num_epochs + 1), training_losses, label='Training Loss', color='tab:red')
    ax1.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a secondary y-axis for training accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(range(1, num_epochs + 1), training_accuracies, label='Training Accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 1.0)

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Training Loss, Validation Loss, and Training Accuracy Over Epochs')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig("data_state/train_val_loss.png")
    plt.show()


def plot_fi_prec_rec(training_f1_scores, training_precisions, training_recalls, num_epochs, filename=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot precision, recall, and f1 score
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    ax.plot(range(1, num_epochs + 1), training_precisions, label='Precision', color='tab:green')
    ax.plot(range(1, num_epochs + 1), training_recalls, label='Recall', color='tab:purple')
    ax.plot(range(1, num_epochs + 1), training_f1_scores, label='F1 Score', color='tab:brown')
    ax.tick_params(axis='y')
    ax.set_ylim(0, 1.0)  # Set y-axis limits from 0 to 1.0

    # Add legends
    ax.legend(loc='upper left')

    plt.title('Precision, Recall, and F1 Score Over Epochs')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig("data_state/train_f1.png")
    plt.show()
