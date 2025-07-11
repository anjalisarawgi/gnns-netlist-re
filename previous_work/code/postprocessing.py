import torch


def majority_postprocessing(pred, probabilities, data, labels):
    all_label_counts = torch.bincount(pred, minlength=len(labels))
    all_majority_label = all_label_counts.argmax()

    for i, prob in enumerate(probabilities):
        # get neighbors and compare labels
        node_label = pred[i]
        edge_index = data.edge_index
        neighbors = edge_index[1, edge_index[0] == i]
        neighbor_labels = pred[neighbors]

        neighbor_label_counts = torch.bincount(neighbor_labels, minlength=len(labels))
        neighbor_majority_label = neighbor_label_counts.argmax()
        # print(neighbor_majority_label)
        neighbor_majority_count = neighbor_label_counts[neighbor_majority_label]
        if ((neighbor_majority_label != node_label)
                and (neighbor_majority_count > (len(neighbors) // 2))
                and (all_majority_label == neighbor_majority_label)
        ):
            # extra check if most of graph has the same label as node before changing as well as
            # checking weather the probs of all possible labels of the node are roughly equal
            pred[i] = neighbor_majority_label
        elif ((neighbor_majority_label == node_label)
              and (all_majority_label != neighbor_majority_label)
              and (neighbor_majority_count <= (len(neighbors) // 2))
        ):
            pred[i] = all_majority_label

    #   print('post processing complete')
    return pred


def probability_postprocessing(pred, probabilities):
    overall_mean_probabilities = probabilities.mean(dim=0)
    overall_std = probabilities.std(dim=0)

    for i, prob in enumerate(probabilities):
        node_mean_prob = prob.mean().item()
        node_max_prob = prob.max().item()
        node_std_prob = prob.std().item()
        # Check node standard deviation -> when low -> change label when overall standard deviation is high to overall label
        if (node_std_prob < 0.1 * node_mean_prob) and (overall_std.max() > 0.1):
            overall_label = overall_mean_probabilities.argmax().item()
            if node_max_prob < overall_mean_probabilities[overall_label].item():
                pred[i] = overall_label

    return pred


def use_majority_label(pred):
    majority_label = torch.max(pred).item()
    pred.fill_(majority_label)
    return pred
