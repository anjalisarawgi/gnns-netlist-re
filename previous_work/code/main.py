import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.utils import to_networkx

from gat import GAT
from graphSAGE import GraphSAGE
from postprocessing import majority_postprocessing
from preprocessing import load_aisec_data, \
    load_GNN_RE_data
from training import train
from utils.evaluation import draw_checkgraph, print_closest_colors, plot_acc_and_val_loss, \
    plot_fi_prec_rec, draw_classgraph

if __name__ == '__main__':
    # NOTE: preloaded graphs currently are assumed to use cut feature vector, and not differentiated when loaded
    # -> possible undefined behavior or errors when using preloading with uncut features

    # default config
    config = {
        "model": "graphSAGE",  # possible values: graphSAGE, GAT
        "layers": 3,  # number of layers including fully connected
        "hidden": 256,  # size of hidden dimension
        "learning rate": 0.01,
        "dropout": 0.1,
        "weight decay": 1e-5,
        "loss": "CrossEntropy",  # possible values: CrossEntropy, consistency
        "epochs": 500,
        "data": "aisec",  # possible values: aisec, GNN-RE, aisec-osu, aisec-small, aisec-arithmetic, aisec-components
        "labels": 27,  # only used for aisec data, (final condensed label set has 27 labels)
        "features": "cut",  # possible values: cut, full (only used for aisec data)
        "shuffle": False,  # should labels be shuffled before choosing (only does something if labels < max_labels
        "preload": True,  # should graphs be loaded from disk (if available)/ stored to disk (if not already)
        "root-nodes": 3000,     # random walk sampler root nodes (only used for aisec data)
        "walk-length": 2    # random walk sampler walk length (only used for aisec data)
    }

    parser = argparse.ArgumentParser(description='start training and evaluation of configured model')
    parser.add_argument("-c", '--config', type=Path, help='config file')
    args = parser.parse_args()

    if args.config:
        with open("config.json", "r") as f:
            loaded_config = json.load(f)

    if loaded_config:
        print(f"Loaded config: {loaded_config}")
        for key, value in loaded_config.items():
            if key == "shuffle" or key == "preload":
                config[key] = value.casefold() == "True".casefold()
            else:
                config[key] = value
    else:
        print("No config file -> using default config")

    num_epochs = config["epochs"]
    re_data = config["data"] == "GNN-RE"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    timestamp_format = '%Y-%m-%d_%H-%M'

    timestamp_start = datetime.now()

    weights = None
    if re_data:
        dataset, train_loader, val_loader, test_loader, labels, datamap = load_GNN_RE_data(torch.cuda.is_available())
    else:
        dataset, train_loader, val_loader, test_loader, labels, datamap, weights = load_aisec_data(config["labels"],
                                                                                                   config["shuffle"],
                                                                                                   config[
                                                                                                       "features"] == "cut",
                                                                                                   config["preload"],
                                                                                                   torch.cuda.is_available(),
                                                                                                   config["data"],
                                                                                                   config["root-nodes"],
                                                                                                   config["walk-length"])

    timestamp_train = datetime.now()

    if config["model"] == "graphSAGE":
        model = GraphSAGE(in_channels=dataset[0].x.shape[1], hidden_channels=config["hidden"], out_channels=len(labels),
                          num_layers=config["layers"], dropout=config["dropout"])
    else:
        model = GAT(in_channels=dataset[0].x.shape[1], hidden_channels=config["hidden"], out_channels=len(labels),
                    heads=8, num_layers=config["layers"], dropout=config["dropout"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning rate"], weight_decay=config["weight decay"])

    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    model.to(device)

    train_acc, train_val_loss, train_loss, train_f1, train_prec, train_recall = train(model, optimizer, criterion,
                                                                                      train_loader, val_loader,
                                                                                      num_epochs,
                                                                                      cons_loss=config[
                                                                                                    "loss"] == "consistency",
                                                                                      device=device)

    timestamp_test = datetime.now()

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            data.x = data.x.float()

            out = model(data)

            # Get the index of the max log-probability
            pred = out.argmax(dim=1)
            if not re_data:
                pred = majority_postprocessing(pred, torch.nn.functional.softmax(out, dim=1), data, labels)
                # pred = use_majority_label(pred)

            # Collect predictions and true labels
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

    # Convert lists to tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    timestamp_finish = datetime.now()

    model.eval()
    for i, _ in enumerate(labels):
        with torch.no_grad():
            if re_data:
                testgraphs = [g for g in list(datamap.keys()) if "Test" in g]
                testgraph = random.randint(0, len(testgraphs) - 1)
                data = datamap[testgraphs[testgraph]]
            else:
                testgraph = -1
                data = datamap[list(datamap.keys())[testgraph]]
                while (len(data.x) > 2500 or data.y[0] != i) and testgraph + len(datamap) > 0:
                    testgraph -= 1
                    data = datamap[list(datamap.keys())[testgraph]]
            data = data.to(device)
            data.x = data.x.float()
            out = model(data)
            predictions = out.argmax(dim=1)
            probabilities = torch.nn.functional.softmax(out, dim=1)
            print(labels)
            for j, prob in enumerate(probabilities):
                formatted_probs = [f"{p:.4f}" for p in prob]
                print(f"Node {j}: {formatted_probs}")
            G = to_networkx(data, to_undirected=True)

            print(f"showing Graph {list(datamap.keys())[testgraph]} with label {data.y[0]}")
            colormap = plt.cm.get_cmap('tab20', len(labels))
            colors = [colormap(i)[:3] for i in range(len(labels))]
            pos = nx.kamada_kawai_layout(G)
            # draw_classgraph(G, predictions, pos, colors)

            true_labels = data.y

            outfile_name = config["data"] + "_" + config["model"] + "_" + list(datamap.keys())[
                testgraph] + "_" + timestamp_start.strftime(timestamp_format)

            if i == 9:  # should be "sub" for small dataset
                draw_checkgraph(G, true_labels, predictions, pos,
                                filename="results/" + outfile_name + "check_before.png")
            if not re_data:
                predictions = majority_postprocessing(predictions, probabilities, data, labels)
                # predictions = use_majority_label(predictions)
                if i == 9:  # should be "sub" for small dataset
                    draw_classgraph(G, predictions, pos, colors, filename="results/" + outfile_name + "class_after.png")
                    draw_checkgraph(G, true_labels, predictions, pos,
                                    filename="results/" + outfile_name + "check_after.png")
        if re_data:
            break

    print_closest_colors(colors, labels)

    outfile_name = config["data"] + "_" + config["model"] + "_" + timestamp_start.strftime(timestamp_format)
    dataloading_time = (timestamp_train - timestamp_start).total_seconds()
    training_time = (timestamp_test - timestamp_train).total_seconds()
    total_time = (timestamp_finish - timestamp_start).total_seconds()

    plot_acc_and_val_loss(train_acc, train_val_loss, train_loss, num_epochs,
                          filename="results/" + outfile_name + "_acc_val_loss.png")
    plot_fi_prec_rec(train_f1, train_prec, train_recall, num_epochs,
                     filename="results/" + outfile_name + "_fi_prec_rec.png")

    print(model)

    number_graphs = len(dataset)
    number_nodes = sum(len(g.x) for g in dataset)

    with open("results/" + outfile_name + ".log", "w") as f:
        f.write(f"Model: {config['model']}\n")
        f.write(str(model) + "\n")
        f.write("-" * 50 + "\n")
        f.write(f"Data loading time: {dataloading_time:.0f}s\n")
        f.write(f"Training time: {training_time:.0f}s\n")
        f.write(f"Total time: {total_time:.0f}s\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of graphs: {number_graphs}\n")
        f.write(f"Number of nodes: {number_nodes}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write("Config:\n")
        pretty_config = json.dumps(config, indent=4)
        f.write(pretty_config)
