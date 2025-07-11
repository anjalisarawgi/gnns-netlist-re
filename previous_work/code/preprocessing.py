import itertools
import json
import os
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import sklearn.preprocessing as sp
import torch
import torch_geometric.loader
from sklearn.utils import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from utils.data_utils import store_loaded_graph, load_prepared_graph

labels_set = {}
features_set = {"INPUT": 0, "OUTPUT": 1}


def preprocessGraph(graph, label, feature_depth=2, feature_map=None):
    features = {}
    to_remove = []
    scaler = sp.StandardScaler()
    for node in graph:
        feature_vec = [0 for _ in
                       range(len(features_set) + 2)]  # 2 more for structural info (incoming and outgoing connections)
        node_type = str(node).split("_")[0]
        if node_type == "INPUT" or node_type == "OUTPUT":
            to_remove.append(node)
            continue
        # should the gate type of the current gate itself be included in the features?
        node_type = str(node_type).split("_")[0]
        if feature_map:
            node_type = feature_map[node_type]
        feature_vec[features_set[node_type]] += 1

        predecessors = graph.predecessors(node)
        for _ in range(feature_depth):
            next = []
            for pred in predecessors:
                pred_type = str(pred).split("_")[0]
                if feature_map:
                    pred_type = feature_map[pred_type]
                feature_vec[features_set[pred_type]] += 1
                next += graph.predecessors(pred)
            predecessors = next
        feature_vec[-2] = len(graph.in_edges(node))
        feature_vec[-1] = len(graph.out_edges(node))
        features[node] = feature_vec

    # remove INPUT and OUTPUT (structural info remains in feature vector)
    graph.remove_nodes_from(to_remove)

    features_normalized = scaler.fit_transform(list(features.values()))
    index = 0
    for node in graph:
        features[node] = features_normalized[index]
        index += 1
    nx.set_node_attributes(graph, features, "x")
    nx.set_node_attributes(graph, label, "y")
    return graph


# same function as above but for test set that includes labels
def preprocessGNNRE_Graph(graph, feature_depth=2):
    features = {}
    labels = {}
    to_remove = []
    scaler = sp.StandardScaler()
    gates = nx.get_node_attributes(graph, "nodeType")

    for node in graph:
        feature_vec = [0 for _ in
                       range(len(features_set) + 2)]  # 2 more for structural info (incoming and outgoing connections)
        node_label = str(node).split("_")[0]
        if node_label == "INPUT" or node_label == "OUTPUT":
            to_remove.append(node)
            continue
        if node_label.startswith("U"):
            labels[node] = labels_set["OTHER"]
        else:
            labels[node] = labels_set[node_label]
        # should the gate type of the current gate itself be included in the features?
        predecessors = graph.predecessors(node)
        for _ in range(feature_depth):
            next = []
            for pred in predecessors:
                if "INPUT" in pred:
                    feature_vec[features_set["INPUT"]] += 1
                elif "OUTPUT" in pred:
                    feature_vec[features_set["OUTPUT"]] += 1
                else:
                    pred_type = str(gates[pred]).split("_")[0]
                    pred_type = "".join(itertools.takewhile(str.isalpha, pred_type))
                    feature_vec[features_set[pred_type]] += 1
                next += graph.predecessors(pred)
            predecessors = next
        feature_vec[-2] = len(graph.in_edges(node))
        feature_vec[-1] = len(graph.out_edges(node))
        features[node] = feature_vec

    # remove INPUT and OUTPUT (structural info remains in feature vector)
    graph.remove_nodes_from(to_remove)

    features_normalized = scaler.fit_transform(list(features.values()))
    index = 0
    for node in graph:
        features[node] = features_normalized[index]
        index += 1
    nx.set_node_attributes(graph, features, "x")
    nx.set_node_attributes(graph, labels, "y")
    return graph


def load_and_preprocess_graph(file, label_lookup, feature_map):
    path = construct_file_name(file)
    label = label_lookup[file]
    print(f"loading Graph: {file}")
    graph = nx.read_gml(path)
    graph = preprocessGraph(graph, labels_set[label], feature_map=feature_map)
    return graph, label


def construct_file_name(file):
    # Note assumes that this repository and the re_data repository are in the same directory
    filepath = "../re-graphs/"
    if "osu035" in file:
        filepath += "osu035/"
    elif "osu018" in file:
        filepath += "osu018/"
    elif "nangate" in file:
        filepath += "nangate/"
    elif "gscl45nm" in file:
        filepath += "gscl45nm/"
    filepath = filepath + file + ".gml"
    return filepath


def get_labels_set():
    return labels_set


def get_aisec_graphs(max_number=10, shuffle=False, cut_features=True, preloaded=True, dataset="aisec"):
    feature_map, label_lookup, test_set_files, train_set_files, validation_set_files = choose_graphs(cut_features,
                                                                                                     max_number,
                                                                                                     shuffle, dataset)
    loaded_data_log = {"training": defaultdict(list),
                       "validation": defaultdict(list),
                       "testing": defaultdict(list)}

    train_set = []
    validation_set = []
    test_set = []
    datamap = {}

    for file in train_set_files:
        graph, label = get_prepared_graph(feature_map, file, label_lookup, preloaded)
        train_set.append(graph)
        loaded_data_log["training"][label].append(file)
        datamap[file] = graph

    for file in validation_set_files:
        graph, label = get_prepared_graph(feature_map, file, label_lookup, preloaded)
        validation_set.append(graph)
        loaded_data_log["validation"][label].append(file)

    for file in test_set_files:
        graph, label = get_prepared_graph(feature_map, file, label_lookup, preloaded)
        test_set.append(graph)
        loaded_data_log["testing"][label].append(file)

    with open("data_state/loaded_data_log.json", "w") as f:
        json.dump(loaded_data_log, f, indent=4)

    return train_set, validation_set, test_set, datamap


def get_prepared_graph(feature_map, file, label_lookup, preloaded=True):
    label = label_lookup[file]
    graph = None
    if preloaded:
        graph = load_prepared_graph(file)
    if not graph:
        print(f"Graph {file} not prepared -> loading now")
        graph, label = load_and_preprocess_graph(file, label_lookup, feature_map)
        graph = to_torchdata(graph)
        if preloaded:
            store_loaded_graph(file, graph)
    graph.y.fill_(labels_set[label])
    return graph, label


def get_GNN_RE_graphs(directory="../re-graphs/GNN-RE_test_data/tum_parser", preloaded=True):
    global features_set
    global labels_set

    with open("data_state/GNN-RE_label_set.json", "r") as f:
        labels_set = json.load(f)

    with open("data_state/GNN-RE_feature_set.json", "r") as f:
        features_set = json.load(f)

    graphs = {}
    for filename in os.listdir(directory):
        if filename.endswith(".gml"):
            graph_name = os.path.splitext(filename)[0]
            graph = None
            if preloaded:
                graph = load_prepared_graph(graph_name)
            if not preloaded or not graph:
                print(f"Loading Graph {graph_name}")
                filepath = os.path.join(directory, filename)
                graph = nx.read_gml(filepath)
                graph = preprocessGNNRE_Graph(graph)
                graph = to_torchdata(graph)
                if preloaded:
                    store_loaded_graph(graph_name, graph)
            graphs[graph_name] = graph

    return graphs


def choose_graphs(cut_features, max_number, shuffle, dataset="aisec"):
    global features_set
    global labels_set
    feature_map = None
    if cut_features:
        with open("data_state/cut_features_set.json", "r") as f:
            features_set = json.load(f)
        with open("data_state/feature_map.json", "r") as f:
            feature_map = json.load(f)
    else:
        with open("data_state/features_set.json", "r") as f:
            features_set = json.load(f)
    if dataset == "aisec":
        with open("data_state/labels_lists_final_cut.json", "r") as f:
            labels_lists = json.load(f)
    elif dataset == "aisec-osu":
        with open("data_state/labels_lists_osu035_final_small.json", "r") as f:
            labels_lists = json.load(f)
        with open("data_state/labels_lists_NOT_osu035_final_small.json", "r") as f:
            labels_lists_tests = json.load(f)
    elif dataset == "aisec-small":
        with open("data_state/labels_lists_final_cut_small.json", "r") as f:
            labels_lists = json.load(f)
    elif dataset == "aisec-arithmetic":
        with open("data_state/labels_lists_final_cut_arithmetic.json", "r") as f:
            labels_lists = json.load(f)
    elif dataset == "aisec-components":
        with open("data_state/labels_lists_final_cut_components.json", "r") as f:
            labels_lists = json.load(f)
    possible_labels = list(labels_lists.keys())
    if shuffle:
        random.shuffle(possible_labels)
    labels_to_load = possible_labels[:int(min(len(possible_labels), max_number))]

    train_set_files = []
    validation_set_files = []
    test_set_files = []
    for label in labels_to_load:
        files = labels_lists[label]

        labels_set[label] = len(labels_set)
        random.shuffle(files)

        if dataset == "aisec-osu":
            test_files = labels_lists_tests[label]
            random.shuffle(test_files)

            num_test = int(max(len(test_files) * 0.15, 1))
            num_validation = int(max(len(test_files) * 0.10, 1))
            num_train = len(files)

            test_set_files += test_files[:num_test]
            validation_set_files += test_files[num_test:num_test + num_validation]
            train_set_files += files
        else:
            num_test = int(max(len(files) * 0.15, 1))
            num_validation = int(max(len(files) * 0.10, 1))
            num_train = len(files) - num_test - num_validation

            # Note: the used file was filtered to only include file lists with at least 4 files
            # -> no outOfBoundsError here
            test_set_files += files[:num_test]
            validation_set_files += files[num_test:num_test + num_validation]
            train_set_files += files[num_test + num_validation:]

    with open("data_state/labels_lists_inverted_final.json", "r") as f:
        label_lookup = json.load(f)

    train_set_files.sort()
    validation_set_files.sort()
    test_set_files.sort()
    return feature_map, label_lookup, test_set_files, train_set_files, validation_set_files


def preprocess_and_store():
    train_set, val_set, test_set, _ = get_aisec_graphs(max_number=1000, shuffle=False, cut_features=True,
                                                       preloaded=True)
    dataset = train_set + val_set + test_set
    for graph_name, graph in dataset:
        data = to_torchdata(graph)
        store_loaded_graph(graph_name, data)


def to_torchdata(graph):
    graph = graph.to_undirected()
    pygraph = from_networkx(graph)
    edge_index = pygraph.edge_index
    x = pygraph.x
    y = pygraph.y
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def combine_graphs(dataset):
    combined_x = []
    combined_edge_index = []
    combined_y = []
    current_node_count = 0

    for data in dataset:
        combined_x.append(data.x)
        combined_y.append(data.y)
        # Adjust edge indices
        edge_index_adjusted = data.edge_index + current_node_count
        combined_edge_index.append(edge_index_adjusted)
        # Update the current node count
        current_node_count += data.x.size(0)

    # Concatenate all node features, edge indices, and labels
    combined_x = torch.cat(combined_x, dim=0)
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_y = torch.cat(combined_y, dim=0)

    # Create a new Data object
    combined_data = Data(x=combined_x, edge_index=combined_edge_index, y=combined_y)
    return combined_data


def count_preloaded_and_graphs_per_label():
    global features_set
    global labels_set
    with open("data_state/feature_map.json", "r") as f:
        feature_map = json.load(f)
    with open("data_state/cut_features_set.json", "r") as f:
        features_set = json.load(f)
    with open("data_state/labels_set.json", "r") as f:
        labels_set = json.load(f)

    with open("data_state/labels_lists_inverted_final.json", "r") as f:
        label_lookup = json.load(f)

    # label_lookup2.update(label_lookup)
    unlabeled = []
    for filename in label_lookup.keys():
        graph_name = filename
        if graph_name not in label_lookup:
            unlabeled.append(graph_name)
            print(f"Graph {graph_name} not labeled")
            continue
        # data, _ = get_preloaded_graph(feature_map, graph_name, label_lookup)
        # store_loaded_graph(graph_name, data)

    osu035 = 0
    osu018 = 0
    nangate = 0
    gscl45nm = 0
    for f in os.listdir("data"):
        if "osu035" in f:
            osu035 += 1
        elif "osu018" in f:
            osu018 += 1
        elif "nangate" in f:
            nangate += 1
        elif "gscl45nm" in f:
            gscl45nm += 1

    print(f"osu035: {osu035}")
    print(f"osu018: {osu018}")
    print(f"nangate: {nangate}")
    print(f"gscl45nm: {gscl45nm}")
    print(f"All: {osu018 + gscl45nm + osu035 + nangate}")

    with open("data_state/labels_lists_final.json", "r") as f:
        labels_lists = json.load(f)
    for label, files in labels_lists.items():
        print(f"Label {label}: {len(files)}")


def load_aisec_data(num_labels=10, shuffle=False, cut_features=True, preloaded=True, cuda=False, dataset="aisec", root_nodes=3000, walk_length=2):
    train_set, val_set, test_set, datamap = get_aisec_graphs(num_labels, shuffle, cut_features, preloaded, dataset)

    labels = list(get_labels_set().keys())
    print(f"{16 if cut_features else 56} Features, {len(labels)} Labels: {labels}")

    dataset = train_set + val_set + test_set
    print(f"Number of graphs: {len(dataset)}")

    data = combine_graphs(train_set)
    data_labels = data.y.numpy()
    weights = compute_class_weight('balanced', classes=np.unique(data_labels), y=data_labels)

    train_loader = torch_geometric.loader.GraphSAINTRandomWalkSampler(data, walk_length=walk_length, batch_size=root_nodes,
                                                                      pin_memory=True if cuda else False)
    # DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True if device else False)
    val_loader = DataLoader(val_set, pin_memory=True if cuda else False)
    test_loader = DataLoader(test_set, pin_memory=True if cuda else False)

    return dataset, train_loader, val_loader, test_loader, labels, datamap, weights


def load_GNN_RE_data(cuda=False):
    graphs = get_GNN_RE_graphs(preloaded=True)

    train_set = [graph for g_name, graph in graphs.items() if "Train" in g_name]
    val_set = [graph for g_name, graph in graphs.items() if "Validate" in g_name]
    test_set = [graph for g_name, graph in graphs.items() if "Test" in g_name]

    labels = list(get_labels_set().keys())
    print(f"20 Features, {len(labels)} Labels: {labels}")

    dataset = train_set + val_set + test_set
    print(f"Number of graphs: {len(dataset)}")

    data = combine_graphs(train_set)

    train_loader = torch_geometric.loader.GraphSAINTRandomWalkSampler(data, walk_length=2, batch_size=3000,
                                                                      pin_memory=True if cuda else False,
                                                                      sample_coverage=50)
    # DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True if device else False)
    val_loader = DataLoader(val_set, pin_memory=True if cuda else False)
    test_loader = DataLoader(test_set, pin_memory=True if cuda else False)

    return dataset, train_loader, val_loader, test_loader, labels, graphs


if __name__ == "__main__":
    count_preloaded_and_graphs_per_label()
    # preprocess_and_store()
