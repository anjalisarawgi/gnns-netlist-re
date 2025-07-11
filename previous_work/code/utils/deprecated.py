import itertools
import json
import os
from collections import defaultdict

import networkx as nx
from sklearn import preprocessing as sp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from preprocessing import get_labels_set, features_set, preprocessGraph, preprocessGNNRE_Graph, \
    choose_graphs, load_and_preprocess_graph
from utils.convert import convert_GNN_RE_graphs


def graphToData(graph):
    graph = graph.to_undirected()
    pygraph = from_networkx(graph)
    edge_index = pygraph.edge_index
    x = pygraph.x  # torch.tensor(features)
    y = pygraph.y  # torch.tensor([label for _ in range(len(features))], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def do_everything_with_re_data(num_graphs=5):
    graphs, labels, features = get_graphs(num_graphs)
    print(labels.values())
    labels_set = get_labels_set()
    labels = [labels_set[label] for label in labels.values()]

    dataset = [graphToData(graph) for graph in list(graphs.values())]
    print(dataset[0].x.shape[1], len(labels))

    datamap = {}
    for g_name, graph in graphs.items():
        datamap[g_name] = graphToData(graph)

    train_loader = DataLoader(dataset[:num_graphs], batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset[:num_graphs], batch_size=1)
    test_loader = DataLoader(dataset[:num_graphs], batch_size=1)

    return dataset, train_loader, val_loader, test_loader, labels, datamap


def do_everything_with_converted_GNN_RE_data():
    graphs, labels, features = get_graphs3()

    train_set = [graphToData(graph) for g_name, graph in graphs.items() if "Train" in g_name]
    val_set = [graphToData(graph) for g_name, graph in graphs.items() if "Validate" in g_name]
    test_set = [graphToData(graph) for g_name, graph in graphs.items() if "Test" in g_name]

    datamap = {}
    for g_name, graph in graphs.items():
        datamap[g_name] = graphToData(graph)

    dataset = train_set + val_set + test_set
    print(len(features), len(labels))

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set)
    test_loader = DataLoader(test_set)

    return dataset, train_loader, val_loader, test_loader, labels, datamap


directory = "C:/Users/lassm/Documents/TUM/TUM Master/IDP/gnntests/re-graphs/osu035"
directory2 = "C:/Users/lassm/Documents/TUM/TUM Master/IDP/gnntests/re-graphs/GNN-RE_test_data/tum_parser"
directory3 = "A:/IDP/2024-idp-lassmann-braun-gnn-re/data"
split = "_final_"


def load_graphs_from_directory(directory, max=5):
    graphs = {}
    labels = {}
    limit = 0
    for filename in os.listdir(directory):
        if filename.endswith(".gml") and split in filename and limit < max:
            print(filename)
            filepath = os.path.join(directory, filename)
            graph_name = os.path.splitext(filename)[0]
            label = graph_name[graph_name.index(split) + len(split):]
            if label not in labels_set:
                labels_set[label] = len(labels_set)
            graph = nx.read_gml(filepath)
            for node in graph:
                name = str(node)
                name = name.split("_")[0]
                if name not in features_set:
                    features_set[name] = len(features_set)
            graphs[graph_name] = graph
            labels[graph_name] = label
            limit += 1
    return graphs, labels


def load_graphs_from_directory2(directory):
    graphs = {}
    global labels_set
    labels_set = {"OTHER": 0}
    for filename in os.listdir(directory):
        if filename.endswith(".gml"):
            filepath = os.path.join(directory, filename)
            graph_name = os.path.splitext(filename)[0]
            graph = nx.read_gml(filepath)
            gates = nx.get_node_attributes(graph, "nodeType")
            print(graph_name)
            print(graph.nodes(data=True))
            for node in graph:
                name = str(node)
                name = name.split("_")[0]
                if "INPUT" in node or "OUTPUT" in node or node.startswith("U"):
                    continue
                if name not in labels_set:
                    labels_set[name] = len(labels_set)
                gate_type = str(gates[node]).split("_")[0]
                gate_type = "".join(itertools.takewhile(str.isalpha, gate_type))
                if gate_type not in features_set:
                    features_set[gate_type] = len(features_set)
            graphs[graph_name] = graph

    return graphs


def get_graphs(limit=10):  # osu018
    scaler = sp.StandardScaler()
    graphs, labels = load_graphs_from_directory(directory, limit)
    all_features = {}
    for g_name, graph in graphs.items():
        features, graph = preprocessGraph(graph, labels_set[labels[g_name]])
        features = scaler.fit_transform(list(features.values()))
        all_features[g_name] = features
    return graphs, labels, all_features


def get_graphs2(dir):  # tum_parser
    graphs = load_graphs_from_directory2(dir)
    for g_name, graph in graphs.items():
        preprocessGNNRE_Graph(graph)
    return graphs, labels_set.keys(), features_set.keys()


def get_graphs3():
    graphs = convert_GNN_RE_graphs()
    scaler = sp.StandardScaler()
    for name, graph in graphs.items():
        features = nx.get_node_attributes(graph, "x")
        features_normalized = scaler.fit_transform(list(features.values()))
        index = 0
        for node in graph:
            features[node] = features_normalized[index]
            index += 1
        nx.set_node_attributes(graph, features, "x")
    return graphs, [0, 1, 2, 3, 4], [i for i in range(34)]


def load_re_graphs(max_number=10, shuffle=True, cut_features=False):
    feature_map, label_lookup, test_set_files, train_set_files, validation_set_files = choose_graphs(cut_features,
                                                                                                     max_number,
                                                                                                     shuffle)
    loaded_data_log = {"training": defaultdict(list),
                       "validation": defaultdict(list),
                       "testing": defaultdict(list)}

    train_set = []
    validation_set = []
    test_set = []

    for file in train_set_files:
        graph, label = load_and_preprocess_graph(file, label_lookup, feature_map)
        train_set.append((file, graph))
        loaded_data_log["training"][label].append(file)

    for file in validation_set_files:
        graph, label = load_and_preprocess_graph(file, label_lookup, feature_map)
        validation_set.append((file, graph))
        loaded_data_log["validation"][label].append(file)

    for file in test_set_files:
        graph, label = load_and_preprocess_graph(file, label_lookup, feature_map)
        test_set.append((file, graph))
        loaded_data_log["testing"][label].append(file)

    with open("data_state/loaded_data_log.json", "w") as f:
        json.dump(loaded_data_log, f, indent=4)

    return train_set, validation_set, test_set
