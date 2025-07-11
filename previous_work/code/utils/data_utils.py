import csv
import json
import os
from itertools import combinations
from pathlib import Path

import editdistance
import torch


def sort_features():
    with open("../data_state/features_set.json", "r") as f:
        features_set = json.load(f)
        names = list(features_set.keys())
        names = names[2:]
        names.sort()
        print(names)

        features_set_sorted = {"INPUT": 0, "OUTPUT": 1}
        for name in names:
            features_set_sorted[name] = len(features_set_sorted)

    with open("../data_state/features_set.json", "w") as f:
        json.dump(features_set_sorted, f, indent=4)


def compare_labels():
    with open("../data_state/original_labels_set.json", "r") as f:
        labels_set = json.load(f)
        labels = list(labels_set.keys())
        pairs = list(combinations(labels, 2))
        distances = [(p[0], p[1], editdistance.eval(p[0], p[1])) for p in pairs]
        filtered_distances = list(filter(lambda x: x[2] < 4, distances))
        edit_distances = {}
        checked_labels = []
        for p1, p2, distance in filtered_distances:
            if p1 not in edit_distances and p1 not in checked_labels:
                edit_distances[p1] = {}
            elif p1 not in edit_distances and p1 in checked_labels:
                continue
            if distance not in edit_distances[p1]:
                edit_distances[p1][distance] = []
            edit_distances[p1][distance].append(p2)
            if p1 not in checked_labels:
                checked_labels.append(p1)
            if p2 not in checked_labels:
                checked_labels.append(p2)

        print(edit_distances)

    with open("../data_state/labels_distance.json", "w") as f:
        json.dump(edit_distances, f, indent=4)


# invert {label -> [file]} to {file -> label}
def invert_dict(d):
    inverted = {}
    for k, v in d.items():
        for x in v:
            inverted[x] = k
    return inverted


# invert {file -> label} to {label -> [file]}
def reinvert_dict(d):
    inverted = {}
    for k, v in d.items():
        if v not in inverted:
            inverted[v] = []
        inverted[v].append(k)
    return inverted


def filter_graphs():
    with open("../data_state/labels_lists_modified.json", "r") as f:
        labels_lists_modified = json.load(f)
        del_list = []
        for label, files in labels_lists_modified.items():
            #    if label.startswith("gf_mult_") or label.startswith("GF8Mult"):
            #         labels_lists_modified["gf_mult"] += files
            if len(files) < 4:
                del_list.append(label)
        for l in del_list:
            del labels_lists_modified[l]
        print(len(labels_lists_modified))
    with open("../data_state/labels_lists_modified_filtered.json", "w") as f:
        json.dump(labels_lists_modified, f, indent=4)
    with open("../data_state/labels_lists_modified_filtered_inverted.json", "w") as f:
        json.dump(invert_dict(labels_lists_modified), f, indent=4)


def store_loaded_graph(name, data):
    torch.save(data, "data/" + name + ".pt")
    print(f"Graph {name} saved to disk")


def load_prepared_graph(name):
    path = Path("data/" + name + ".pt")
    if path.exists():
        return torch.load(path)
    else:
        return None


def catalogue_re_data():
    directory = "A:/IDP/re-graphs/"
    synth_libs = ["osu035", "osu018", "nangate", "gscl45nm"]
    graphs = []
    labels_set = {}
    label_lookup = {}
    labels_lists = {}

    with open("../data_state/label_map.json") as f:
        label_map = json.load(f)

    with open(directory + "catalogue_new__1_.csv", "r") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            shortened_label = row[6]  # .split(",")[0].strip().lower()
            label_lookup[row[0]] = shortened_label

    unlabeled_graphs = []

    for synth_lib in synth_libs:
        for filename in os.listdir(directory + synth_lib):
            if filename.endswith(".gml"):
                print(filename)
                graph_name = filename.replace(".gml", "")
                if graph_name in label_lookup:
                    label = label_lookup[graph_name]
                    if len(label) == 0:
                        unlabeled_graphs.append(graph_name)
                        continue
                    label = label_map[label]
                    if len(label) == 0:
                        unlabeled_graphs.append(graph_name)
                        continue
                    if label not in labels_set:
                        labels_set[label] = len(labels_set)
                    if label not in labels_lists:
                        labels_lists[label] = []
                    labels_lists[label] += [graph_name]
                    graphs.append(graph_name)
                else:
                    unlabeled_graphs.append(graph_name)

    labels = list(labels_set.keys())
    labels.sort(key=len, reverse=True)

    autolabels = {}
    for unlabeled_graph in unlabeled_graphs:
        labeled = False
        for label in labels:
            if label.casefold() in unlabeled_graph.split("_final_")[-1].casefold():
                labels_lists[label].append(unlabeled_graph)
                autolabels[unlabeled_graph] = label
                labeled = True
                break
        if labeled:
            continue
        # for label in labels_set.keys():
        #     if label.casefold() in unlabeled_graph.casefold():
        #         labels_lists[label].append(unlabeled_graph)
        #         autolabels[unlabeled_graph] = label
        #         break
        print(unlabeled_graph)

    print(f"found {len(graphs)} graphs and {len(labels_set)} labels")
    # print(labels_set)
    # print(labels_lists)

    # with open("../data_state/autolabeled_graphs.json", "w") as f:
    #     json.dump(autolabels, f, indent=4)
    #
    # with open("../data_state/checked_autolabeled_graphs.json", "w") as f:
    #     json.dump(reinvert_dict(autolabels), f, indent=4)
    #
    with open("../data_state/labels_set.json", "w") as f:
        json.dump(labels_set, f, indent=4)

    # with open("data_state/features_set.json", "w") as f:
    #     json.dump(features_set, f, indent=4)

    with open("../data_state/labels_lists_final.json", "w") as f:
        json.dump(labels_lists, f, indent=4)


def filter_osu035_graphs():
    with open("../data_state/labels_lists_final_cut.json", "r") as f:
        labels_lists_final = json.load(f)

    label_list_filtered = {}
    label_list_other_filtered = {}
    num_osu035 = 0

    for label, files in labels_lists_final.items():
        label_list_filtered[label] = [file for file in files if "_osu035_" in file]
        label_list_other_filtered[label] = [file for file in files if "_osu035_" not in file]
        num_osu035 += len(label_list_filtered[label])
        print(f"Label: {label}, has {len(label_list_filtered[label])} osu035 graphs")
        print(f"Label: {label}, has {len(label_list_other_filtered[label])} non-osu035 graphs")

    print(f"Osu035 graphs found: {num_osu035}")

    with open("../data_state/labels_lists_osu035_final.json", "w") as f:
        json.dump(label_list_filtered, f, indent=4)

    with open("../data_state/labels_lists_osu035_inverted_final.json", "w") as f:
        json.dump(invert_dict(label_list_filtered), f, indent=4)

    with open("../data_state/labels_lists_NOT_osu035_final.json", "w") as f:
        json.dump(label_list_other_filtered, f, indent=4)

    with open("../data_state/labels_lists_NOT_osu035_inverted_final.json", "w") as f:
        json.dump(invert_dict(label_list_other_filtered), f, indent=4)


def dataset_stats():
    with open("../data_state/labels_lists_inverted_final.json", "r") as f:
        labels_lists_final = json.load(f)

    num_nodes = 0
    for label in labels_lists_final.keys():
        path = Path("../data/" + label + ".pt")
        graph = None
        if path.exists():
            graph = torch.load(path)
        if graph is None:
            print(f"{label} not found")
            continue
        num_nodes += len(graph.x)

    print(f"{num_nodes} nodes in {len(labels_lists_final)} aisec graphs")

    with open("../data_state/labels_lists_osu035_inverted_final.json", "r") as f:
        labels_lists_final = json.load(f)

    num_nodes = 0
    for label in labels_lists_final.keys():
        path = Path("../data/" + label + ".pt")
        graph = None
        if path.exists():
            graph = torch.load(path)
        if graph is None:
            print(f"{label} not found")
            continue
        num_nodes += len(graph.x)

    print(f"{num_nodes} nodes in {len(labels_lists_final)} aisec osu035 graphs")


    num_nodes = 0
    train = 0
    val = 0
    test = 0
    for filename in os.listdir("../data/"):
        if filename.startswith("Train") or filename.startswith("Test") or filename.startswith("Validate"):
            path = Path("../data/" + filename)
            graph = None
            if path.exists():
                graph = torch.load(path)
            if graph is None:
                print(f"{label} not found")
                continue
            if filename.startswith("Train"):
                train += len(graph.x)
            elif filename.startswith("Test"):
                test += len(graph.x)
            elif filename.startswith("Validate"):
                val += len(graph.x)
            else:
                print(f"{filename} is weird")
            num_nodes += len(graph.x)

    print(f"{num_nodes} nodes in 37 GNN-RE graphs")
    print(f"train: {train}, val: {val}, test: {test}")
    print(f"{len(graph.x[0])} features in GNN-RE graphs")


if __name__ == "__main__":
    # filter_graphs()
    # catalogue_re_data()
    # filter_osu035_graphs()
    # dataset_stats()
    with open("../data_state/labels_lists_final_cut_small.json", "r") as f:
        labels_lists_final = json.load(f)

    with open("../data_state/labels_lists_osu035_final_small.json", "r") as f:
        labels_lists_osu035 = json.load(f)

    not_osu = {}
    for label, files in labels_lists_final.items():
        print(f"Label: {label}, has {len(files)} graphs")
        not_osu[label] = []
        for file in files:
            if file not in labels_lists_osu035[label]:
                not_osu[label].append(file)

    with open("../data_state/labels_lists_NOT_osu035_final_small.json", "w") as f:
        json.dump(not_osu, f, indent=4)
    pass
