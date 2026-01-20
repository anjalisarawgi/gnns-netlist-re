# import yaml

# INPUT_YAML  = "config/train_new_graphs_crypto_v4.yml"          # your original YAML
# OUTPUT_YAML = "config/train_new_graphs_crypto_v4.yml"  # filtered YAML

# with open(INPUT_YAML, "r") as f:
#     cfg = yaml.safe_load(f)

# def keep_only(paths, tag):
#     if paths is None:
#         return paths
#     return [p for p in paths if tag in p]

# # filter rules
# cfg["train_gml"] = keep_only(cfg.get("train_gml"), "_m2.gml")
# cfg["val_gml"]   = keep_only(cfg.get("val_gml"), "_m1.gml")
# cfg["test_gml"]  = keep_only(cfg.get("test_gml"), "_m1.gml")

# with open(OUTPUT_YAML, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)

# print("Saved filtered config to:", OUTPUT_YAML)
# print("Train graphs:", len(cfg["train_gml"]))
# print("Val graphs:", len(cfg.get("val_gml", [])))
# print("Test graphs:", len(cfg.get("test_gml", [])))
import cv2
import numpy as np

img = cv2.imread("K_0013_0163.png", cv2.IMREAD_GRAYSCALE)
score = cv2.Laplacian(img, cv2.CV_64F).var()
normalized_sharpness = laplacian_var / (img.shape[0] * img.shape[1])
print(normalized_sharpness)  # higher = sharper