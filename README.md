# gnns-netlist-re
Master’s thesis project on applying GNNs to netlist reverse engineering at LMU / AISEC

Check weekly progress: [link](https://docs.google.com/document/d/1fxMZ9O8KTFYG73E7Se2II2SxFwwSRWmfGm1Ki7sD8dg/edit?usp=sharing)

Running presentation link (clean): [link](https://docs.google.com/presentation/d/1EAxRFIWQK9jqbIfryDy3k-Qyr9-on3AwTmondeWnO2A/edit?usp=sharing)

Running presentation link (messy): [link](https://docs.google.com/presentation/d/1Vi_SqwBsUBzEUffepb23a5A1HPJ8fzIMnLWZEpuBzUU/edit?usp=sharing)

# Training code and usage

Please run training using a config file:

```bash
python src/main.py --config <path_to_config>
```

## For example:

```bash
python src/main.py --config config/training_aes/lodo_final/aes_core.yml
```
---

## Config File

All parameters can be set in a YAML config file.  An example config (can also be found in config/ folder)


```yaml
train_gml:
  - graphs/processed/aes/osu035/design_A.gml
  - graphs/processed/aes/osu035/design_B.gml
val_gml:
  - graphs/processed/aes/nangate/design_C.gml
test_gml:
  - graphs/processed/aes/nangate/design_D.gml

model: BiDirectedGraphSAGE
training_mode: fullgraph
fullgraph_mode: per_design
loss_type: ce_weighted
epochs: 300
lr: 0.01
reduction_method_cel: mean
set_gradient_clipping: true
use_partition_features: false
```
> Included Model architectures
`gat`, `GAAN`, `gcn`, `gin`, `graphsage`, `BiDirectedGraphSAGE`, `JK_GraphSAGE`, `BiDirectedJK_GraphSAGE`

> **Please note that:** graph paths should follow the directory structure `<design_family>/<library>/<graph>.gml`

---

## Outputs

All results are saved to (after training):
- `models/gnns/<run_name>/model.pt`: saves the best model weights (from early stopping)
- `models/gnns/<run_name>/scaler.pkl`: the fitted feature scaler (on training graphs)
- `models/gnns/<run_name>/metadata.json`: run config and metadata 
- `results/gnns/<run_name>/<design>__<lib>__<graph>_predictions.gml`: predictions on each test graph as included in the config

Each prediction GML includes per-node fields: `prob`, `pred_default (at argmax 0.5) `, `pred_best (best threshold)`, `pred_ratio (top-k)`, and `pred_class_*`  for each method (TP/FP/FN/TN).
