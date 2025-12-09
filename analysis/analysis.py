import pandas as pd

df = pd.read_csv("results/models/0.01perc_train_326_m1_gat_graphsaint_walk20_3000ep_for_aes_128_combined_m1/model.pt/eval_results.csv")
# df = pd.read_csv("results/models/1perc_batch_size/train_full_m1_gat_graphsaint_rw_1000ep_aes_cipher_top_combined_m1+aes_key_expand_128_combined_m1+test_for_aes_128_combined_m1/model.pt/eval_results.csv")

averages = df.mean(numeric_only=True)

print("Averages:")
print(averages)