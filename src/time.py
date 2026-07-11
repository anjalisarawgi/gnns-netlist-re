import wandb

api = wandb.Api()
run = api.run("anjalisarawgi/gnn-parition-detection/897l91rk")

history = run.history(keys=["time/train_sec"])
total = history["time/train_sec"].sum()
print(f"Total train time: {total:.2f}s ({total/60:.2f} min)")