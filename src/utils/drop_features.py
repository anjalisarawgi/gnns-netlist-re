import numpy as np

# load features.txt

def load_features(file_path):
    features = np.loadtxt(file_path)
    return features 

def drop_feature_columns(features):
    # keep_cols = list(range(32))# + [20, 21] # keep only first 11, 21, 22
    # features = features[:, keep_cols]
    return features

if __name__ =="__main__":
    save_path = "feat_dropped.txt"
    file_path = "feat.txt"
    features = load_features(file_path)
    features_dropped = drop_feature_columns(features)
    features = features.astype(np.int32) 
    np.savetxt(save_path, features_dropped,  fmt='%d')
    np.save("feat_dropped.npy", features_dropped)
    print("done")
