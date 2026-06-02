from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_classif
from mrmr import mrmr_classif
from scipy.stats import mannwhitneyu

from tqdm import tqdm
from functools import wraps
import numpy as np
import pandas as pd
import os

import gc
import time
import tracemalloc

ROOT_FOLDER_NAME = "autoencoder"
dirs = os.getcwd().split(os.path.sep)
index = dirs.index(ROOT_FOLDER_NAME)
ROOT_DIR = os.path.sep.join(dirs[:index + 1])

XL_PATH = os.path.join(ROOT_DIR, "inputs", "radiomicsFeaturesWithLabels.csv")
PERTURBATIONS_FILE = os.path.join(ROOT_DIR, "outputs", "data_perturbations.npy")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")

NON_FEATURE_COLS = ["id", "label"]
LABEL = "label"

NUM_FEATS_TO_SELECT = 5

def profile_time_and_memory(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        
        start_time = time.perf_counter()
        
        gc.collect()
        tracemalloc.start()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = end_time - start_time

        return result, elapsed_time, peak_mem
    
    return wrapper

@profile_time_and_memory
def filter_fs(df, fs_method):

    features = [feat for feat in df.columns if feat not in NON_FEATURE_COLS]

    X_df = df[features]
    y_df = df[LABEL]

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()

    is_ascending = True
    scores = []
    if fs_method.__name__=="mannwhitneyu":
        for i in range(len(features)):
            stat, p = fs_method(X[y==0,i], X[y==1,i], alternative="two-sided")
            scores.append(p)
    elif fs_method.__name__=="mrmr_classif":
        ordered_features = fs_method(X_df, y_df, K=len(features), n_jobs=1, show_progress=False)
        feature_to_score = {feat:i+1 for i,feat in enumerate(ordered_features)}
        scores = [feature_to_score[feat] for feat in features]
    elif fs_method.__name__=="mutual_info_classif":
        scores = fs_method(X, y, random_state=42, n_jobs=1)
        is_ascending = False
    
    rank_dict = {"feature":features, "score":scores}
    rank_df = pd.DataFrame(rank_dict)
    rank_df["rank"] = rank_df["score"].rank(method="min", ascending=is_ascending).astype(int)
    rank_df.sort_values("rank", inplace=True)
    rank_df.reset_index(drop=True, inplace=True)

    rank_dict = rank_df.to_dict(orient="list")

    return rank_dict


def main():

    radiomics_df = pd.read_csv(XL_PATH)

    perturbations = np.load(PERTURBATIONS_FILE, allow_pickle=True).item()
    
    fs_methods = [mannwhitneyu, mutual_info_classif, mrmr_classif]

    for fs_method in fs_methods:
        for perturb_id in tqdm(perturbations, desc=f"Running soft data-perturbation on {fs_method.__name__}", position=0):
            
            train_pids = perturbations[perturb_id]["train"]
            test_pids = perturbations[perturb_id]["val"]

            train_df = radiomics_df[radiomics_df.id.isin(train_pids)]
            val_df = radiomics_df[radiomics_df.id.isin(test_pids)]

            rank_dict, elapsed_time, peak_mem = filter_fs(train_df, fs_method)

            rank_df = pd.DataFrame(rank_dict)
            rank_df.sort_values("rank", inplace=True)
            rank_df.reset_index(drop=True, inplace=True)

            selected_feats = rank_df.head(NUM_FEATS_TO_SELECT).feature.to_list()

            X_train = train_df[selected_feats]
            y_train = train_df[LABEL]

            X_val = val_df[selected_feats]
            y_val = val_df[LABEL]

            pred_model = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10_000, random_state = 42)) #no penalty
            pred_model.fit(X_train, y_train)
        
            predictions = pred_model.predict_proba(X_val)[:,1]
            targets = y_val.to_numpy().ravel()

            outdir = os.path.join(OUT_DIR, "filter", fs_method.__name__)
            os.makedirs(outdir, exist_ok=True)
        
            out_path = os.path.join(outdir, f"{perturb_id}.npz")
            np.savez_compressed(out_path, rank_dict=np.array(rank_dict, dtype=object), predictions=predictions, targets=targets, elapsed_time=elapsed_time, peak_memory=peak_mem)

if __name__ == "__main__":
    main()


# # sanity check
# import numpy as np
# import os
# from sklearn.metrics import roc_auc_score

# root_dir = "/Users/sithin/research/phd/autoencoder"

# fs_method = "filter/mutual_info_classif"
# _ = np.load(os.path.join(root_dir, "outputs", fs_method, "0.npz"), allow_pickle=True)



# rank_df = pd.DataFrame(_["rank_dict"].item())
# predictions = _["predictions"]
# targets = _["targets"]

# print(roc_auc_score(targets, predictions))
# display(rank_df.head())
# display(rank_df.tail())
