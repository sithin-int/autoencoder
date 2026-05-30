from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from functools import wraps
import numpy as np
import pandas as pd
import os

import gc
import time
import tracemalloc

ROOT_FOLDER_NAME = "autoencoder"
dirs = os.getcwd().split("/")
index = dirs.index(ROOT_FOLDER_NAME)
ROOT_DIR = "/".join(dirs[:index + 1])

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
def lasso_fs(X_df, y_df, cv=5):

    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, l1_ratio=1, solver="liblinear", max_iter=10_000, random_state=42))

    features = X_df.columns.to_numpy()

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()

    estimator.fit(X, y)

    rank_dict = {"feature":features, "absolute_coef":np.abs(estimator[-1].coef_[0])}
    rank_df = pd.DataFrame(rank_dict)
    
    rank_df["rank"] = rank_df["absolute_coef"].rank(method="min", ascending=False).astype(int)
    rank_df.sort_values("rank", inplace=True)
    rank_df.reset_index(drop=True, inplace=True)

    rank_dict = rank_df.to_dict(orient="list")

    return rank_dict


def run_lasso(train_df):

    features = [feat for feat in train_df.columns if feat not in NON_FEATURE_COLS]

    rank_dict, elapsed_time, peak_mem = lasso_fs(train_df[features], train_df[LABEL])

    return rank_dict, elapsed_time, peak_mem

def main():

    radiomics_df = pd.read_csv(XL_PATH)

    perturbations = np.load(PERTURBATIONS_FILE, allow_pickle=True).item()


    for perturb_id in tqdm(perturbations, desc="Running soft data-perturbation on LASSO", position=0):
    
        train_pids, train_labels = perturbations[perturb_id]["train"]
        test_pids, test_labels = perturbations[perturb_id]["val"]

        train_df = radiomics_df[radiomics_df.id.isin(train_pids)]
        val_df = radiomics_df[radiomics_df.id.isin(test_pids)]

        rank_dict, elapsed_time, peak_mem = run_lasso(train_df)

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

        outdir = os.path.join(OUT_DIR, "embedded", "LASSO")
        os.makedirs(outdir, exist_ok=True)
    
        out_path = os.path.join(outdir, f"{perturb_id}.npz")
        np.savez_compressed(out_path, rank_dict=np.array(rank_dict, dtype=object), predictions=predictions, targets=targets, elapsed_time=elapsed_time, peak_memory=peak_mem)

if __name__ == "__main__":
    main()
