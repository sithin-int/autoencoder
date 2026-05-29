from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm
from functools import wraps
import numpy as np
import pandas as pd
import os

import gc
import time
import tracemalloc

XL_PATH = "inputs/radiomicsFeaturesWithLabels.csv"
PERTURBATIONS_FILE = "outputs/data_perturbations.npy"
OUT_DIR = "outputs"

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
def bsfs(estimator, X_df, y_df, cv=5):

    features = X_df.columns.to_numpy()

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()

    sfs = SFS(estimator, n_features_to_select=len(features)-1, direction='backward', scoring="roc_auc", cv=cv, n_jobs=1)
    sfs.fit(X,y)
   
    kept = features[sfs.support_]
    eliminated = features[~sfs.support_][0]
    
    return {"kept": kept, "eliminated": eliminated}


def iterative_bsfs(df, estimator):

    remaining = [feat for feat in df.columns if feat not in NON_FEATURE_COLS]
    
    rank_dict = {"feature":[], "rank":[]}

    elapsed_timex, peak_memx = [], []
    for k in tqdm(range(len(remaining), 1, -1), desc="running iterative bsfs", position=0):
        
        X_df = df[remaining]
        y_df = df[LABEL]
        
        result, elapsed_time, peak_mem = bsfs(clone(estimator), X_df, y_df)
        remaining = result["kept"]
        eliminated = result["eliminated"]

        rank_dict["feature"].append(eliminated)
        rank_dict["rank"].append(k)

        elapsed_timex.append(elapsed_time)
        peak_memx.append(peak_mem)

    rank_dict["feature"].append(remaining[0])
    rank_dict["rank"].append(1)
    
    tot_elapsed_time = np.sum(elapsed_timex)
    tot_peak_mem = np.sum(peak_memx)

    return rank_dict, tot_elapsed_time, tot_peak_mem


def main():

    estimators = [
        make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10_000, random_state=42)), #no penalty
        make_pipeline(StandardScaler(), SVC(kernel="linear", max_iter=10_000, random_state=42, probability=True)),
        make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs=1, random_state=42)),
        make_pipeline(StandardScaler(), MLPClassifier(max_iter=10_000, random_state=42))
    ]

    radiomics_df = pd.read_csv(XL_PATH)

    perturbations = np.load(PERTURBATIONS_FILE, allow_pickle=True).item()

    for estimator in estimators:

        for perturb_id in tqdm(perturbations, position=0, desc=f"Running soft data-perturbations on {estimator[-1].__class__.__name__}"):

            train_pids, train_labels = perturbations[perturb_id]["train"]
            test_pids, test_labels = perturbations[perturb_id]["val"]

            train_df = radiomics_df[radiomics_df.id.isin(train_pids)]
            val_df = radiomics_df[radiomics_df.id.isin(test_pids)]

            rank_dict, tot_elapsed_time, tot_peak_mem = iterative_bsfs(train_df, estimator)

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

            outdir = os.path.join(OUT_DIR, "wrapper", estimator[-1].__class__.__name__)
            os.makedirs(outdir, exist_ok=True)
        
            out_path = os.path.join(outdir, f"{perturb_id}.npz")
            np.savez_compressed(out_path, rank_dict=np.array(rank_dict, dtype=object), predictions=predictions, targets=targets, elapsed_time=tot_elapsed_time, peak_memory=tot_peak_mem)

if __name__ == "__main__":
    main()
