from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import gc
import tracemalloc

import time
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

XL_PATH = os.path.join("..", r"radiomicsFeatures/radiomicsFeaturesWithLabels.csv")
OUT_DIR = r"outputs/backwardSFS"
MASK_FEATS = ["id", "label"]

NUM_REPEATS = 100

feats_df = pd.read_csv(XL_PATH)
pids = feats_df.id.to_numpy()
labels = feats_df.label.to_numpy()


def run_bsfs(estimator, feats_df):

    global MASK_FEATS

    features = feats_df.columns[~feats_df.columns.isin(MASK_FEATS)].to_list()
    
    n = len(features)
    rank_df = {"feature":[], "rank":[]}
    
    pbar = tqdm(range(n-1),desc=f"Running Backwards SFS with {estimator.__class__.__name__}", position=0)
    
    mem_usage = []
    
    while n>1:
        
        gc.collect()
        tracemalloc.start()
    
        X = feats_df[features].to_numpy()
        y = feats_df["label"].to_numpy().ravel()

        sfs_pipeline = make_pipeline(StandardScaler(), SFS(estimator, n_features_to_select=n-1, direction='backward', scoring="roc_auc", cv=5))
        sfs_pipeline.fit(X,y)
   
        eliminated_feature = np.array(features)[~sfs_pipeline['sequentialfeatureselector'].support_][0]
        rank_df["feature"].append(eliminated_feature)
        rank_df["rank"].append(n)

        features = np.array(features)[sfs_pipeline['sequentialfeatureselector'].support_]

        n -= 1
        
        _, peak_mem = tracemalloc.get_traced_memory()
        mem_usage.append(peak_mem)
        
        tracemalloc.stop()
        
        pbar.update()

    rank_df["feature"].append(features[0])
    rank_df["rank"].append(n)
        
    return pd.DataFrame(rank_df), sum(mem_usage)
    
if __name__=="__main__":
    
    estimators = [LogisticRegression(penalty='none', max_iter=10_000), SVC(kernel="linear", max_iter=10_000, probability=True), RandomForestClassifier(), MLPClassifier(max_iter=10_000)]

    results_df = {"outer_seed":[], "estimator":[], "exe_time":[], "mem_usage":[]}
    for estimator in estimators:

        out_dir = os.path.join(OUT_DIR, estimator.__class__.__name__)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(NUM_REPEATS):

            train_pids, test_pids, train_labels, test_labels = train_test_split(pids, labels, test_size=0.25, random_state=i, stratify=labels)

            print(f"Running for repeat#- {i+1}")
            print("-"*50)

            train_feats_df = feats_df[feats_df["id"].isin(train_pids)]

            start_time = time.perf_counter()
            rank_df, mem_usage = run_bsfs(estimator, train_feats_df)
            exe_time = time.perf_counter() - start_time

            results_df["outer_seed"].append(i)
            results_df["estimator"].append(estimator.__class__.__name__)
            results_df["exe_time"].append(exe_time)
            results_df["mem_usage"].append(mem_usage)

            rank_df.to_csv(os.path.join(out_dir, f"rank_df{i}.csv"), index=False)

        _results_df = pd.DataFrame(results_df)

        _results_df[_results_df.estimator==estimator.__class__.__name__].to_csv(os.path.join(out_dir, "results_df.csv"))

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(os.path.join(OUT_DIR, "results_df.csv"))  
