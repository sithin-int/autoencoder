import sys
sys.path.append("..") #to access custom "utils" package

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tqdm import tqdm


MASK_FEATS = ["id", "label"]
XL_PATH = os.path.join("..", r"radiomicsFeatures/radiomicsFeaturesWithLabels.csv")
OUT_DIR = r"outputs/random"

NUM_REPEATS = 100

data_df = pd.read_csv(XL_PATH)

pids = data_df.id.to_numpy()
labels = data_df.label.to_numpy()

### Feature Selection Pipeline

class RandomFS(object):

    def __init__(self):
        pass

    def __call__(self, feats_df):
        
        features = feats_df.columns.to_list()
        
        ranks = np.arange(len(features))+1
        ranks = np.random.permutation(ranks).tolist()

        rank_df = pd.DataFrame({"feature":features, "rank":ranks})

        return rank_df

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    
for i in tqdm(range(NUM_REPEATS)):
    
    features = data_df.columns[~data_df.columns.isin(MASK_FEATS)].to_list()

    rank_df = RandomFS()(data_df[features])
    rank_df.to_csv(os.path.join(OUT_DIR, f"rank_df{i}.csv"), index=False)