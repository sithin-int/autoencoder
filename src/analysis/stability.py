#%%
import os
import sys
sys.path.append("..")

import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import similarity_index

ROOT_FOLDER_NAME = "autoencoder"
dirs = os.getcwd().split(os.path.sep)
index = dirs.index(ROOT_FOLDER_NAME)
ROOT_DIR = os.path.sep.join(dirs[:index + 1])
XL_PATH = os.path.join(ROOT_DIR, "inputs", "radiomicsFeaturesWithLabels.csv")
DATA_DIR = os.path.join(ROOT_DIR, "outputs")
OUT_DIR = os.path.join(DATA_DIR, "analysis")

RADIOMICS_DF = pd.read_csv(XL_PATH)
FS_METHODS = ["filtering/mannwhitneyu", "filtering/mrmr_classif", "filtering/mutual_info_classif"]
SIMILARITY_METHODS= {"jaccard":similarity_index.jaccard, "dice":similarity_index.dice, "kuncheva":similarity_index.kuncheva, "mwm":similarity_index.mwm}

NUM_DATA_PERTURBATIONS = 100 #1 to 100, 0 is ignored
TOP_K_LIST = [5, 10, 15, 20, 25]


stability_df = {"fs_method":[], "similarity_measure":[], "top_k":[], "estimate":[]}
for fs_method in tqdm(FS_METHODS):
    for i in range(NUM_DATA_PERTURBATIONS):
        for j in range(i+1, NUM_DATA_PERTURBATIONS):
            df1 = pd.DataFrame(np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)["rank_dict"].item())
            df2 = pd.DataFrame(np.load(os.path.join(DATA_DIR, fs_method, f"{j+1}.npz"), allow_pickle=True)["rank_dict"].item())
            
            for k in TOP_K_LIST:
                
                for similarity_measure, similarity_func in SIMILARITY_METHODS.items():

                    estimate = similarity_func(df1=df1, df2=df2, k=k, feats_df = RADIOMICS_DF)
                    
                    stability_df["fs_method"].append(fs_method)
                    stability_df["similarity_measure"].append(similarity_measure)
                    stability_df["top_k"].append(k)
                    stability_df["estimate"].append(estimate)

            estimate = similarity_index.global_spearman(df1, df2)

            stability_df["fs_method"].append(fs_method)
            stability_df["similarity_measure"].append("global_spearman")
            stability_df["top_k"].append("NA")
            stability_df["estimate"].append(estimate)
                    
            

