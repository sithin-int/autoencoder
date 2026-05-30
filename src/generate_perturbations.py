from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT_FOLDER_NAME = "autoencoder"
dirs = os.getcwd().split(os.path.sep)
index = dirs.index(ROOT_FOLDER_NAME)
ROOT_DIR = os.path.sep.join(dirs[:index + 1])

XL_PATH = os.path.join(ROOT_DIR, "inputs", "radiomicsFeaturesWithLabels.csv")
PERTURBATIONS_FILE = os.path.join(ROOT_DIR, "outputs", "data_perturbations.npy")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")

NUM_DATA_PERTURBATIONS = 101

if __name__=="__main__":

    perturbations = {}

    feats_df = pd.read_csv(XL_PATH)
    labels = feats_df.label.to_numpy()
    pids = feats_df.id.to_numpy()

    for i in tqdm(range(NUM_DATA_PERTURBATIONS), desc="Generating perturbations", position=0):

        train_pids, test_pids, train_labels, test_labels = train_test_split(pids, labels, test_size=0.25, random_state=i, stratify=labels)
        perturbations[i] = {"train":train_pids, "val":test_pids}

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "data_perturbations.npy"), np.array(perturbations, dtype=object), allow_pickle=True)