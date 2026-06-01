#%%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from functools import wraps
import random
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")

import gc
import time
import tracemalloc

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import nn_utils

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
B = 100 # Monte Carlo samples
SEED = 42

if torch.cuda.is_available():
    CUDA_DEVICE_ID = 0
    DEVICE = torch.device(f"cuda:{CUDA_DEVICE_ID}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using execution device: {DEVICE}")

def manual_seed(seed):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.mps.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            return func(*args, **kwargs)
        return wrapper
    return decorator
            

def profile_time_and_memory(device):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
    
            gc.collect()
            tracemalloc.start()

            if "cuda" in device.type:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                get_memory_func = lambda device: torch.cuda.memory_reserved(device)
            elif "mps" in device.type:
                torch.mps.empty_cache()
                get_memory_func = lambda device: torch.mps.driver_allocated_memory() #driver_allocated_memory()
            else:
                get_memory_func = lambda device: 0.0

            start_gpu_memory = get_memory_func(device=device)
            result = func(*args, **kwargs)
        

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            _, cpu_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_gpu_memory = get_memory_func(device=device)
            gpu_mem = end_gpu_memory-start_gpu_memory
            peak_mem = cpu_mem + gpu_mem

            return result, elapsed_time, peak_mem, cpu_mem, gpu_mem
        return wrapper
    return decorator

@manual_seed(SEED)
@profile_time_and_memory(device=DEVICE)
def ensemble_dsae_fs(df, num_ensembles):
    features = [feat for feat in df.columns if feat not in NON_FEATURE_COLS]

    # Ensemble DSAE Hyperparameters
    num_epochs = 1000
    batch_size = 32
    loss_fn = nn.MSELoss()
    lr = 1e-3
    l1_lambda = 1e-2 
    input_dim = len(features)
    latent_dim = 10
    hidden_dims = [50, 30, 20] 

    X_df = df[features]
    y_df = df[LABEL]

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()

    # Split into normal vs anomaly
    X_norm, y_norm, X_anomaly, y_anomaly = nn_utils.norm_anomaly_split(X, y)
    
    # Train B independent Autoencoders on resampled normal data
    ensemble_deltas = []
    num_anomaly = len(X_anomaly)
    for b in range(num_ensembles):
        # Local seed to ensure consistent sub-sampling per iteration b
        print(f"Running for ensemble # {b+1}/{num_ensembles}...")
        np.random.seed(b) 
        idx = np.random.permutation(len(X_norm))
        
        # Sub-sample train/test for this specific ensemble model
        X_train = X_norm[idx[:-num_anomaly]]
        y_train = y_norm[idx[:-num_anomaly]]

        X_test = X_norm[idx[-num_anomaly:]]
        y_test = y_norm[idx[-num_anomaly:]]

        X_test = np.concatenate([X_test, X_anomaly], axis=0)
        y_test = np.concatenate([y_test, y_anomaly], axis=0)

        # Standardize and clip
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = np.clip(X_train, -3, 3)
        
        X_test = scaler.transform(X_test)
        X_test = np.clip(X_test, -3, 3)
        
        # Convert to Tensors
        X_train = torch.from_numpy(X_train).float().to(DEVICE)
        y_train = torch.from_numpy(y_train).float().to(DEVICE)

        X_test = torch.from_numpy(X_test).float().to(DEVICE)
        y_test = torch.from_numpy(y_test).float().to(DEVICE)

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        val_ds = torch.utils.data.TensorDataset(X_test, y_test)

        dls = {
            "train": torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        }
        
        # Initialize and Train Standard Autoencoder
        dsae = nn_utils.Autoencoder(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        model = nn_utils.Model(dsae)
        model.compile(lr, l1_lambda, loss_fn, device=DEVICE)
        model.fit(dls, num_epochs, verbose=False)
        
        # Evaluate Reconstruction Errors
        model.net.eval()
        with torch.no_grad():
            recon_X_test, _ = model.net(X_test)
            
            re_test = nn.MSELoss(reduction="none")(recon_X_test, X_test)
            re_test0 = re_test[y_test == 0].mean(dim=0)
            re_test1 = re_test[y_test == 1].mean(dim=0)
            
            deltas = (re_test1 - re_test0).cpu().numpy()
            ensemble_deltas.append(deltas)

        del model, dsae, dls, train_ds, val_ds, X_train, y_train, X_test, y_test
        if "cuda" in DEVICE.type:
            torch.cuda.empty_cache()
        elif "mps" in DEVICE.type:
            torch.mps.empty_cache()
        else:
            pass
            
    # Calculate the mean score across the B trained models
    mean_deltas = np.mean(ensemble_deltas, axis=0)
    
    rank_dict = {"feature": features, "score": mean_deltas}
    rank_df = pd.DataFrame(rank_dict)
    
    rank_df["rank"] = rank_df["score"].rank(method="min", ascending=False).astype(int) 
    rank_df.sort_values("rank", inplace=True)
    rank_df.reset_index(drop=True, inplace=True)

    return rank_df.to_dict(orient="list")

def main():

    radiomics_df = pd.read_csv(XL_PATH)
    perturbations = np.load(PERTURBATIONS_FILE, allow_pickle=True).item()
    
    method_name = "ensembleAE"
    outdir = os.path.join(OUT_DIR, "filtering", method_name)
    os.makedirs(outdir, exist_ok=True)

    for perturb_id in tqdm(perturbations, desc=f"Running data-perturbation on {method_name}", position=0):
        
        train_pids = perturbations[perturb_id]["train"]
        test_pids = perturbations[perturb_id]["val"]

        train_df = radiomics_df[radiomics_df.id.isin(train_pids)]
        val_df = radiomics_df[radiomics_df.id.isin(test_pids)]

        # Call the new profiled Ensemble function
        rank_dict, elapsed_time, peak_mem, cpu_mem, gpu_mem = ensemble_dsae_fs(train_df, num_ensembles=B)

        rank_df = pd.DataFrame(rank_dict)
        rank_df.sort_values("rank", inplace=True)
        rank_df.reset_index(drop=True, inplace=True)

        # Select Top N Features
        selected_feats = rank_df.head(NUM_FEATS_TO_SELECT).feature.to_list()

        X_train = train_df[selected_feats]
        y_train = train_df[LABEL]

        X_val = val_df[selected_feats]
        y_val = val_df[LABEL]

        # Downstream Evaluation Model
        pred_model = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10_000, random_state=SEED))
        pred_model.fit(X_train, y_train)
    
        predictions = pred_model.predict_proba(X_val)[:, 1]
        targets = y_val.to_numpy().ravel()

        out_path = os.path.join(outdir, f"{perturb_id}.npz")
        np.savez_compressed(
            out_path, 
            rank_dict=np.array(rank_dict, dtype=object), 
            predictions=predictions, 
            targets=targets, 
            elapsed_time=elapsed_time, 
            peak_memory=peak_mem, 
            cpu_mem=cpu_mem, 
            gpu_mem=gpu_mem
        )

if __name__ == "__main__":
    main()


