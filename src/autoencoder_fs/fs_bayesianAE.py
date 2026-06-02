#%%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
import threading

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
            
class TraceGPUAllocation:
    def __init__(self, device):
        self.keep_measuring = True
        self.peak_gpu_mem = 0
        self.device = device
        
        if "cuda" in device.type:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            self.get_memory_func = lambda device: torch.cuda.memory_allocated(device)
        elif "mps" in device.type:
            torch.mps.empty_cache()
            self.get_memory_func = lambda device: torch.mps.current_allocated_memory()
        else:
            self.get_memory_func = lambda device: 0.0

        self.start_mem = self.get_memory_func(self.device)

    def start(self):
        self.thread = threading.Thread(target=self.trace_memory, daemon=True)
        self.thread.start()

    def trace_memory(self):
        while self.keep_measuring:
            # Check current memory
            current_mem = self.get_memory_func(self.device) - self.start_mem
            # Update peak if current is higher
            if current_mem > self.peak_gpu_mem:
                self.peak_gpu_mem = current_mem
            # Pause briefly to prevent the while-loop from hogging the CPU
            time.sleep(0.001)

    def get_traced_memory(self):
        return self.peak_gpu_mem

    def stop(self):
        self.keep_measuring = False
        self.thread.join()
        
def profile_time_and_memory(device):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()
    
            gc.collect()
            tracemalloc.start()

            tracegalloc = TraceGPUAllocation(device)
            tracegalloc.start()

            result = func(*args, **kwargs)
            
            gpu_mem = tracegalloc.get_traced_memory()
            tracegalloc.stop()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            _, cpu_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            
            peak_mem = cpu_mem + gpu_mem

            return result, elapsed_time, peak_mem, cpu_mem, gpu_mem
        return wrapper
    return decorator

@manual_seed(SEED)
@profile_time_and_memory(device=DEVICE)
def bayesian_dsae_fs(df, mc_samples):
    features = [feat for feat in df.columns if feat not in NON_FEATURE_COLS]

    # Bayesian DSAE Hyperparameters
    num_epochs = 1000
    batch_size = 32
    loss_fn = nn.MSELoss()
    lr = 1e-3
    l1_lambda = 1e-2 
    input_dim = len(features)
    latent_dim = 10
    hidden_dims= [50, 30, 20] 


    X_df = df[features]
    y_df = df[LABEL]

    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()

    X_norm, y_norm, X_anomaly, y_anomaly = nn_utils.norm_anomaly_split(X, y)
    
    X_train = X_norm[:-len(X_anomaly)]
    y_train = y_norm[:-len(X_anomaly)]

    X_test = np.concatenate([X_norm[-len(X_anomaly):], X_anomaly], axis=0)
    y_test = np.concatenate([y_norm[-len(X_anomaly):], y_anomaly], axis=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = np.clip(X_train, -3, 3)
    
    X_test = scaler.transform(X_test)
    X_test = np.clip(X_test, -3, 3)
    
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
    
    bayesian_dsae = nn_utils.bayesianAutoencoder(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    model = nn_utils.Model(bayesian_dsae)
    model.compile(lr, l1_lambda, loss_fn, device=DEVICE)
    model.fit(dls, num_epochs, verbose=False)
    
    # Monte Carlo Inference Phase
    model.net.train() # Keeps dropout active for stochasticity 
    
    mc_deltas = []
    
    with torch.no_grad():
        for _ in range(mc_samples):
            recon_X_test, _ = model.net(X_test)
            
            re_test = nn.MSELoss(reduction="none")(recon_X_test, X_test)
            re_test0 = re_test[y_test == 0].mean(dim=0)
            re_test1 = re_test[y_test == 1].mean(dim=0)
            
            deltas = (re_test1 - re_test0).cpu().numpy()
            mc_deltas.append(deltas)

    del bayesian_dsae, model, X_train, y_train, X_test, y_test, train_ds, val_ds, dls
    if "cuda" in DEVICE.type:   
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif "mps" in DEVICE.type:
        torch.mps.empty_cache()
    else:
        pass
            
    # Calculate the mean score across the B iterations
    mean_deltas = np.mean(mc_deltas, axis=0)
    
    rank_dict = {"feature": features, "score": mean_deltas}
    rank_df = pd.DataFrame(rank_dict)
    
    rank_df["rank"] = rank_df["score"].rank(method="min", ascending=False).astype(int) 
    rank_df.sort_values("rank", inplace=True)
    rank_df.reset_index(drop=True, inplace=True)

    return rank_df.to_dict(orient="list")

def main():

    radiomics_df = pd.read_csv(XL_PATH)
    perturbations = np.load(PERTURBATIONS_FILE, allow_pickle=True).item()
    
    method_name = "bayesianAE"
    outdir = os.path.join(OUT_DIR, "filter", method_name)
    os.makedirs(outdir, exist_ok=True)

    for perturb_id in tqdm(perturbations, desc=f"Running data-perturbation on {method_name}", position=0):
        
        train_pids = perturbations[perturb_id]["train"]
        test_pids = perturbations[perturb_id]["val"]

        train_df = radiomics_df[radiomics_df.id.isin(train_pids)]
        val_df = radiomics_df[radiomics_df.id.isin(test_pids)]

        rank_dict, elapsed_time, peak_mem, cpu_mem, gpu_mem = bayesian_dsae_fs(train_df, mc_samples=B)

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

# #%%
# # sanity check
# import numpy as np
# import os
# from sklearn.metrics import roc_auc_score

# root_dir = "/Users/sithin/research/phd/autoencoder"

# fs_method = "filter/bayesianAE"
# _ = np.load(os.path.join(root_dir, "outputs", fs_method, "0.npz"), allow_pickle=True)


# print(_)

# rank_df = pd.DataFrame(_["rank_dict"].item())
# predictions = _["predictions"]
# targets = _["targets"]

# print(roc_auc_score(targets, predictions))
# display(rank_df.head())
# display(rank_df.tail())
# print(_["elapsed_time"], _["peak_memory"]/2**20, _["cpu_mem"]/2**20, _["gpu_mem"]/2**20)
