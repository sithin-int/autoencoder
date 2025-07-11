import sys
sys.path.append("..") #to access custom "utils" package

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time as time
import copy as copy
import gc
import tracemalloc
import GPUtil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import nn_utils
from utils import similarity_index

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

XL_PATH = os.path.join("..", r"inputs/radiomicsFeatures.csv")
OUT_DIR = r"outputs/oneDSAE"
MASK_FEATS = ["id", "label"]

VERBOSE = False

CUDA_DEVICE_ID = 1
NUM_REPEATS = 100

feats_df = pd.read_csv(XL_PATH)

pids = feats_df.id.to_numpy()
labels = feats_df.label.to_numpy()

def get_gpu_memory(init_mem):
    
    global CUDA_DEVICE_ID
    
    return (GPUtil.getGPUs()[CUDA_DEVICE_ID].memoryUsed-init_mem) #in MiB

### Feature Selection Pipeline with MonteCarlo Resampling
init_gpu_memory = get_gpu_memory(0.0) #in MiB

feats = feats_df.columns[~feats_df.columns.isin(MASK_FEATS)].to_list()

results_df = {**{"outer_seed":[], "exe_time":[], "cpu_mem":[], "gpu_mem":[], "re_mean0":[], "re_mean1":[]}, **{"delta_"+feat:[] for feat in feats}} # {**dict1, **dict2,...} is a way to merge multiple dictionaries

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

for i in tqdm(range(NUM_REPEATS), position=0, desc="running oneDSAE"):

    if VERBOSE:
        print(f"Running for repeat#- {i+1}")
        print("-"*50)

    start_time = time.perf_counter()
    gc.collect()
    tracemalloc.start()

    num_epochs = 1_000
    batch_size = 32
    loss_fn = nn.MSELoss()
    
    lr = 1e-3
    h_lambda = 1e-2 #with l1 regularization
    
    input_dim = len(feats)
    latent_dim = 10
    
    activation_fn = nn.LeakyReLU()
    encoder_layers = [50, 30, 20] #under-complete hidden layers

    train_pids, test_pids, train_labels, test_labels = train_test_split(pids, labels, test_size=0.25, random_state=i, stratify=labels)

    
    X =  feats_df[feats_df["id"].isin(train_pids)][feats].to_numpy()
    y = feats_df[feats_df["id"].isin(train_pids)].label.to_numpy()

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X[X>=3] = 3
    # X[X<=-3] = -3

    X_norm, X_anomaly = nn_utils.norm_anomaly_split(X, y)
    
    np.random.seed(0)
    idx = np.random.permutation(len(X_norm))
    
    X_train= X_norm[idx[:-len(X_anomaly)]]
    X_test_norm = X_norm[idx[-len(X_anomaly):]]
    X_test_anomaly = X_anomaly

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train[X_train>=3] = 3
    X_train[X_train<=-3] = -3
    
    X_test_norm = scaler.transform(X_test_norm)
    X_test_norm[X_test_norm>=3] = 3
    X_test_norm[X_test_norm<=-3] = -3
    
    X_test_anomaly = scaler.transform(X_test_anomaly)
    X_test_anomaly[X_test_anomaly>=3] = 3
    X_test_anomaly[X_test_anomaly<=-3] = -3
    
    
    X_train =  torch.from_numpy(X_train).float()
    X_test_norm = torch.from_numpy(X_test_norm).float()
    X_test_anomaly = torch.from_numpy(X_test_anomaly).float()
    X_test = torch.cat([X_test_norm, X_test_anomaly])

    train_ds = nn_utils.Dataset(X_train)
    val_ds = nn_utils.Dataset(X_train)
    dls = {"train":torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),"val":torch.utils.data.DataLoader(val_ds, batch_size=batch_size)}
    
    dsae = nn_utils.Autoencoder(input_dim, encoder_layers=encoder_layers, latent_dim=latent_dim, activation_fn = activation_fn)
    model = nn_utils.Model(dsae)
    model.compile(lr, h_lambda, loss_fn, cuda_device_id=CUDA_DEVICE_ID)
    _ = model.fit(dls, num_epochs, verbose=False)

    gpu_mem = get_gpu_memory(init_gpu_memory) * 2**20
    current, cpu_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    exe_time = time.perf_counter()-start_time
    
    model.net.eval()
    
    recon_X_test_norm, h_norm = model.net(X_test_norm)
    recon_X_test_anomaly, h_anomaly = model.net(X_test_anomaly)

    recon_X_test = torch.cat([recon_X_test_norm, recon_X_test_anomaly])
    y_test = torch.cat([torch.zeros(len(recon_X_test_norm)), torch.ones(len(recon_X_test_anomaly))])
    
    re_test = nn.MSELoss(reduction="none")(recon_X_test, X_test)
    re_test0 = re_test[y_test==0].mean(dim=0)
    re_test1 = re_test[y_test==1].mean(dim=0)
    
    deltas = re_test1 - re_test0
    
    results_df["outer_seed"].append(i)
    results_df["exe_time"].append(exe_time)
    results_df["cpu_mem"].append(cpu_mem)
    results_df["gpu_mem"].append(gpu_mem)
    results_df["re_mean0"].append(re_test0.mean().item())
    results_df["re_mean1"].append(re_test1.mean().item())
    
    for feat, delta in zip(feats, deltas):
        results_df["delta_"+feat].append(delta.item())

    if VERBOSE:
        print("normal_mse=", re_test0.mean().item(), "anomaly_mse=", re_test1.mean().item(), "anomaly_mse>normal_mse=", re_test1.mean().item()>re_test0.mean().item())
    
    _df = pd.DataFrame(results_df)
    _results_df = _df[_df.outer_seed==i].mean()

   
    
    delta_df = _results_df[["delta_"+feat for feat in feats]]
    
    ranks = (len(delta_df) - (delta_df.argsort().argsort() + 1) + 1).to_list()
    
    rank_df = pd.DataFrame({"feature":feats, "rank":ranks})
    rank_df.to_csv(os.path.join(OUT_DIR, f"rank_df{i}.csv"), index=False)

 
    
results_df = pd.DataFrame(results_df) 
results_df.to_csv(os.path.join(OUT_DIR, "results_df.csv"), index=False)

print("Completed successfully")