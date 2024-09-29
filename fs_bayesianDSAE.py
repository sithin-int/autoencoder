import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time as time
import copy as copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as utils
import similarity_index as similarity_index

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

import tracemalloc
import GPUtil

XL_PATH = r"inputs/radiomicsFeatures.csv"
OUT_DIR = r"outputs_new/bayesianDSAE"
MASK_FEATS = ["id", "label"]

CUDA_DEVICE_ID = 2
NUM_REPEATS = 100

B = 100

feats_df = pd.read_csv(XL_PATH)
pids = feats_df.id.to_numpy()
labels = feats_df.label.to_numpy()

feats = feats_df.columns[~feats_df.columns.isin(MASK_FEATS)].to_list()

results_df = {**{"outer_seed":[], "exe_time":[], "memory":[], "b":[], "re_mean":[]}, **{"re_"+feat:[] for feat in feats}, **{"label":[]}} # {**dict1, **dict2,...} is a way to merge multiple dictionaries

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

for i in range(NUM_REPEATS):

    print(f"Running for repeat#- {i+1}")
    print("-"*50)

    start_time = time.time()
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

    X_norm, X_anomaly = utils.norm_anomaly_split(X, y)
    
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

    train_ds = utils.Dataset(X_train)
    val_ds = utils.Dataset(X_train)
    dls = {"train":torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),"val":torch.utils.data.DataLoader(val_ds, batch_size=batch_size)}
    
    bayesian_dsae = utils.bayesianAutoencoder(input_dim, encoder_layers=encoder_layers, latent_dim=latent_dim, activation_fn = activation_fn, dropout_prob=0.5)
    model = utils.Model(bayesian_dsae)
    model.compile(lr, h_lambda, loss_fn, cuda_device_id=CUDA_DEVICE_ID)
    _ = model.fit(dls, num_epochs, verbose=False)

    gpu_mem = GPUtil.getGPUs()[CUDA_DEVICE_ID].memoryUsed
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exe_time = time.time()-start_time

    for b in range(B):

        model.net.train() #to enable dropout for stochasticity during inference
        
        recon_X_test_norm, h_norm = model.net(X_test_norm)
        recon_X_test_anomaly, h_anomaly = model.net(X_test_anomaly)

        recon_X_test = torch.cat([recon_X_test_norm, recon_X_test_anomaly])
        y_test = torch.cat([torch.zeros(len(recon_X_test_norm)), torch.ones(len(recon_X_test_anomaly))])
        
        re_test = nn.MSELoss(reduction="none")(recon_X_test, X_test)

        for re_row, label in zip(re_test, y_test):
            results_df["outer_seed"].append(i)
            results_df["exe_time"].append(exe_time)
            results_df["memory"].append(gpu_mem + (peak/2**20))
            results_df["b"].append(b)
            results_df["re_mean"].append(re_row.mean().item())
    
            for feat, re_feat in zip(feats, re_row):
                results_df["re_"+feat].append(re_feat.item())
    
            results_df["label"].append(label.item())

        _df = pd.DataFrame(results_df)
        grp_mean_df = _df[(_df.outer_seed==i)&(_df.b==b)].groupby(by=["label"]).mean()
        
        print("b=", b, "normal_mse=",grp_mean_df.loc[0].re_mean, "anomaly_mse=", grp_mean_df.loc[1].re_mean, "anomaly_mse>normal_mse=", grp_mean_df.loc[1].re_mean>grp_mean_df.loc[0].re_mean)
       

    _df = pd.DataFrame(results_df)
    grp_mean_df = _df[_df.outer_seed==i].groupby(by=["label"]).mean()
    
    print("normal_mse=",grp_mean_df.loc[0].re_mean, "anomaly_mse=", grp_mean_df.loc[1].re_mean, "anomaly_mse>normal_mse=", grp_mean_df.loc[1].re_mean>grp_mean_df.loc[0].re_mean)

    grp_mean_df = grp_mean_df[["re_"+feat for feat in feats]]
    delta = grp_mean_df.loc[1] - grp_mean_df.loc[0]

    rank = len(delta) - (delta.argsort().argsort() + 1) + 1
    rank_df = pd.DataFrame({"feature":feats, "rank":rank})
    rank_df.to_csv(os.path.join(OUT_DIR, f"rank_df{i}.csv"), index=False)
    
    
results_df = pd.DataFrame(results_df) 
results_df.to_csv(os.path.join(OUT_DIR, "results_df.csv"), index=False)