#%%
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

ROOT_FOLDER_NAME = "autoencoder"
ROOT_DIR = next(p for p in Path(__file__).parents if p.name == ROOT_FOLDER_NAME)
sys.path.append(os.path.join(ROOT_DIR, "src"))
from utils import similarity_index

XL_PATH = os.path.join(ROOT_DIR, "inputs", "radiomicsFeaturesWithLabels.csv")
DATA_DIR = os.path.join(ROOT_DIR, "outputs")
OUT_DIR = os.path.join(DATA_DIR, "analysis")

RADIOMICS_DF = pd.read_csv(XL_PATH)
FS_METHODS = ["filter/mannwhitneyu", "filter/mutual_info_classif", "filter/mrmr_classif", "embedded/LASSO", "wrapper/LogisticRegression", "wrapper/SVC", "wrapper/RandomForestClassifier", "wrapper/MLPClassifier", "filter/singleAE", "filter/bayesianAE", "filter/ensembleAE"]
# FS_METHODS = ["filter/memory_allocated/singleAE", "filter/memory_allocated/bayesianAE", "filter/memory_allocated/ensembleAE"]

SIMILARITY_METHODS= {"jaccard":similarity_index.jaccard, "dice":similarity_index.dice, "kuncheva":similarity_index.kuncheva, "mwm":similarity_index.mwm}

NUM_DATA_PERTURBATIONS = 100 #total=101, we only use 1 to 100, 0 is ignored
TOP_K_LIST = [5, 10, 15, 20, 25]


#%%
stability_df = {"fs_method":[], "similarity_measure":[], "top_k":[], "estimate":[], "perturb_idx":[]}

for fs_method in tqdm(["random"] + FS_METHODS):
    for i in range(NUM_DATA_PERTURBATIONS):
        for j in range(i+1, NUM_DATA_PERTURBATIONS):
            
            dict1 = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
            dict2 = np.load(os.path.join(DATA_DIR, fs_method, f"{j+1}.npz"), allow_pickle=True)
            
            map_idx = (i * NUM_DATA_PERTURBATIONS) + j # to map the tuple (i,j) to a unique integer
            
            df1 = pd.DataFrame(dict1["rank_dict"].item())
            df2 = pd.DataFrame(dict2["rank_dict"].item())

            for k in TOP_K_LIST:
                
                for similarity_measure, similarity_func in SIMILARITY_METHODS.items():

                    estimate = similarity_func(df1=df1, df2=df2, k=k, feats_df = RADIOMICS_DF)
                    
                    stability_df["fs_method"].append(fs_method)
                    stability_df["similarity_measure"].append(similarity_measure)
                    stability_df["top_k"].append(k)
                    stability_df["estimate"].append(estimate)
                    stability_df["perturb_idx"].append(map_idx)



            estimate = similarity_index.global_spearman(df1, df2)

            stability_df["fs_method"].append(fs_method)
            stability_df["similarity_measure"].append("global_spearman")
            stability_df["top_k"].append(-1)
            stability_df["estimate"].append(estimate)
            stability_df["perturb_idx"].append(map_idx)

os.makedirs(OUT_DIR, exist_ok=True)
stability_df = pd.DataFrame(stability_df)
stability_df.to_csv(os.path.join(OUT_DIR, 'stability.csv'), index=False)

#%%
# Displaying mean stability estimates
mean_stability_df = stability_df.groupby(by=["fs_method", "similarity_measure", "top_k"]).mean()

for fs_method in ["random"] + FS_METHODS:
    print(f"{fs_method}")
    display(mean_stability_df.xs(fs_method, level="fs_method"))
    print("\n")

#%%
# Figure 1
stability_df = pd.read_csv(os.path.join(OUT_DIR, 'stability.csv'))

# Define the custom order and mapping
custom_order = [
    "random",
    "filter/mannwhitneyu",
    "filter/mutual_info_classif",
    "filter/mrmr_classif",
    "embedded/LASSO",
    "wrapper/LogisticRegression",
    "wrapper/SVC",
    "wrapper/RandomForestClassifier",
    "wrapper/MLPClassifier",
    "filter/singleAE",
    "filter/bayesianAE",
    "filter/ensembleAE"
]

label_mapping = {
    "random": "random",
    "filter/mannwhitneyu": "WLCX",
    "filter/mutual_info_classif": "MIM",
    "filter/mrmr_classif": "MRMR",
    "embedded/LASSO": "LASSO",
    "wrapper/LogisticRegression": "SBS+LR",
    "wrapper/SVC": "SBS+L-SVM",
    "wrapper/RandomForestClassifier": "SBS+RF",
    "wrapper/MLPClassifier": "SBS+MLP",
    "filter/singleAE": "singleAE",
    "filter/bayesianAE": "bayesianAE",
    "filter/ensembleAE": "ensembleAE"
}

# Filter and map
plot_data = stability_df[
    (stability_df.similarity_measure.isin(["global_spearman", "kuncheva", "mwm"])) &
    (stability_df.top_k.isin([-1, 5]))
].copy()

plot_data['fs_method'] = pd.Categorical(plot_data['fs_method'], categories=custom_order, ordered=True)
plot_data['fs_method'] = plot_data['fs_method'].map(label_mapping)

sim_mapping = {
    "global_spearman": "spearman (global)",
    "kuncheva": "kuncheva (top-5)",
    "mwm": "mwm (top-5)"
}
plot_data['similarity_measure'] = plot_data['similarity_measure'].map(sim_mapping)

plt.figure(figsize=(18, 6)) # Increased width from 12 to 18 to add spacing between x-axis points


sns.lineplot(
    data=plot_data,
    x='fs_method',
    y='estimate',
    errorbar='sd',
    marker='o',
    hue='similarity_measure',
    style='similarity_measure',
    markers={
        'kuncheva (top-5)': 'o',           # Circle
        'mwm (top-5)': 's',                # Square
        'spearman (global)': 'D'     # Diamond
    },
    hue_order=["spearman (global)", "kuncheva (top-5)", "mwm (top-5)"],
    style_order=["spearman (global)", "kuncheva (top-5)", "mwm (top-5)"]
)

plt.xlabel("Feature Selection Method", fontweight='bold', labelpad=12) # Expanded label slightly
plt.ylabel("Stability Estimate", fontweight='bold', labelpad=12)


plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "stability_plot.tif"), format="tiff", dpi=600)

plt.show()

#%%
# Figure 1 detailed
custom_order = [
    "random", "filter/mannwhitneyu", "filter/mutual_info_classif",
    "filter/mrmr_classif", "embedded/LASSO", "wrapper/LogisticRegression",
    "wrapper/SVC", "wrapper/RandomForestClassifier", "wrapper/MLPClassifier",
    "filter/singleAE", "filter/bayesianAE", "filter/ensembleAE"
]

label_mapping = {
    "random": "random", "filter/mannwhitneyu": "WLCX",
    "filter/mutual_info_classif": "MIM", "filter/mrmr_classif": "MRMR",
    "embedded/LASSO": "LASSO", "wrapper/LogisticRegression": "SBS+LR",
    "wrapper/SVC": "SBS+L-SVM", "wrapper/RandomForestClassifier": "SBS+RF",
    "wrapper/MLPClassifier": "SBS+MLP", "filter/singleAE": "singleAE",
    "filter/bayesianAE": "bayesianAE", "filter/ensembleAE": "ensembleAE"
}

plot_data = stability_df[
    (stability_df.similarity_measure.isin(["global_spearman", "kuncheva", "mwm"])) &
    (stability_df.top_k.isin([-1, 5]))
].copy()

plot_data['fs_method'] = pd.Categorical(plot_data['fs_method'], categories=custom_order, ordered=True)
plot_data['fs_method'] = plot_data['fs_method'].map(label_mapping)

sim_mapping = {
    "global_spearman": "spearman (global)",
    "kuncheva": "kuncheva (top-5)",
    "mwm": "mwm (top-5)"
}
plot_data['similarity_measure'] = plot_data['similarity_measure'].map(sim_mapping)

# --- Plotting ---

plt.figure(figsize=(20, 8))

hue_order = ["spearman (global)", "kuncheva (top-5)", "mwm (top-5)"]

# 1. Violin Plot: Structure maintained, now using refined colors
sns.violinplot(
    data=plot_data,
    x='fs_method',
    y='estimate',
    hue='similarity_measure',
    hue_order=hue_order,
    inner=None,               # decluttered center
    cut=0,            
    linewidth=1,            
    density_norm="width", 
    alpha=0.5                 # soft violins
)

# 2. Stripplot: Points pop cleanly against new colors
sns.stripplot(
    data=plot_data,
    x='fs_method',
    y='estimate',
    hue='similarity_measure',
    hue_order=hue_order,
    dodge=True,       
    alpha=0.9,                # vibrant points
    jitter=True,      
    legend=False,
    size=4.5,               
    linewidth=0.6,          
    edgecolor="white",        # critical white border
    zorder=2                
)

sns.pointplot(
    data=plot_data,
    x='fs_method',
    y='estimate',
    hue='similarity_measure',
    hue_order=hue_order,
    markers = ['D', 'o', 's'],
    linestyles=['-', '--', ':'],
    estimator='mean',          
    errorbar=None,             
    dodge=0.533,               # <--- FIXED: Mathematical dodge value for 3 hues
    linewidth=2.5,             
    markeredgecolor='white',   
    markeredgewidth=1.5,
    zorder=3,                  
    legend=False               
)

# --- 4. Aesthetics & Cleanup ---
plt.xlabel("Feature Selection Method", fontweight='bold', labelpad=12) # Expanded label slightly
plt.ylabel("Stability Estimate", fontweight='bold', labelpad=12)

# plt.xticks(rotation=45, ha='right')

# Legend: removed redundant borders and simplified title
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[:3], 
    labels[:3], 
    title='Stability Metric',  # Made title more precise
    bbox_to_anchor=(1.02, 1), 
    loc='upper left',
    frameon=False
)

# Remove unneeded chart boundaries
sns.despine()

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "stability_plot_violin.tif"), format="tiff", dpi=600, bbox_inches="tight")
plt.show()

#%%
# Time and Memory Usage

usage_df = {"perturb_idx":[], "fs_method":[], "time":[], "peak_memory":[]}

for fs_method in FS_METHODS:
    for i in range(NUM_DATA_PERTURBATIONS):
        dict_i = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
        
        usage_df["perturb_idx"].append(i+1)
        usage_df["fs_method"].append(fs_method)
        usage_df["time"].append(dict_i["elapsed_time"])
        usage_df["peak_memory"].append(dict_i["peak_memory"]/2**20)

usage_df = pd.DataFrame(usage_df)
usage_df.to_csv(os.path.join(OUT_DIR, 'usage.csv'), index=False)

print("min usage***")
# Displaying mean usage estimates
agg_usage_df = usage_df.groupby(by=["fs_method"]).agg(list)

for fs_method in FS_METHODS:
    print(f"{fs_method}")
    for attribute in ["time", "peak_memory"]:
        arr = np.array(agg_usage_df.loc[fs_method][attribute])
        min, max, mean = arr.min().item(), arr.max().item(), arr.mean().item()
        print(f"{attribute}: {mean:.4f} (min: {min:.4f}, max: {max:.4f})")
    print("\n")

#%%
# Time and Memory Usage plot
df = usage_df.copy()
# sns.set(style="whitegrid") # Removed invalid palette="gray"

# Convert time from seconds to minutes
df['time'] = df['time'] / 60.0

custom_order = [
    "filter/mannwhitneyu", "filter/mutual_info_classif",
    "filter/mrmr_classif", "embedded/LASSO", "wrapper/LogisticRegression",
    "wrapper/SVC", "wrapper/RandomForestClassifier", "wrapper/MLPClassifier",
    "filter/singleAE", "filter/bayesianAE", "filter/ensembleAE"
]

label_mapping = {
    "filter/mannwhitneyu": "WLCX",
    "filter/mutual_info_classif": "MIM", "filter/mrmr_classif": "MRMR",
    "embedded/LASSO": "LASSO", "wrapper/LogisticRegression": "SBS+LR",
    "wrapper/SVC": "SBS+L-SVM", "wrapper/RandomForestClassifier": "SBS+RF",
    "wrapper/MLPClassifier": "SBS+MLP", "filter/singleAE": "singleAE",
    "filter/bayesianAE": "bayesianAE", "filter/ensembleAE": "ensembleAE"
}

# Apply label mapping to the order list so the boxplot knows the new names
mapped_order = [label_mapping[m] for m in custom_order if m in df['fs_method'].unique()]

df['fs_method'] = df['fs_method'].map(label_mapping)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Top-left: Boxplot of Execution time
sns.boxplot(
    x='time', y='fs_method', data=df, ax=axs[0],
    order=mapped_order, # Added order so methods are sorted correctly
    color='white', fliersize=3, linewidth=1, width=0.4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black')
)
sns.stripplot(
    x='time', y='fs_method', data=df, ax=axs[0],
    order=mapped_order,
    color='none', edgecolor='black', linewidth=1, alpha=0.1, size=3, jitter=0.05
)
axs[0].set_xlabel("Execution Time (Minutes)", fontweight='bold', labelpad=12)
axs[0].set_ylabel("Feature Selection Method", fontweight='bold', labelpad=12)

# Top-right: Boxplot of Memory usage

sns.boxplot(
    x='peak_memory', y='fs_method', data=df, ax=axs[1],
    order=mapped_order, # Added order so methods are sorted correctly
    color='white', fliersize=3, linewidth=1, width=0.4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    medianprops=dict(color='black')
)

sns.stripplot(
    x='peak_memory', y='fs_method', data=df, ax=axs[1],
    order=mapped_order,
    color='none', edgecolor='black', linewidth=1, alpha=0.1, size=3, jitter=0.05
)


axs[1].set_xlabel("Peak Memory (MiB)", fontweight='bold', labelpad=12)
axs[1].set_ylabel("") # Remove y-label for the second plot since they align


for ax in axs.flat:
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "complexity_plot.tif"), format="tiff", dpi=600)
plt.show()
#%%
# Measuring the prediction performance of the signatures generated by the pipeline
results_df = {"perturb_idx":[], "fs_method":[], "predictions":[], "targets":[], "auc_scores":[]}

for fs_method in FS_METHODS:
    for i in range(NUM_DATA_PERTURBATIONS):
        dict_i = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
        
        results_df["perturb_idx"].append(i)
        results_df["fs_method"].append(fs_method)
        results_df["predictions"].append(dict_i["predictions"])
        results_df["targets"].append(dict_i["targets"])
        results_df["auc_scores"].append(roc_auc_score(dict_i["targets"], dict_i["predictions"]))

results_df = pd.DataFrame(results_df)
results_df.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)
agg_results_df = results_df.groupby(by=["fs_method"]).agg(list)

for fs_method in FS_METHODS:
    print(f"{fs_method}")

    micro_auc = roc_auc_score(np.concatenate(agg_results_df.loc[fs_method, "targets"]), np.concatenate(agg_results_df.loc[fs_method, "predictions"]))
    macro_auc = np.mean(agg_results_df.loc[fs_method, "auc_scores"])
    macro_auc_std = np.std(agg_results_df.loc[fs_method, "auc_scores"])
    
    print(f"Micro AUC: {micro_auc:.4f}")
    print(f"Macro AUC: {macro_auc:.4f} (+-{macro_auc_std:.4f})")
    print("\n")




#%%
# Selecting the signature
top_k = 5

selected_feats_df = {"fs_method":[], "selected_feats":[]}
for fs_method in FS_METHODS:

    rank_df = pd.read_csv(os.path.join(DATA_DIR, fs_method, "rank_df.csv"))
    selected_feats = rank_df.sort_values(by="rank").head(top_k).feature.to_list()
    selected_feats_df["fs_method"].append(fs_method)
    selected_feats_df["selected_feats"].append(selected_feats)

selected_feats_df = pd.DataFrame(selected_feats_df)
selected_feats_df.to_csv(os.path.join(OUT_DIR, "selected_feats_df.csv"), index=False)

display(selected_feats_df)

import seaborn as sns
import matplotlib.pyplot as plt


# Seaborn styling for a cleaner look
# sns.set(style="whitegrid", context="notebook", font_scale=1.1)
df_exploded = selected_feats_df.explode("selected_feats").reset_index(drop=True)
# Pivot table creation (as you already did)
pivot = pd.crosstab(df_exploded["fs_method"], df_exploded["selected_feats"])

# Sort features by frequency (most to least)
top_features = pivot.sum().sort_values(ascending=False).index
pivot = pivot[top_features]

label_mapping = {
    "filter/mannwhitneyu":"WLCX",
    "filter/mrmr_classif":"MRMR",
    "filter/mutual_info_classif":"MIM",
    "embedded/LASSO":"LASSO",
    "wrapper/LogisticRegression": "SBS+LR",
    "wrapper/SVC": "SBS+L-SVM",
    "wrapper/RandomForestClassifier": "SBS+RF",
    "wrapper/MLPClassifier": "SBS+MLP",
    "filter/singleAE":"singleAE", 
    "filter/bayesianAE": "bayesianAE", 
    "filter/ensembleAE": "ensembleAE"
}

pivot = pivot.reindex(FS_METHODS)


# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot 'x' markers for each selected feature-method pair
for i, method in enumerate(pivot.index):
    for j, feat in enumerate(pivot.columns):
        if pivot.loc[method, feat]:
            ax.scatter(j, i, marker='x', color='black', s=60, linewidths=1.5)

# Format axes

y_labels = [label_mapping.get(method, method) for method in pivot.index]
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(y_labels, fontsize=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)

# Add labels and styling
ax.set_xlabel("Top-5 Selected Features", fontweight='bold', labelpad=12)
ax.set_ylabel("Feature Selection Method", fontweight='bold', labelpad=12)
# ax.set_title("Top Features Selected by Different Methods", fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.tick_params(axis='both', which='major', length=0)

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "signatures.tif"), format="tiff", dpi=600)


plt.show()


#%%
# Identifying the frequent features

top_k = 5
frequent_feats_df = {"fs_method":[], "frequent_feats":[]}

for fs_method in FS_METHODS:
    
    freq_dict = {}
    
    for i in range(NUM_DATA_PERTURBATIONS):

        dict_i = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
        rank_df = pd.DataFrame(dict_i["rank_dict"].item())
        selected_feats = rank_df.sort_values(by="rank").head(top_k).feature.to_list()
        
        for feat in selected_feats:
            freq_dict[feat] = freq_dict.get(feat,0)+1

    feats, freq = zip(*freq_dict.items())
    
    freq_df = pd.DataFrame({"feature":feats, "frequency":freq})
    
    frequent_feats_df["fs_method"].append(fs_method)
    frequent_feats_df["frequent_feats"].append(sorted(freq_df.sort_values(by="frequency", ascending=False).head(top_k).feature.to_list()))
    

frequent_feats_df = pd.DataFrame(frequent_feats_df)
frequent_feats_df.to_csv(os.path.join(OUT_DIR, "frequent_feats_df.csv"), index=False)

display(frequent_feats_df)


import seaborn as sns
import matplotlib.pyplot as plt


# Seaborn styling for a cleaner look
sns.set(style="whitegrid", context="notebook", font_scale=1.1)
df_exploded = frequent_feats_df.explode("frequent_feats").reset_index(drop=True)
# Pivot table creation (as you already did)
pivot = pd.crosstab(df_exploded["fs_method"], df_exploded["frequent_feats"])

# Sort features by frequency (most to least)
top_features = pivot.sum().sort_values(ascending=False).index
pivot = pivot[top_features]

label_mapping = {
    "filter/mannwhitneyu":"WLCX",
    "filter/mrmr_classif":"MRMR",
    "filter/mutual_info_classif":"MIM",
    "embedded/LASSO":"LASSO",
    "wrapper/LogisticRegression": "SBS+LR",
    "wrapper/SVC": "SBS+L-SVM",
    "wrapper/RandomForestClassifier": "SBS+RF",
    "wrapper/MLPClassifier": "SBS+MLP",
    "filter/singleAE":"singleAE", 
    "filter/bayesianAE": "bayesianAE", 
    "filter/ensembleAE": "ensembleAE"
}

pivot = pivot.reindex(FS_METHODS)


# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot 'x' markers for each selected feature-method pair
for i, method in enumerate(pivot.index):
    for j, feat in enumerate(pivot.columns):
        if pivot.loc[method, feat]:
            ax.scatter(j, i, marker='x', color='black', s=60, linewidths=1.5)

# Format axes

y_labels = [label_mapping.get(method, method) for method in pivot.index]
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(y_labels, fontsize=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)

# Add labels and styling
ax.set_xlabel("Top-5 Frequent Features", fontweight='bold', labelpad=12)
ax.set_ylabel("Feature Selection Method", fontweight='bold', labelpad=12)
# ax.set_title("Top Features Selected by Different Methods", fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.tick_params(axis='both', which='major', length=0)

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "frequent_feats.tif"), format="tiff", dpi=600)


plt.show()




#%%
## Comparing stability of best performing FS method from each family with best AE-FS method using Wilcoxon Signed Rank Test
from scipy import stats


# for fs_method1 in FS_METHODS:
#     for fs_method2 in FS_METHODS:
#         for similariy_measure in SIMILARITY_METHODS:

#             x = stability_df[(stability_df.fs_method==fs_method1) & (stability_df.top_k==5) & (stability_df.similarity_measure==similariy_measure)]
#             y = stability_df[(stability_df.fs_method==fs_method2) & (stability_df.top_k==5) & (stability_df.similarity_measure==similariy_measure)]
#             display(x)


#             break;
#         break;
#     break;

        
        

