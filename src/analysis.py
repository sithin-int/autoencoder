#%%
%load_ext autoreload
%autoreload 2

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
# Stability Analysis
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
stability_df = pd.read_csv(os.path.join(OUT_DIR, 'stability.csv'))
mean_stability_df = stability_df.groupby(by=["fs_method", "similarity_measure", "top_k"]).mean()

for fs_method in ["random"] + FS_METHODS:
    print(f"{fs_method}")
    display(mean_stability_df.xs(fs_method, level="fs_method"))
    print("\n")



#%%
## Comparing stability of FS methods using Wilcoxon Signed Rank Test
from scipy import stats

import itertools


def compare_fs_methods(fs_methods, stability_df, top_k, similarity_measure='spearman'):
    
    df = pd.DataFrame(np.zeros((len(fs_methods), len(fs_methods))), index=fs_methods, columns=fs_methods)
    
    pairs = list(itertools.product(fs_methods, repeat=2))
    
    for fs_method1, fs_method2 in pairs:

        stability_values1 = stability_df[(stability_df.fs_method == fs_method1) & (stability_df.top_k == top_k) & (stability_df.similarity_measure == similarity_measure)].sort_values(by="perturb_idx").reset_index(drop=True).estimate.to_list()
        stability_values2 = stability_df[(stability_df.fs_method == fs_method2) & (stability_df.top_k == top_k) & (stability_df.similarity_measure == similarity_measure)].sort_values(by="perturb_idx").reset_index(drop=True).estimate.to_list()

        differences = np.array(stability_values1) - np.array(stability_values2)
        if np.all(differences == 0):
            print("Arrays are identical. Wilcoxon test is undefined.")
            df.loc[fs_method1, fs_method2] = 1.0
        else:
            df.loc[fs_method1, fs_method2] = stats.wilcoxon(stability_values1, stability_values2).pvalue

    return df


for simlarity_measure in ["global_spearman", "kuncheva", "mwm"]:

    k = -1 if simlarity_measure == "global_spearman" else 5
    
    wilcoxon_df = compare_fs_methods(FS_METHODS, stability_df, top_k=k, similarity_measure=simlarity_measure)

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

    wilcoxon_df = wilcoxon_df.rename(index=label_mapping, columns=label_mapping)

    fig, axes = plt.subplots(1, 1, figsize=(20, 8))

    # Define a beautiful colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    hm = sns.heatmap(
        wilcoxon_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
        vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=axes #cbar_kws={'shrink': 0.75}, 
    )

    for text in hm.texts:
        # Read the text value cast it to float
        val = float(text.get_text())
        if val > 0.05:
            text.set_weight('bold')
            text.set_color('black')
            text.set_size(12)      # Make it slightly larger to stand out
    # ---------------------------------------

    axes.set_title(f"Wilcoxon Signed Rank Test ({simlarity_measure})")

    cbar = fig.colorbar(hm.collections[0], ax=axes, shrink=0.75, location='right')

    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, f"wilcoxon_{simlarity_measure}.tif"), format="tiff", dpi=600)

    plt.show()


#%%
# Deep dive in LASSO feature selection
stability_df = pd.read_csv(os.path.join(OUT_DIR, 'stability.csv'))
lasso_df = {"coef":[], "f1_count":[], "f2_count":[], "overlap_count":[], "estimate":[], "perturb_idx":[]}

fs_method = "embedded/LASSO"

for i in range(NUM_DATA_PERTURBATIONS):
        for j in range(i+1, NUM_DATA_PERTURBATIONS):
            
            dict1 = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
            dict2 = np.load(os.path.join(DATA_DIR, fs_method, f"{j+1}.npz"), allow_pickle=True)
            
            map_idx = (i * NUM_DATA_PERTURBATIONS) + j # to map the tuple (i,j) to a unique integer
            
            df1 = pd.DataFrame(dict1["rank_dict"].item())
            df2 = pd.DataFrame(dict2["rank_dict"].item())

            for coef_filter in ["all", "non-zero", "zero"]:

                df1_temp = df1[df1.absolute_coef != 0].copy() if coef_filter == "non-zero" else (df1[df1.absolute_coef == 0].copy() if coef_filter == "zero" else df1.copy())
                df2_temp = df2[df2.absolute_coef != 0].copy() if coef_filter == "non-zero" else (df2[df2.absolute_coef == 0].copy() if coef_filter == "zero" else df2.copy())

                estimate = similarity_index.global_spearman(df1_temp, df2_temp, ignore_cardinality=True)
                
                lasso_df["coef"].append(coef_filter)
                lasso_df["f1_count"].append(len(df1_temp))
                lasso_df["f2_count"].append(len(df2_temp))
                lasso_df["overlap_count"].append(len(df1_temp[df1_temp.feature.isin(df2_temp.feature)]))
                lasso_df["estimate"].append(estimate)
                lasso_df["perturb_idx"].append(map_idx)

 
display(pd.DataFrame(lasso_df).groupby(by=["coef"]).mean())

lasso_df_summary = pd.DataFrame(lasso_df)
lasso_df_agg = lasso_df_summary.groupby("coef")[["f1_count", "f2_count", "overlap_count"]].mean()

coef_order = ["all", "non-zero", "zero"]

lasso_df_plot = pd.DataFrame(lasso_df)
coef_label_map = {"all": "All", "non-zero": "Non-Zero", "zero": "Zero"}
lasso_df_plot["coef_label"] = lasso_df_plot["coef"].map(coef_label_map)
coef_label_order = ["All", "Non-Zero", "Zero"]

fig, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: Grouped bar chart ---
width = 0.3
offsets = [-width / 2, width / 2]
group_vals_combined = {
    "All Features":    [(lasso_df_agg.loc[c, "f1_count"] + lasso_df_agg.loc[c, "f2_count"]) / 2 for c in coef_order],
    "Overlapped Features": [lasso_df_agg.loc[c, "overlap_count"] for c in coef_order],
}
x = np.arange(len(coef_order))
bar_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for (offset, (label, vals)), color in zip(zip(offsets, group_vals_combined.items()), bar_colors):
    bars = ax_bar.bar(x + offset, vals, width=width, label=label,
                      color=color, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.0f}",
            ha="center", va="bottom", fontsize=8, color="black"
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(["All", "Non-Zero", "Zero"])
ax_bar.set_xlabel("LASSO Coefficients", fontweight="bold", labelpad=10)
ax_bar.set_ylabel("Mean Count\n(across perturbation pairs)", fontweight="bold", labelpad=10)
ax_bar.legend(frameon=False)
sns.despine(ax=ax_bar)

# --- Right: Stability estimate distribution per coef filter ---
sns.violinplot(
    data=lasso_df_plot, x="coef_label", y="estimate",
    order=coef_label_order, inner=None, cut=0,
    linewidth=1, density_norm="width", alpha=0.5, ax=ax_dist
)
sns.stripplot(
    data=lasso_df_plot, x="coef_label", y="estimate",
    order=coef_label_order, color="black", alpha=0.15,
    size=2.5, jitter=0.12, ax=ax_dist, zorder=2
)
# Mean marker: diamond per group, drawn on top
means = lasso_df_plot.groupby("coef_label")["estimate"].mean()
for i, coef_label in enumerate(coef_label_order):
    ax_dist.scatter(i, means[coef_label], marker="D", color="white",
                    edgecolors="black", linewidths=1.5,
                    s=20, zorder=6, label="Mean" if i == 0 else None)
ax_dist.legend(frameon=False, fontsize=8)


ax_dist.set_xlabel("LASSO Coefficients", fontweight="bold", labelpad=10)
ax_dist.set_ylabel("Spearman Correlation (global)", fontweight="bold", labelpad=10)
sns.despine(ax=ax_dist)

# plt.suptitle("LASSO: Feature Selection Output & Stability by Coefficient Filter",
#              fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lasso_deepdive.tif"), format="tiff", dpi=300, bbox_inches="tight")
plt.show()


# #%%
# # Pie charts: feature overlap breakdown for each coef_filter
# lasso_df_summary = pd.DataFrame(lasso_df)
# lasso_df_agg = lasso_df_summary.groupby("coef")[["f1_count", "f2_count", "overlap_count"]].mean()

# fig, ax = plt.subplots(figsize=(7, 5))

# coef_order = ["all", "non-zero", "zero"]
# x_labels = ["All", "Non-zero", "Zero"]
# gray_palette = {"Only in f₁": "#2b2b2b", "Overlap": "#888888", "Only in f₂": "#d0d0d0"}

# only_f1_vals, overlap_vals, only_f2_vals = [], [], []

# for coef_filter in coef_order:
#     row = lasso_df_agg.loc[coef_filter]
#     overlap = row["overlap_count"]
#     only_f1_vals.append(row["f1_count"] - overlap)
#     overlap_vals.append(overlap)
#     only_f2_vals.append(row["f2_count"] - overlap)

# x = range(len(coef_order))

# bars1 = ax.bar(x, only_f1_vals, label="Only in f₁", color=gray_palette["Only in f₁"], edgecolor="white", linewidth=0.8)
# bars2 = ax.bar(x, overlap_vals,  label="Overlap",    color=gray_palette["Overlap"],    edgecolor="white", linewidth=0.8, bottom=only_f1_vals)
# bars3 = ax.bar(x, only_f2_vals,  label="Only in f₂", color=gray_palette["Only in f₂"], edgecolor="white", linewidth=0.8,
#                bottom=[a + b for a, b in zip(only_f1_vals, overlap_vals)])

# # Annotate each segment with its value
# for bars, bottoms in [(bars1, [0]*3), (bars2, only_f1_vals), (bars3, [a+b for a,b in zip(only_f1_vals, overlap_vals)])]:
#     for bar, bot in zip(bars, bottoms):
#         h = bar.get_height()
#         if h > 1:  # only label if segment is big enough to read
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bot + h / 2,
#                 f"{h:.0f}",
#                 ha="center", va="center",
#                 fontsize=9, color="white", fontweight="bold"
#             )

# ax.set_xticks(list(x))
# ax.set_xticklabels(x_labels)
# ax.set_xlabel("Coefficient Filter", fontweight="bold", labelpad=10)
# ax.set_ylabel("Mean Feature Count\n(across perturbation pairs)", fontweight="bold", labelpad=10)
# ax.legend(loc="upper right", frameon=False)
# sns.despine(ax=ax)

# plt.suptitle("LASSO Feature Overlap by Coefficient Filter", fontweight="bold", y=1.01)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "lasso_overlap_bar.tif"), format="tiff", dpi=300, bbox_inches="tight")
# plt.show()

# #%%
# # Grouped bar chart: mean feature count vs overlap per coef filter
# fig, ax = plt.subplots(figsize=(7, 5))

# width = 0.3
# offsets = [-width / 2, width / 2]  # two bars per group

# group_vals = {
#     "All Features":  [(lasso_df_agg.loc[c, "f1_count"] + lasso_df_agg.loc[c, "f2_count"]) / 2 for c in coef_order],
#     "Common Features": [lasso_df_agg.loc[c, "overlap_count"] for c in coef_order],
# }

# x = np.arange(len(coef_order))

# for offset, (label, vals) in zip(offsets, group_vals.items()):
#     bars = ax.bar(x + offset, vals, width=width, label=label,
#                   edgecolor="white", linewidth=0.8)
#     for bar, v in zip(bars, vals):
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 0.5,
#             f"{v:.0f}",
#             ha="center", va="bottom",
#             fontsize=8, color="black"
#         )

# ax.set_xticks(x)
# ax.set_xticklabels(["All", "Non-zero", "Zero"])
# ax.set_xlabel("Coefficient Filter", fontweight="bold", labelpad=10)
# ax.set_ylabel("Mean Count\n(across perturbation pairs)", fontweight="bold", labelpad=10)
# ax.legend(frameon=False)
# sns.despine(ax=ax)

# plt.suptitle("LASSO Selection Output Stratified by Coefficient Filter", fontweight="bold", y=1.01)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "lasso_overlap_grouped_bar.tif"), format="tiff", dpi=300, bbox_inches="tight")
# plt.show()

# Combined: grouped bar chart (left) + stability estimate distribution (right)

#%% Deep dive into MIM

mim_df = {"coef":[], "f1_count":[], "f2_count":[], "overlap_count":[], "estimate":[], "perturb_idx":[]}

fs_method = "filter/mutual_info_classif"

for i in range(NUM_DATA_PERTURBATIONS):
        for j in range(i+1, NUM_DATA_PERTURBATIONS):
            
            dict1 = np.load(os.path.join(DATA_DIR, fs_method, f"{i+1}.npz"), allow_pickle=True)
            dict2 = np.load(os.path.join(DATA_DIR, fs_method, f"{j+1}.npz"), allow_pickle=True)
            
            map_idx = (i * NUM_DATA_PERTURBATIONS) + j # to map the tuple (i,j) to a unique integer
            
            df1 = pd.DataFrame(dict1["rank_dict"].item())
            df2 = pd.DataFrame(dict2["rank_dict"].item())


            for coef_filter in ["all", "non-zero", "zero"]:

                df1_temp = df1[df1.score != 0].copy() if coef_filter == "non-zero" else (df1[df1.score == 0].copy() if coef_filter == "zero" else df1.copy())
                df2_temp = df2[df2.score != 0].copy() if coef_filter == "non-zero" else (df2[df2.score == 0].copy() if coef_filter == "zero" else df2.copy())

                estimate = similarity_index.global_spearman(df1_temp, df2_temp, ignore_cardinality=True)
                
                mim_df["coef"].append(coef_filter)
                mim_df["f1_count"].append(len(df1_temp))
                mim_df["f2_count"].append(len(df2_temp))
                mim_df["overlap_count"].append(len(df1_temp[df1_temp.feature.isin(df2_temp.feature)]))
                mim_df["estimate"].append(estimate)
                mim_df["perturb_idx"].append(map_idx)

 
display(pd.DataFrame(mim_df).groupby(by=["coef"]).mean())

mim_df_summary = pd.DataFrame(mim_df)
mim_df_agg = mim_df_summary.groupby("coef")[["f1_count", "f2_count", "overlap_count"]].mean()

coef_order = ["all", "non-zero", "zero"]

mim_df_plot = pd.DataFrame(mim_df)
coef_label_map = {"all": "All", "non-zero": "Non-Zero", "zero": "Zero"}
mim_df_plot["coef_label"] = mim_df_plot["coef"].map(coef_label_map)
coef_label_order = ["All", "Non-Zero", "Zero"]

fig, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: Grouped bar chart ---
width = 0.3
offsets = [-width / 2, width / 2]
group_vals_combined = {
    "All Features":    [(mim_df_agg.loc[c, "f1_count"] + mim_df_agg.loc[c, "f2_count"]) / 2 for c in coef_order],
    "Overlapped Features": [mim_df_agg.loc[c, "overlap_count"] for c in coef_order],
}
x = np.arange(len(coef_order))
bar_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for (offset, (label, vals)), color in zip(zip(offsets, group_vals_combined.items()), bar_colors):
    bars = ax_bar.bar(x + offset, vals, width=width, label=label,
                      color=color, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.0f}",
            ha="center", va="bottom", fontsize=8, color="black"
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(["All", "Non-Zero", "Zero"])
ax_bar.set_xlabel("Mutual Information", fontweight="bold", labelpad=10)
ax_bar.set_ylabel("Mean Count\n(across perturbation pairs)", fontweight="bold", labelpad=10)
ax_bar.legend(frameon=False)
sns.despine(ax=ax_bar)

# --- Right: Stability estimate distribution per coef filter ---
sns.violinplot(
    data=mim_df_plot, x="coef_label", y="estimate",
    order=coef_label_order, inner=None, cut=0,
    linewidth=1, density_norm="width", alpha=0.5, ax=ax_dist
)
sns.stripplot(
    data=mim_df_plot, x="coef_label", y="estimate",
    order=coef_label_order, color="black", alpha=0.15,
    size=2.5, jitter=0.12, ax=ax_dist, zorder=2
)
# Mean marker: diamond per group, drawn on top
means = mim_df_plot.groupby("coef_label")["estimate"].mean()
for i, coef_label in enumerate(coef_label_order):
    ax_dist.scatter(i, means[coef_label], marker="D", color="white",
                    edgecolors="black", linewidths=1.5,
                    s=20, zorder=6, label="Mean" if i == 0 else None)
ax_dist.legend(frameon=False, fontsize=8)


ax_dist.set_xlabel("Mutual Information", fontweight="bold", labelpad=10)
ax_dist.set_ylabel("Spearman Correlation (global)", fontweight="bold", labelpad=10)
sns.despine(ax=ax_dist)

# plt.suptitle("MIM: Feature Selection Output & Stability by Coefficient Filter",
#              fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mim_deepdive.tif"), format="tiff", dpi=300, bbox_inches="tight")
plt.show()

#%%
# Figure 1 visualizing mean stability characteristics
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
# Figure 1 detailed stability distribution plots
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


from MLstatkit import Delong_test
import itertools


def compare_performance(fs_methods, results_df):
    
    df = pd.DataFrame(np.zeros((len(fs_methods), len(fs_methods))), index=fs_methods, columns=fs_methods)
    
    pairs = list(itertools.product(fs_methods, repeat=2))
    
    for fs_method1, fs_method2 in pairs:

        predictions1 = results_df[(results_df.fs_method == fs_method1)].sort_values(by="perturb_idx").reset_index(drop=True).predictions.to_list()
        predictions2 = results_df[(results_df.fs_method == fs_method2)].sort_values(by="perturb_idx").reset_index(drop=True).predictions.to_list()
        targets = results_df[(results_df.fs_method == fs_method1)].sort_values(by="perturb_idx").reset_index(drop=True).targets.to_list()

        differences = np.array(predictions1) - np.array(predictions2)
        if np.all(differences == 0):
            p = 1
        else:
            _, p = Delong_test(targets, predictions1, predictions2, return_ci=False, return_auc=False, verbose=0)

        
        df.loc[fs_method1, fs_method2] = p

    return df

# Test the performance of the feature selection methods
delonge_df = compare_performance(FS_METHODS, results_df)
delonge_df.to_csv(os.path.join(OUT_DIR, 'delong_test.csv'), index=False)

delonge_df = delonge_df.rename(index=label_mapping, columns=label_mapping)

fig, axes = plt.subplots(1, 1, figsize=(20, 8))

# Define a beautiful colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

hm = sns.heatmap(
    delonge_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
    vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=axes #cbar_kws={'shrink': 0.75}, 
)

for text in hm.texts:
    # Read the text value cast it to float
    val = float(text.get_text())
    if val > 0.05:
        text.set_weight('bold')
        text.set_color('black')
        text.set_size(12)      # Make it slightly larger to stand out
# ---------------------------------------

axes.set_title(f"Delong")

cbar = fig.colorbar(hm.collections[0], ax=axes, shrink=0.75, location='right')

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, f"delong_test.tif"), format="tiff", dpi=600)


plt.show()


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

#%%
# Identifying the top-5 frequent features

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
# comparing frequent and selected features

fs_families = {
    "filter/mannwhitneyu": ["filter/mannwhitneyu"],
    "filter/mrmr_classif": ["filter/mrmr_classif"],
    "filter/mutual_info_classif": ["filter/mutual_info_classif"],
    "embedded": ["embedded/LASSO"],
    "wrapper/LogisticRegression": ["wrapper/LogisticRegression"],
    "wrapper/SVC": ["wrapper/SVC"],
    "wrapper/RandomForestClassifier": ["wrapper/RandomForestClassifier"],
    "wrapper/MLPClassifier": ["wrapper/MLPClassifier"],
    "filter/singleAE": ["filter/singleAE"],
    "filter/bayesianAE": ["filter/bayesianAE"],
    "filter/ensembleAE": ["filter/ensembleAE"],
    "filter": ["filter/mannwhitneyu", "filter/mrmr_classif", "filter/mutual_info_classif"],
    "filter_wlcx_vs_mrmr":["filter/mannwhitneyu", "filter/mrmr_classif"],
    "wrapper": ["wrapper/LogisticRegression", "wrapper/SVC", "wrapper/RandomForestClassifier", "wrapper/MLPClassifier"],
    "wrapper_linear":["wrapper/LogisticRegression", "wrapper/SVC"],
    "wrapper_non_linear":["wrapper/RandomForestClassifier", "wrapper/MLPClassifier"],
    "ae": ["filter/singleAE", "filter/bayesianAE", "filter/ensembleAE"],
    "ae_single_vs_bayesian": ["filter/singleAE", "filter/bayesianAE"],
    "ae_bayesian_vs_ensemble": ["filter/bayesianAE", "filter/ensembleAE"],
    "ae_single_vs_ensemble": ["filter/singleAE", "filter/ensembleAE"],
    "lasso_vs_lsvm": ["embedded/LASSO", "wrapper/SVC"]
}

for family, fs_methods in fs_families.items():
    frequent_feats = []
    for fs_method in fs_methods:
        frequent_feats.append(set(frequent_feats_df[frequent_feats_df.fs_method==fs_method].frequent_feats.iloc[0]))

    overlap_freq_feats = set.intersection(*frequent_feats)
    frequent_feats = list(set.union(*frequent_feats))
    
    selected_feats = []
    for fs_method in fs_methods:
        selected_feats.append(set(selected_feats_df[selected_feats_df.fs_method==fs_method].selected_feats.iloc[0]))

    overlap_selected_feats = set.intersection(*selected_feats)
    selected_feats = list(set.union(*selected_feats))
    
    common_feats = overlap_freq_feats.intersection(overlap_selected_feats)
    print("*"*5)
    print(f"Family: {family}")
    print(f"Selected Features: {len(overlap_selected_feats)}")
    print(f"Frequent Features: {len(overlap_freq_feats)}")
    print(f"Common Features: {len(common_feats)}")

#%%
# Correlation Analysis

import itertools
from scipy.optimize import linear_sum_assignment

def mwm(signature_dict, feats_df, corr_method='spearman'):
    
    fs_methods, signatures = zip(*signature_dict.items())
    
    df = pd.DataFrame(np.zeros((len(fs_methods), len(fs_methods))), index=fs_methods, columns=fs_methods)
    
    fs_methods = list(fs_methods)
    signatures = list(signatures)

    f = list(set(sum(signatures, [])))
    corr_matrix = feats_df[f].corr(method=corr_method).abs()
    
    pairs = list(itertools.product(fs_methods, repeat=2))
    
    for fs_method1, fs_method2 in pairs:
        
        
        f1_k, f2_k = signature_dict[fs_method1], signature_dict[fs_method2]
        cost_matrix = corr_matrix.loc[f1_k, f2_k]
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        
        cost = cost_matrix.values[row_ind, col_ind].sum()/len(f1_k)
        
        df.loc[fs_method1, fs_method2] = cost

    return df


for df_type, df in {"selected_feats":selected_feats_df, "frequent_feats":frequent_feats_df}.items():
    
    corr_dfs = {}
    for corr_method in ["pearson", "spearman"]:
        fs_methods = df.fs_method.to_list()
        top_feats = df[df_type].to_list()
        signature_dict = dict(zip(fs_methods, top_feats))

        corr_df = mwm(signature_dict, RADIOMICS_DF, corr_method=corr_method).abs()
        corr_dfs[corr_method] = corr_df
        

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

    pearson_corr_df = corr_dfs["pearson"].rename(index=label_mapping, columns=label_mapping)
    spearman_corr_df = corr_dfs["spearman"].rename(index=label_mapping, columns=label_mapping)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Define a beautiful colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax1 = axes[0]
    hm1 = sns.heatmap(
        pearson_corr_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
        vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=ax1 #cbar_kws={'shrink': 0.75}, 
    )
    ax1.set_title("Pearson Correlation")

    # --- Right: Spearman ---
    ax2 = axes[1]
    hm2 = sns.heatmap(
        spearman_corr_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
        vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=ax2
    )

    cbar = fig.colorbar(hm2.collections[0], ax=ax2, shrink=0.75, location='right')

    ax2.set_title("Spearman Correlation")

    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, f"{df_type}_corr_plot.tif"), format="tiff", dpi=600)

    plt.show()

#%%

_selected_feats_df = selected_feats_df.copy()
_selected_feats_df = _selected_feats_df.rename(columns={"selected_feats": "features"})
_selected_feats_df["fs_method"] = ["top-5_"+fs_method for fs_method in _selected_feats_df["fs_method"]]

_frequent_feats_df = frequent_feats_df.copy()
_frequent_feats_df = _frequent_feats_df.rename(columns={"frequent_feats": "features"})
_frequent_feats_df["fs_method"] = ["frequent_top-5_"+fs_method for fs_method in _frequent_feats_df["fs_method"]]
combined_df = pd.concat([_selected_feats_df, _frequent_feats_df], axis=0)

corr_dfs = {}
for corr_method in ["pearson", "spearman"]:
    fs_methods = combined_df.fs_method.to_list()
    top_feats = combined_df.features.to_list()
    signature_dict = dict(zip(fs_methods, top_feats))

    corr_df = mwm(signature_dict, RADIOMICS_DF, corr_method=corr_method).abs()
    corr_dfs[corr_method] = corr_df
        

    label_mapping = {
        "top-5_filter/mannwhitneyu":"top-5 WLCX",
        "top-5_filter/mrmr_classif":"top-5 MRMR",
        "top-5_filter/mutual_info_classif":"top-5 MIM",
        "top-5_embedded/LASSO":"top-5 LASSO",
        "top-5_wrapper/LogisticRegression": "top-5 SBS+LR",
        "top-5_wrapper/SVC": "top-5 SBS+L-SVM",
        "top-5_wrapper/RandomForestClassifier": "top-5 SBS+RF",
        "top-5_wrapper/MLPClassifier": "top-5 SBS+MLP",
        "top-5_filter/singleAE":"top-5 singleAE", 
        "top-5_filter/bayesianAE": "top-5 bayesianAE", 
        "top-5_filter/ensembleAE": "top-5 ensembleAE",
        "frequent_top-5_filter/mannwhitneyu":"frequent top-5 WLCX",
        "frequent_top-5_filter/mrmr_classif":"frequent top-5 MRMR",
        "frequent_top-5_filter/mutual_info_classif":"frequent top-5 MIM",
        "frequent_top-5_embedded/LASSO":"frequent top-5 LASSO",
        "frequent_top-5_wrapper/LogisticRegression": "frequent top-5 SBS+LR",
        "frequent_top-5_wrapper/SVC": "frequent top-5 SBS+L-SVM",
        "frequent_top-5_wrapper/RandomForestClassifier": "frequent top-5 SBS+RF",
        "frequent_top-5_wrapper/MLPClassifier": "frequent top-5 SBS+MLP",
        "frequent_top-5_filter/singleAE":"frequent top-5 singleAE", 
        "frequent_top-5_filter/bayesianAE": "frequent top-5 bayesianAE", 
        "frequent_top-5_filter/ensembleAE": "frequent top-5 ensembleAE"

    }

pearson_corr_df = corr_dfs["pearson"].rename(index=label_mapping, columns=label_mapping)
spearman_corr_df = corr_dfs["spearman"].rename(index=label_mapping, columns=label_mapping)

fig, axes = plt.subplots(1, 2, figsize=(40, 16))

# Define a beautiful colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

ax1 = axes[0]
hm1 = sns.heatmap(
    pearson_corr_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
    vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=ax1 #cbar_kws={'shrink': 0.75}, 
)
ax1.set_title("Pearson Correlation")

# --- Right: Spearman ---
ax2 = axes[1]
hm2 = sns.heatmap(
    spearman_corr_df, cmap=cmap, square=True, linewidth=0.5, cbar=False, 
    vmin=0, vmax=1, annot = True, fmt=".2f", annot_kws={"size":10}, ax=ax2
)

cbar = fig.colorbar(hm2.collections[0], ax=ax2, shrink=0.75, location='right')

ax2.set_title("Spearman Correlation")

plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, f"combined_top-5_vs_frequent_corr_plot.tif"), format="tiff", dpi=600)

plt.show()
        


        



        
        

