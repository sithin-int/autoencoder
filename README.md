# Exploring Self-supervised Deep Sparse Autoencoders for Robust Feature Selection in Radiomics Analysis
Repository supporting the article submitted to Scientific Reports

If you use this codebase for your research, please cite our paper if available; otherwise, please cite this repository:
```bibtex
TBA
```
### **Repository structure**
#### **Overview:**

* Feature Selection Using SBS vs. AE Variants
* AE Pytorch Utilities
* Feature Selection Stability Analysis
* Time and Space Complexity Analysis 
* Visualizations
* Sanity Checks
* Jupyter Notebooks and Python Scripts

#### **Contents:**
```
autoencoder
├── radiomicsFeatures                       # radiomics database
│    └── radiomicsFeatures3D.csv            #----radiomics features extracted for the complete analysis per patient, per sequence
│    └── label_df.csv                       #----label information of the patients
│    └── radiomicsFeaturesWithLabels.csv    #----radiomics features + labels
├── scripts                                 # directory containing feature selection Python scripts
│    └── fs_backwardSFS.py                  #----feature selection using SBS variants
│    └── fs_*DSAE.py                        #----feature selection using AE variants (fs_singleDSAE.py, fs_bayesianDSAE.py, fs_ensembleDSAE.py)
├── utils                                   # directory containing utilities
│    └── nn_utils.py                        #----Pytorch utilities containing class definitions for singleAE, and bayesianAE architectures. 
│    └── similarity_index.py                #----definitions of feature selection stability indices: jaccard, dice, kuncheva, mwm, global spearman rank index
├── notebooks                               # Jupyter notebooks (scripts folder contains a subset of Python files exported from the notebooks)
│    └── data_preparation.ipynb             #----notebook for merging raw radiomics features with the label
│    └── fs_random.ipynb                    #----reference random feature selection pipeline
│    └── fs_backwardSFS.ipynb               #----notebook for SBS feature selection pipelines 
│    └── fs_singleDSAE.ipynb                #----notebook for singleAE feature selection pipeline 
│    └── fs_bayesianDSAE.ipynb              #----notebook for bayesianAE feature selection pipeline
│    └── fs_ensembleDSAE.ipynb              #----notebook for ensembleAE feature selection pipeline
│    └── stability_analysis.ipynb           #----notebook for stability analysis, complexity analysis, and visualization
├── README.md

```

