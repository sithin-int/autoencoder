# Exploring Self-supervised Deep Sparse Autoencoders for Robust Feature Selection in Radiomics Analysis

Repository supporting the article submitted to Scientific Reports

### **Citation**
[Exploring self-supervised deep sparse autoencoders for robust feature selection in radiomics analysis](https://doi.org/10.1038/s41598-026-62420-7). *Sci Rep (2026)*

>Thulasi Seetha, S., Messina, A., Casale, A. et al. Exploring self-supervised deep sparse autoencoders for robust feature selection in radiomics analysis. Sci Rep (2026). https://doi.org/10.1038/s41598-026-62420-7
<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21292936.svg)](https://doi.org/10.5281/zenodo.21292936)-->
### **Repository structure**
#### **Overview:**

* Feature Selection Using Classical Methods (Filter, Embedded, Wrapper) vs. AE Variants
* AE Pytorch Utilities
* Feature Selection Stability Analysis
* Computational Cost Analysis 
* Visualizations
* Sanity Checks
* Python Scripts

#### **Contents:**
```
autoencoder
├── inputs/                                  
│    └── radiomicsFeaturesWithLabels.csv    #----- radiomics features extracted per patient, per sequence + labels
├── src/                                     
│    └── generate_perturbations.py          # train-test partitions associated with perturbation runs (# runs = # splits = 100) 
│    └── random_fs.py                       #----- reference random feature selection (FS) baseline: randomly assigns ranks to features       
│    └── autoencoder_fs/                    # directory containing scripts associated with AE Variants
│        └── fs_singleAE.py  
│        └── fs_bayesianAE.py
│        └── fs_ensembleAE.py  
│    └── classical_fs/                      # directory containing scripts associated with classical FS methods: Filter, wrapper, embedded
│        └── fs_filtering.py                #----- Filter methods include: Wilcoxon Test, Mutual Information, mRMR
│        └── fs_wrapper.py                  #----- Backward Elimination + {LR, L-SVM, RF, MLP}
│        └── fs_embedded.py                 #----- LASSO
│    └── utils/                             # directory containing utilities
│        └── nn_utils.py                    #----- Pytorch utilities containing class definitions for singleAE, and bayesianAE architectures.
│        └── similarity_index.py            #----- definitions of feature selection stability indices: jaccard, dice, kuncheva, mwm, global
│    └── analysis.py                        # stability analysis, computational cost analysis, and visualization
├── README.md

```

