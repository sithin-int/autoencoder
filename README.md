# Exploring Self-supervised Deep Sparse Autoencoders for Robust Feature Selection in Radiomics Analysis
Repository supporting the article submitted to Scientific Reports

[![DOI](https://zenodo.org/badge/784240373.svg)](https://doi.org/10.5281/zenodo.21292935)

If you use this codebase for your research, please cite our paper if available; otherwise, please cite this repository:
```bibtex
TBA
```
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

