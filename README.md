# Overview

Provides a unified front-end interface for extracting local and global features for the hierarchical localization pipeline using the kapture format.

Each front-end (local or global) requires its own conda environments.

# Setup

General note: run `pip install -e .` in the main repo directory to setup paths AFTER setting up the relevant conda environment.

## Local features

# 1. R2D2

Conda environment: TBA, some messiness with kapture and R2D2

## Global features

# 1. NetVLAD

Setup the conda environment using the following commands:

```
conda create -n netvlad tensorflow-gpu==1.12.0 cudatoolkit==9.0
conda activate netvlad
pip install opencv-python scipy matplotlib kapture
```

# Note

Do not use the `extract_[local|global]_features.py` script in the main repo folder. A single call script will be useful later, but not neccessary now.
