# Overview

Provides a set of useful tools for interacting with the [kapture](https://github.com/naver/kapture.git) package developed by Naver Labs. It provides a *front-end* interface which allows a unified feature extraction pipeline for various visual features for keypoints/local descriptors as well as global features used for a image retrieval step.

Each front-end (local or global) requires its own conda environments for the time being and will need to be activated before running any feature extraction scripts. This will be hard to fix currently due to wildly different versions of dependencies between repos.

# Setup

[NOTE] run `pip install -e .` in the main repo directory to setup paths AFTER setting up the relevant conda environment.

## Datasets

# 1. RobotCar Seasons

Follow the instructions [here](https://github.com/naver/kapture/blob/master/doc/datasets.adoc#robotcar_seasons-v2) and download the original RobotCar Seasons dataset from the cvg website with the *exact format and filestructure* as in the link. Do not download extra files (e.g. pesky desktop.ini files) to avoid hard to debug error messages. 

`kapture_import_RobotCar_Seasons.py` will import the RobotCar directory into the kapture format without copying over the raw data (uses symlinks). Unlike the [original Naver script](https://github.com/naver/kapture/blob/master/tools/kapture_import_RobotCar_Seasons.py), our custom script will import the reference map, training images (poses from a different set of conditions to the mapping run with provided ground truth poses provided in the RobotCar Seasons v2 dataset) and query images (unknown ground truth) separately into different kaptures. The original script combines training and mapping into a single kapture. This allows for quick evaluation of your algorithms on the training set without uploading results to <https://www.visuallocalization.net>.

If the original data is in `/path/to/RobotCar_Seasons/`, I would suggest extracting the data to `/path/to/kapture/RobotCar_Seasons[-v2]` depending on whether or not you choose to use v1 or v2 version of the dataset. This will create the kaptures in a "base" subdirectory which includes the records data (i.e. images) only.

## Local features

For all local features, you can use the `local/kapture_extract_local_all_submaps.py` script to extract local features for all submaps inside of a root directory (i.e. `/path/to/RobotCar_Seasons-v2/` which contains the `xx` submap folders). This is especially useful for RobotCar Seasons and CMU which is split into many submaps, each with their own separate kapture. Simply use the `--feature-name` argument to select the desired feature to use.

Example usage:

```
python3 local/kapture_extract_local_all_submaps.py --kapture-root /path/to/RobotCar_Seasons[-v2]/base --output-dir /path/to/RobotCar_Seasons[-v2]/local_features --feature-name r2d2
```

# 1. R2D2

Setup the conda environment using the following commands:

```
conda create -n r2d2 python=3.6 tqdm pillow numpy matplotlib scipy
conda install conda install pytorch=1.4 torchvision cudatoolkit=10.1 -c pytorch
pip install kapture
pip install -e .
```

## Global features

For all global features, you can use the `global/kapture_extract_global_all_submaps.py` script to extract global features for all submaps inside of a root directory. This is especially useful for RobotCar Seasons and CMU which is split into many submaps, each with their own separate kapture.

Example usage:
```
python3 global/kapture_extract_global_all_submaps.py --kapture-root /path/to/RobotCar_Seasons[-v2]/base --output-dir /path/to/RobotCar_Seasons[-v2]/global_features --feature-name netvlad
```

# 1. NetVLAD

Setup the conda environment using the following commands:

```
conda create -n netvlad python=3.6 tensorflow-gpu==1.12.0 cudatoolkit==9.0
conda activate netvlad
pip install opencv-python scipy matplotlib kapture
pip install -e .
```
