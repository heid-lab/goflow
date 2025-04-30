# GoFlow: Efficient Transition State Geometry Prediction with Flow Matching and E(3)-Equivariant Neural Networks

GoFlow is an open-source model for predicting transition state geometries of single-step organic reactions.
This repository contains the official implementation, including all scripts to fully reproduce the results reported in the paper.

## Installation
Install GoFlow dependencies with Conda (recommended):

```bash
# Create environment
conda env create -f environment.yml
# Activate environment
conda activate goflow
```

## Configuration
We use [Hydra](https://hydra.cc/) for managing model configurations and experiments.

All hyper-parameters are found in the configs directory and its subdirectories (`./configs`).

## Dataset
The dataset used to train and evaluate GoFlow is the open-source [RDB7 database](https://zenodo.org/records/13328872) by [Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6)

The raw `.csv` and `.xyz` files are found in the `data/RDB7/raw_data` directory. The processed data, i.e. each reaction saved as [PyG](https://pytorch-geometric.readthedocs.io/) object in a Python list, is found in the `data/RDB7/processed_data` directory in the `data.pkl` file.

Indices for the random, reaction core, and barrier height dataset splits are found in the `data/RDB7/splits` directory.

The data is preprocessed by running the `preprocessing.sh` shell script. Inside the script, edit the paths to the `csv` and `xyz` files as needed.

## Usage
Each experiment has a separate shell script (.sh files).

E.g. to train and test the model for all dataset splits run `bash train_test_all_splits.sh` in a unix shell.

Modify the shell scripts as required to set custom paths for your input and output directories. Also edit the configuration files as needed.

## Acknowledgement

GoFlow is built upon open-source code provided by [TsDiff](https://github.com/seonghann/tsdiff) and [GotenNet](https://github.com/sarpaykent/GotenNet).

## License
Our model and code are released under MIT License.

## Cite

If you use this code in your research, please cite the following paper:

```bibtex
@article{galustian2025goflow,
  author = {Galustian, Leonard and Mark, Konstantin and Karwounopoulos, Johannes and Kovar, Maximilian and Heid, Esther},
  title = {GoFlow: Efficient Transition State Geometry Prediction with Flow Matching and E(3)-Equivariant Neural Networks},
  year = {2025},
  doi = {10.26434/chemrxiv-2025-bk2rh},
  journal = {ChemRxiv}
}
```

