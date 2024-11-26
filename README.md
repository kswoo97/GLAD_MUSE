# Official code implementation of MUSE.

## Paper information
- ***Title***: Rethinking Reconstruction-based Graph-Level Anomaly Detection: Limitations and a Simple Remedy
- ***Venue***: Accepted to **NeurIPS 2024**
- ***Authors***: Sunwoo Kim, Soo Yong Lee, Fanchen Bu, Shinhwan Kang, Kyungho Lee, Jaemin Yoo, and Kijung Shin
- ***Affiliation***: KAIST AI and KAIST EE
- ***Paper link***: https://arxiv.org/abs/2410.20366


## TL;DR

- We provide the code implementation of the proposed graph-level anomaly detection method, **MUSE**.
- One can reproduce the MUSE's results of the main anomaly detection experiment with the below command
  ```
  python3 main.py -data dd -anom_type 0 -lr 0.001 -dim 256 -n_layers 4 -gamma 2.0 -device cuda:0 
  ```

## Datasets

We support 10 graph benchmark datasets, whose overall statistics are as follows:

| Full name | Command name |\# of graphs | Avg. \# of nodes | Avg. \# of edges |
|------|---|---|---|---|
| DD | dd |1,178 | 284.32 | 715.66 |
| Protein| protein | 1,113 | 39.06 | 72.82 |
| NCI1 | nci1| 4,110 | 29.87 | 32.30 |
| AIDS | aids | 2,000 | 15.69 | 16.20 |
| Reddit-binary | reddit | 2,000 | 429.63 | 497.75 |
| IMDB-binary | imdb | 1,000 | 19.77 | 96.53 |
| Mutagenicity | mutag | 4,337 | 30.32 | 30.77 |
| DHFR | dhfr | 756 | 42.43 | 44.54 |
| BZR | bzr | 405 | 35.75 | 38.36 |
| Tox21-ER | er | 7,697 | 17.58 | 17.94 |

## How to run?

- ``main.py`` supports the running the main experiment.
- This file contains four arguments.
  - **-data** [String]: Put the command name of the dataset one wants to utilize.
    - (E.g., if running the DD dataset, one can give ``-data dd``)
  - **-anom_type** [0 or 1]: Set the anomaly graph class. Since all the datasets are binary class datasets, the other class is automatically becoming a normal class.
    - (E.g., if setting class 0 as anomaly class, one can give ``anom_type 0``)
  - **-lr** [Float]: Set the learning rate of MUSE.
    - (E.g., if setting the learning rate as $10^{-3}$, one can give ``-lr 0.001``)
  - **-dim** [Integer]: Set the hidden dimension of MUSE.
    - (E.g., if setting the hidden dimension as $256$, one can give ``-dim 256``)
  - **-n_layers** [Integer]: Set the number of GNN layers of MUSE.
    - (E.g., if setting the number of layers as $5$, one can give ``-n_layers 4``)
  - **-gamma** [Float]: Set the gamma weight regarding the 1-entry of the adjacency matrix.
    - (E.g., if setting the wegith as $1.0$, one can give ``-gamma 1.0``)
  - **-device** [String]: Set the GPU (or CPU) device.
    - (E.g., if using the 0-th GPU, one can give ``-device cuda:0``)
- Example ``main.py`` of the above example settings are
  ```
  python3 main.py -data dd -anom_type 0 -lr 0.001 -dim 256 -n_layers 4 -gamma 1.0 -device cuda:0 
  ```

## Hyperparameter configurations
- We provide the MUSE's best validation-split-based hyperparameter configuration of each dataset in ``MUSE_hyperparameters.pickle``.
