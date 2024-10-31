# Official code implementation of MUSE.

## [Notification] Modified code will be uploaded soon!

## Paper information
- Title: You May Better Reconstruct Anomalous over Normal Graphs: Analysis and a Simple Method for Reconstruction-based Graph-Level Anomaly Detection
- Status: Submitted to NeurIPS 2024
- Summary: We detect graph-level anomalies with error distributions.

## TL;DR

- We provide the code implementation of the proposed graph-level anomaly detection method, **MUSE**.
- One can reproduce the MUSE's results of the main anomaly detection experiment with the below command
  ```
  python3 main.py -data dd -anom_type 0 -lr 0.001 -dim 256 -n_layers 4 -device cuda:0 
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
  - **-device** [String]: Set the GPU (or CPU) device.
    - (E.g., if using the 0-th GPU, one can give ``-device cuda:0``)
- Example ``main.py`` of the above example settings are
  ```
  python3 main.py -data dd -anom_type 0 -lr 0.001 -dim 256 -n_layers 4 -device cuda:0 
  ```

## Hyperparameter configurations
- We provide the MUSE's best validation-split-based hyperparameter configuration of each dataset in ``MUSE_hyperparameters.pickle``.
