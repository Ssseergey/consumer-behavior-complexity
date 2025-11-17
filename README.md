# Consumer behavior complexity
This repository contains the code for the study: %link%

Abstract: %abstract here%

The goal of this project is to reproduce the results of the referenced study using a full pipeline that includes preprocessing, multiprocess training, and experimental analysis.

---

# Pipeline Overview

The pipeline consists of three main stages:
1. Preprocessing  
2. Multiprocess training  
3. Experiments and analysis  

All steps must be executed sequentially after preparing the dataset and configuring the parameters.

---

# Installation

## Initialization
1. Install pipx: `python3 -m pip install --user pipx; python3 -m pipx ensurepath`. Docs: https://github.com/pypa/pipx
2. Install Poetry: `pipx install poetry` and restart shell. Docs: https://python-poetry.org/docs
3. Run the initialization script from the root directory: `./init.sh`

Use poetry env to execute scripts.

---

# Dataset

The dataset `raif.zip` is a subset of a Kaggle competition dataset by Raiffeisen Bank (publicly available during the competition period).

It contains transaction histories of 10,000 bank clients from April to September 2017.

Each record includes:
- transaction date
- transaction amount
- MCC code (merchant category code)

---

# Reproducing the Study

Follow these steps in order:

1. Run preprocessing notebook:
   preprocessing.ipynb

2. Run training script:
   multiprocess_training.py

3. Run experiments notebook:
   experiments.ipynb

The final results (plots and correlation coefficients) will be available in `experiments.ipynb`.

---

# Using Your Own Data

## 1. Prepare data

Create a folder:
data/<your_folder>/

Place CSV files containing transaction data. Each file must include:
- client_id (convertible to integer)
- MCC code
- transaction amount
- date

Column names can be arbitrary at this stage.

---

## 2. Configure preprocessing

Edit preprocessing.ipynb and update:
column_mapping_based_on_folder_name

Set:
- folder_type
- basic_value

based on your dataset.

---

## 3. Run preprocessing

Run preprocessing.ipynb and wait for completion.

---

## 4. Configure training

Edit multiprocess_training.py:

- device: "cpu" or "cuda"
- fwd: number of forward prediction steps
- split: number of training samples per client
- epochs: number of training epochs
- batch_size: recommended value is 1 for best results
- test_batch_size: depends on hardware
- ignore_existing:
  True  -> skip already trained files  
  False -> train on all files
- processes: number of parallel training processes
- folder_type: dataset folder name (data/<folder_type>/raif_values.csv)

Training is executed sequentially per file, but parallelized across processes.

---

## 5. Run training

python multiprocess_training.py

Wait until training finishes.

---

## 6. Run experiments

Set folder_type in experiments.ipynb and run the notebook.

---

# Output

Final outputs include:
- plots
- correlation coefficients
- analysis results

All results are stored in experiments.ipynb.

