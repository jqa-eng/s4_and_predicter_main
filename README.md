# Triazole Antibacterial Molecule Design

This repository contains the source code and experimental scripts for a triazole antibacterial molecule property prediction and directed molecular design project. The project integrates molecular property prediction, SMILES-based molecular generation, reinforcement learning optimization and molecular filtering.

The current version is mainly released for research reproducibility and paper/code reference. Some scripts were produced during iterative experiments, so the recommended reproduction pipeline is described below.

## 1. Project Overview

The project focuses on the computational design of triazole antibacterial candidate molecules. The main workflow includes:

1. Dataset preprocessing and train/validation/test splitting;
2. Feature enhancement for molecular property prediction;
3. Training of molecular property prediction models;
4. Reinforcement learning optimization of the molecular generation model;
5. Molecular generation with fixed scaffold/prefix constraints;
6. Candidate molecule filtering and QED/SA calculation,.

The property prediction module includes:

- `S.aureus` MIC regression model;
- `E.coli` MIC regression model;
- Cytotoxicity classification model.

The generation module is based on an S4 sequence generation model. Reinforcement learning is used to further optimize the generator toward better antibacterial activity and lower toxicity risk.

## 2. Environment Setup

The dependencies used in this project are listed in `all_requirements.txt`.

A basic installation command is:

```bash
pip install -r all_requirements.txt
```

The project was mainly tested with Python 3.10, PyTorch, RDKit-related molecular processing tools, scikit-learn, XGBoost, pandas, NumPy, and matplotlib.

### Cairo-related Issue

If errors related to Cairo, pycairo, or rendering dependencies occur during installation or molecular visualization, the following conda command may help:

```bash
conda install -c conda-forge cairo pycairo pkg-config meson ninja
```

## 3. Recommended Main Workflow

The repository contains multiple experimental scripts. For reproducibility, the following workflow is recommended.

### 3.1 Dataset Construction

#### 3.1.1 Construct the S.aureus MIC Dataset

```bash
python get_datasets.py \
  --src datasets/standard_datasets/clean_data_v2.csv \
  --out-dir datasets/standard_datasets/aureus_mic_datasets \
  --task aureus \
  --target-col S.aureus \
  --random-split
```

This command generates the train/validation/test datasets for the `S.aureus` MIC prediction task.

Note: This dataset is constructed from the original data only and does not include the newly added experimentally validated samples.

#### 3.1.2 Clean the E.coli Dataset

```bash
python clean_ecoli_data.py
```

This command generates the cleaned `E.coli` antibacterial activity dataset.

#### 3.1.3 Split the E.coli MIC Dataset

```bash
python get_datasets.py \
  --src datasets/standard_datasets/clean_data_Ecoli.csv \
  --out-dir datasets/standard_datasets/ecoli_mic_datasets \
  --task ecoli \
  --target-col Ecoli_MIC_ugmL \
  --smiles-col SMILES \
  --random-split
```

This command splits the cleaned `E.coli` MIC dataset into train/validation/test subsets.

#### 3.1.4 Construct the Cytotoxicity Classification Dataset

```bash
python get_datasets.py \
  --src datasets/standard_datasets/clean_data_v2.csv \
  --out-dir datasets/standard_datasets/toxicity_datasets \
  --task toxicity \
  --target-col cytotoxicity \
  --classifier \
  --random-split
```

This command generates the dataset for cytotoxicity classification.

#### 3.1.5 Construct the Merged S.aureus Dataset

```bash
python log_and_split_aureus.py \
  --input datasets/standard_datasets/aureus_merged.csv \
  --output_dir datasets/standard_datasets/aureus_merged_datasets/
```

This command merges newly added `S.aureus` data with the original data, applies logarithmic transformation, and splits the dataset.

## 4. Feature Enhancement

Before training the prediction models, feature enhancement should be applied to the corresponding train/validation/test datasets.

Example for the merged `S.aureus` dataset:

```bash
python enhanced_datasets.py \
  --in datasets/standard_datasets/aureus_merged_datasets/aureus_train.csv \
  --in datasets/standard_datasets/aureus_merged_datasets/aureus_val.csv \
  --in datasets/standard_datasets/aureus_merged_datasets/aureus_test.csv
```

The same feature enhancement process should also be applied to the `E.coli` MIC dataset and the cytotoxicity dataset before model training.

## 5. Property Prediction Model Training

### 5.1 Train the E.coli MIC Regression Model

```bash
python train_ecoli_paperalign.py \
  --train datasets/standard_datasets/ecoli_mic_datasets/ecoli_train.csv \
  --val datasets/standard_datasets/ecoli_mic_datasets/ecoli_val.csv \
  --test datasets/standard_datasets/ecoli_mic_datasets/ecoli_test.csv \
  --output-dir models_predicter/ecoli_regresser_new
```

This script trains the `E.coli` MIC regression model.

### 5.2 Train the Cytotoxicity Classification Model

```bash
python train_classifier.py \
  --train datasets/standard_datasets/toxicity_datasets/toxicity_train.csv \
  --val datasets/standard_datasets/toxicity_datasets/toxicity_val.csv \
  --test datasets/standard_datasets/toxicity_datasets/toxicity_test.csv \
  --task toxicity \
  --feat-start 2 \
  --feat-count 4274 \
  --output-dir models_predicter/toxicity_classifier_4274d/
```

This script trains the cytotoxicity classification model using enhanced molecular features.

### 5.3 Train the S.aureus MIC Regression Model

```bash
python train_aureus_regresser_fixed.py \
  --train datasets/standard_datasets/aureus_merged_datasets/aureus_train.csv \
  --val datasets/standard_datasets/aureus_merged_datasets/aureus_val.csv \
  --test datasets/standard_datasets/aureus_merged_datasets/aureus_test.csv \
  --output-dir models_predicter/aureus_regresser_merged/
```

This script trains the `S.aureus` MIC regression model using the merged dataset.

## 6. Reinforcement Learning Optimization

### 6.1 Main RL Script: Chebyshev Aggregation + KL Regularization

```bash
python run_rl_v3.py \
  --base-model models_s4/025 \
  --output-dir reinforced_s4_v3_both \
  --device cuda \
  --anti-weights pos=0.45,neg=0.55 \
  --len-max 80
```

This is the recommended basic RL optimization script. It uses Chebyshev aggregation for multi-objective antibacterial activity optimization and KL regularization to stabilize policy updates.

Note: This version provides relatively stable RL training, but the cytotoxicity optimization may still require further adjustment depending on the property models used.

### 6.2 Continue RL Training

```bash
python run_rl_v3_continue.py \
  --base-model reinforced_s4_v3_continue/checkpoints/step_001000 \
  --output-dir reinforced_s4_v3_continue_tox \
  --device cuda \
  --len-max 80
```

This script continues RL training from a saved checkpoint. It can be used for further curriculum-style optimization, especially when additional adjustment is needed for cytotoxicity-related objectives.

### 6.3 Linear Weighted RL Baseline

```bash
python run_rl_linear.py \
  --base-model models_s4/025 \
  --output-dir reinforced_s4_linear \
  --device cuda
```

This script provides a linear weighted RL baseline for comparison with the Chebyshev-based RL strategy.

### 6.4 Earlier RL Versions

The repository may also contain earlier RL scripts such as:

```bash
python run_rl_v1.py --device cuda
python run_rl_v2.py --base-model models_s4/025 --device cuda
```

These earlier versions are retained as experimental records. They are not recommended as the main reproduction entry because they are less complete than the v3 implementation.

## 7. Molecular Generation

### 7.1 Generate Molecules with Fixed Prefix and Filtering

```bash
python genmol_v1_fixed.py \
  --model-dir models_s4/025 \
  --n 5000 \
  --batch 1000 \
  --out molecules.csv \
  --required-prefix "O=C(c1ccc(Br)c(Br)c1)c2n[nH]nc2N(Cc3ccccc3)C"
```

This script generates molecules under a required SMILES prefix constraint and applies built-in filtering.

### 7.2 Generate Raw Molecules without Filtering

```bash
python genmol_raw.py \
  --model-dir reinforced_s4_linear \
  --temp 1.1 \
  --required-prefix-smiles c1n[nH]nc1 \
  --output-csv molecules_raw_linear_RL.csv
```

This script directly generates molecules without filtering. It is useful for evaluating the raw generation distribution of a model.

### 7.3 Search/Exploration Generation Script

```bash
python genmol_search.py \
  --model-dir reinforced_s4_v3_continue_tox
```

This script is retained for search/exploration-style molecule generation. It is not necessarily the primary reproduction script for the final workflow.

## 8. Molecular Property Post-processing

### 8.1 Calculate QED and SA

```bash
python get_qed_sa.py \
  --csv molecules/molecules_linear_RL_temp1.1/molecules_raw_linear_RL_temp1.1.csv
```

This script calculates QED and synthetic accessibility (SA) values for molecules in a CSV file and appends the results as additional columns.

### 8.2 Filter Candidate Molecules

```bash
python mol_filter.py \
  --input molecules/molecules_v3_continue_tox_10/mol_processed.csv \
  --output molecules/molecules_v3_continue_tox_10/mol_filtered.csv
```

This script filters molecules according to the candidate selection rule. In the current version, the filtering criterion is based on whether `aureus_MIC` is less than 20.

## 9. Visualization

### 9.1 3D Scatter Plot

```bash
python cubic_graph.py \
  --mol-processed molecules/molecules_v3_continue_tox_10/mol_processed.csv \
  --mol-filtered molecules/molecules_v3_continue_tox_10/mol_filtered.csv
```

This script draws a 3D scatter plot and highlights molecules that satisfy the candidate filtering criterion.

### 9.2 UMAP Embedding Plot

```bash
python embedded_graph.py \
  --mol-processed molecules/molecules_v3_continue_tox_10/mol_processed.csv \
  --mol-filtered molecules/molecules_v3_continue_tox_10/mol_filtered.csv
```

This script draws a UMAP-based molecular embedding plot and marks the top candidate molecules, such as the top 100 molecules with the lowest `aureus_MIC` values.

## 10. Suggested Repository Structure

The repository may include the following main directories:

```text
.
├── datasets/
│   └── standard_datasets/          # Cleaned datasets and train/validation/test splits
├── models_s4/                      # Pretrained S4 generation models
├── reinforced_s4/                  # RL-optimized S4 models and checkpoints
├── models_predicter/               # Trained property prediction models
│   ├── aureus_regresser/           # Original S.aureus MIC regression model
│   ├── aureus_regresser_merged/    # S.aureus MIC regression model trained with merged data
│   ├── ecoli_regresser/            # E.coli MIC regression model
│   └── toxicity_classifier_4274d/  # Cytotoxicity classification model
├── molecules/                      # Generated molecules and post-processed molecule files
├── validation/                     # Run mol_prediction.py to predict SMILES in mol_topred.csv                                       and output mol_processed.csv
├── scripts and training files      # Main scripts for preprocessing, training, RL, generation, and visualization
├── all_requirements.txt            # Python dependencies
└── README.md                       # Project documentation
```

Large trained model checkpoints, generated molecule files, and intermediate datasets may be excluded from GitHub if they are too large. In that case, provide download links or instructions for regenerating them.

## 11. Notes

- This repository contains both final workflow scripts and experimental scripts.
- For paper reproduction, please follow the recommended workflow in this README.
- Early RL scripts such as `run_rl_v1.py` and `run_rl_v2.py` are kept for development traceability but are not the recommended entry points.
- The v3 RL script is the main implementation for stable reinforcement learning optimization.
- Some generated results depend on the trained S4 base model, property prediction models, and random sampling settings.
- The prediction models are computational evaluation tools and should not be regarded as replacements for experimental biological validation.

## 12. Citation

If this repository is useful for your research, please cite the corresponding paper after publication.

```bibtex
@article{triazole_design_placeholder,
  title   = {Triazole Antibacterial Molecule Property Prediction and Directed Design},
  author  = {Author Names},
  journal = {Journal Name},
  year    = {Year}
}
```

