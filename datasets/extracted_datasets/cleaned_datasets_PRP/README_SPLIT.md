# Aureus Dataset Splits (2025-09-24, Asia/Tokyo)

- Source file: `aureus_datasets.csv`
- Cleaning:
  - Dropped rows with missing/invalid MIC or SMILES
  - Removed exact duplicate SMILES (removed 132 duplicates)
  - Computed `logMIC = log10(MIC)`
  - Labeled `is_active_<25` with threshold 25 μg/mL (log10 ≈ 1.39794)

## Split Strategy
- **Scaffold-aware split**:
  - Computed Bemis–Murcko scaffolds via RDKit: True
  - Grouped by scaffold, then split into Train/Val/Test (70/15/15) by **scaffold blocks**
  - Stratified on "scaffold contains any active" to balance rare actives across splits
  - Post-fix: ensured at least 1 positive in Val/Test when possible

## Split Stats
- Train: n=342, actives=97 (28.36%), scaffolds=111
- Val:   n=70, actives=30 (42.86%), scaffolds=24
- Test:  n=74, actives=23 (31.08%), scaffolds=24

## Notes
- Threshold 25 μg/mL corresponds to log10 ≈ 1.39794.
- If you need exact reproducibility, fix `random_state=2025` (already used).
- For model training:
  - Use **Huber** or **Quantile** loss for `logMIC` (q=0.1, 0.5, 0.9)。
  - Train a lightweight classifier for `is_active_<25` (optionally nnPU) 并联动到RL奖励。
- Files:
  - `train.csv`, `val.csv`, `test.csv` under `/mnt/data/split_aureus_20250924`.
