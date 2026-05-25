#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_qed_sa.py - 为CSV文件中的分子计算QED和SA分数

功能：
- 读取包含smiles列的CSV文件
- 计算每个分子的QED（药物相似性）和SA（合成难度）
- 将QED和SA追加为新列
- 覆盖原文件保存

用法：
    python get_qed_sa.py --csv molecules/your_file.csv
    python get_qed_sa.py --csv molecules/your_file.csv --smiles-col molecule_smiles
"""

import argparse
import pandas as pd
import sys
import os

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import QED

# Import SA Score
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except Exception as e:
    print(f"[ERROR] Cannot import SA Score module: {e}")
    print("[INFO] Make sure RDKit is properly installed with contrib modules")
    sys.exit(1)


def calculate_qed_sa(smiles: str):
    """
    Calculate QED and SA for a SMILES string

    Args:
        smiles: SMILES string

    Returns:
        tuple: (QED, SA) or (None, None) if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        qed = QED.qed(mol)
        sa = sascorer.calculateScore(mol)

        return float(qed), float(sa)
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate QED and SA scores for molecules in CSV file"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file containing SMILES"
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Name of SMILES column (default: smiles)"
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.csv):
        print(f"[ERROR] File not found: {args.csv}")
        sys.exit(1)

    # Read CSV
    print(f"[INFO] Reading CSV: {args.csv}")
    try:
        df = pd.read_csv(args.csv, encoding='utf-8-sig')
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    # Check if SMILES column exists
    if args.smiles_col not in df.columns:
        print(f"[ERROR] SMILES column '{args.smiles_col}' not found in CSV")
        print(f"[INFO] Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    print(f"[INFO] Found {len(df)} molecules")

    # Calculate QED and SA
    print("[INFO] Calculating QED and SA scores...")
    qed_scores = []
    sa_scores = []

    for i, smiles in enumerate(df[args.smiles_col], 1):
        qed, sa = calculate_qed_sa(smiles)
        qed_scores.append(qed)
        sa_scores.append(sa)

        if i % 100 == 0 or i == len(df):
            print(f"[INFO] Progress: {i}/{len(df)}", end='\r')

    print()  # New line after progress

    # Add scores to dataframe
    df['QED'] = qed_scores
    df['SA'] = sa_scores

    # Count valid scores
    valid_count = sum(1 for q, s in zip(qed_scores, sa_scores) if q is not None and s is not None)
    invalid_count = len(df) - valid_count

    print(f"[INFO] Successfully calculated: {valid_count}/{len(df)}")
    if invalid_count > 0:
        print(f"[WARN] Failed to calculate: {invalid_count} molecules (invalid SMILES)")

    # Statistics
    qed_valid = [q for q in qed_scores if q is not None]
    sa_valid = [s for s in sa_scores if s is not None]

    if qed_valid:
        import numpy as np
        print(f"\n[STATS] QED: mean={np.mean(qed_valid):.3f}, "
              f"median={np.median(qed_valid):.3f}, "
              f"range=[{min(qed_valid):.3f}, {max(qed_valid):.3f}]")
        print(f"[STATS] SA:  mean={np.mean(sa_valid):.2f}, "
              f"median={np.median(sa_valid):.2f}, "
              f"range=[{min(sa_valid):.2f}, {max(sa_valid):.2f}]")

    # Save to original file (overwrite)
    print(f"\n[INFO] Overwriting original file: {args.csv}")
    df.to_csv(args.csv, index=False, encoding='utf-8-sig')
    print(f"[DONE] Saved {len(df)} molecules with QED and SA scores")
    print(f"[INFO] Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()