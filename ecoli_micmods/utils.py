#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - E.Coli MIC回归器工具函数

功能：
1. MIC分桶逻辑（4个bin：excellent/good/moderate/poor）
2. Scaffold计算（Bemis-Murcko）
3. 目录创建等通用工具
"""

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bin_by_mic(logMIC):
    """
    将logMIC分桶（模拟分类任务的类别）

    Args:
        logMIC: float - log10(MIC)值

    Returns:
        str - 桶标签（'excellent', 'good', 'moderate', 'poor'）

    分桶规则：
    - Excellent: logMIC < 0.477 (MIC < 3 μg/mL)
    - Good: 0.477 <= logMIC < 1.079 (3 <= MIC < 12)
    - Moderate: 1.079 <= logMIC < 1.5 (12 <= MIC < 31.6)
    - Poor: logMIC >= 1.5 (MIC >= 31.6)
    """
    if logMIC < 0.477:
        return 'excellent'
    elif logMIC < 1.079:
        return 'good'
    elif logMIC < 1.5:
        return 'moderate'
    else:
        return 'poor'


def compute_scaffold(smiles):
    """
    计算Bemis-Murcko骨架SMILES

    Args:
        smiles: str - SMILES字符串

    Returns:
        str - Scaffold SMILES（失败返回'[NO_SCAFFOLD]'）
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "[NO_SCAFFOLD]"

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return "[NO_SCAFFOLD]"

        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles if scaffold_smiles else "[NO_SCAFFOLD]"

    except Exception:
        return "[NO_SCAFFOLD]"


def ensure_dir(path):
    """
    确保目录存在

    Args:
        path: str - 目录路径
    """
    os.makedirs(path, exist_ok=True)


def validate_smiles(smiles):
    """
    验证SMILES有效性

    Args:
        smiles: str - SMILES字符串

    Returns:
        bool - 是否有效
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def compute_mic_bin_distribution(df, logmic_col='ecoli_logMIC'):
    """
    计算MIC桶的分布统计

    Args:
        df: pd.DataFrame - 数据集
        logmic_col: str - logMIC列名

    Returns:
        dict - {'excellent': count, 'good': count, ...}
    """
    from collections import Counter
    bins = df[logmic_col].apply(bin_by_mic)
    return dict(Counter(bins))


def print_mic_distribution(df, split_name='Dataset', logmic_col='ecoli_logMIC'):
    """
    打印MIC分布统计

    Args:
        df: pd.DataFrame - 数据集
        split_name: str - 数据集名称
        logmic_col: str - logMIC列名
    """
    dist = compute_mic_bin_distribution(df, logmic_col)
    total = len(df)

    print(f"\n{split_name} MIC分布 (n={total}):")
    for bin_name in ['excellent', 'good', 'moderate', 'poor']:
        count = dist.get(bin_name, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {bin_name:>10s}: {count:>3d} ({pct:>5.1f}%)")
