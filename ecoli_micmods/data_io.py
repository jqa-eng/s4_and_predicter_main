#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_io.py - 数据加载与样本重加权（对抗分布偏移）

核心功能：
1. 加载CSV并统一为log10(MIC)
2. 样本重加权（MIC桶权重 + MolLogP漂移校正）
3. 计算分布诊断指标（K-S检验、SMD）

创建日期：2025-10-10
版本：v1.0
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from rdkit import Chem
from rdkit.Chem import Descriptors

from .utils import bin_by_mic


def load_csv_with_logmic(csv_path, task='ecoli', encoding='utf-8-sig'):
    """
    加载CSV并确保logMIC列存在

    Args:
        csv_path: str - CSV文件路径
        task: str - 任务名（用于列名）
        encoding: str - 编码格式

    Returns:
        pd.DataFrame - 包含logMIC列的DataFrame
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    mic_col = f'{task}_MIC'
    logmic_col = f'{task}_logMIC'

    # 如果logMIC不存在，从MIC计算
    if logmic_col not in df.columns and mic_col in df.columns:
        df[logmic_col] = np.log10(np.maximum(df[mic_col], 1e-9))

    if logmic_col not in df.columns:
        raise ValueError(f"找不到{logmic_col}或{mic_col}列")

    return df


def compute_mologp(smiles_series):
    """
    计算MolLogP（脂水分配系数）

    Args:
        smiles_series: pd.Series - SMILES列

    Returns:
        np.array - MolLogP值数组
    """
    mologp_list = []

    for smiles in smiles_series:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mologp_list.append(np.nan)
            else:
                mologp_list.append(Descriptors.MolLogP(mol))
        except:
            mologp_list.append(np.nan)

    return np.array(mologp_list)


def compute_sample_weights_stratified(
    df_train,
    df_val,
    task='ecoli',
    mic_桶_weights={'excellent': 30, 'good': 8, 'moderate': 3, 'poor': 1},
    mologp_correction=True,
    mologp_bandwidth=0.5
):
    """
    计算样本权重（MIC桶权重 + MolLogP漂移校正）

    目标：让训练集的MIC分布和MolLogP分布向验证集靠拢

    Args:
        df_train: pd.DataFrame - 训练集
        df_val: pd.DataFrame - 验证集（用于计算目标分布）
        task: str - 任务名
        mic_桶_weights: dict - MIC桶的基准权重
        mologp_correction: bool - 是否启用MolLogP校正
        mologp_bandwidth: float - MolLogP核密度估计带宽

    Returns:
        np.array - 训练集样本权重（长度=len(df_train)）
    """
    logmic_col = f'{task}_logMIC'

    # 1. MIC桶权重
    train_bins = df_train[logmic_col].apply(bin_by_mic)
    bucket_weights = np.array([mic_桶_weights.get(b, 1.0) for b in train_bins])

    # 归一化桶权重（使均值=1）
    bucket_weights = bucket_weights / bucket_weights.mean()

    # 2. MolLogP漂移校正（可选）
    if mologp_correction and 'smiles' in df_train.columns and 'smiles' in df_val.columns:
        train_mologp = compute_mologp(df_train['smiles'])
        val_mologp = compute_mologp(df_val['smiles'])

        # 移除NaN
        train_mologp_valid = train_mologp[~np.isnan(train_mologp)]
        val_mologp_valid = val_mologp[~np.isnan(val_mologp)]

        if len(train_mologp_valid) > 0 and len(val_mologp_valid) > 0:
            # 计算Val集的MolLogP密度（目标分布）
            from scipy.stats import gaussian_kde
            val_kde = gaussian_kde(val_mologp_valid, bw_method=mologp_bandwidth)
            train_kde = gaussian_kde(train_mologp_valid, bw_method=mologp_bandwidth)

            # 重要性权重 = p_val(x) / p_train(x)
            mologp_weights = np.ones(len(df_train))

            for i, mologp in enumerate(train_mologp):
                if not np.isnan(mologp):
                    p_val = val_kde.evaluate(mologp)[0]
                    p_train = train_kde.evaluate(mologp)[0]

                    if p_train > 1e-6:
                        mologp_weights[i] = p_val / p_train
                    else:
                        mologp_weights[i] = 1.0

            # Clip权重（避免极端值）
            mologp_weights = np.clip(mologp_weights, 0.1, 10.0)

            # 归一化（使均值=1）
            mologp_weights = mologp_weights / mologp_weights.mean()
        else:
            mologp_weights = np.ones(len(df_train))
    else:
        mologp_weights = np.ones(len(df_train))

    # 3. 综合权重
    final_weights = bucket_weights * mologp_weights

    # 剪切到 [0.5, 2.0]（避免极端权重）
    final_weights = np.clip(final_weights, 0.5, 2.0)

    # 最终归一化（使均值=1）
    final_weights = final_weights / final_weights.mean()

    print(f"\n样本权重统计:")
    print(f"  MIC桶权重: min={bucket_weights.min():.2f}, max={bucket_weights.max():.2f}, mean={bucket_weights.mean():.2f}")
    print(f"  MolLogP权重: min={mologp_weights.min():.2f}, max={mologp_weights.max():.2f}, mean={mologp_weights.mean():.2f}")
    print(f"  综合权重（剪切前）: min={bucket_weights.min() * mologp_weights.min():.2f}, max={bucket_weights.max() * mologp_weights.max():.2f}")
    print(f"  综合权重（剪切后）: min={final_weights.min():.2f}, max={final_weights.max():.2f}, mean={final_weights.mean():.2f}")

    return final_weights


def diagnose_distribution_shift(df_train, df_val, df_test, task='ecoli'):
    """
    诊断Train/Val/Test的分布偏移

    返回K-S检验和MolLogP的SMD（Standardized Mean Difference）

    Args:
        df_train: pd.DataFrame - 训练集
        df_val: pd.DataFrame - 验证集
        df_test: pd.DataFrame - 测试集
        task: str - 任务名

    Returns:
        dict - 诊断结果
    """
    logmic_col = f'{task}_logMIC'

    # 1. K-S检验（logMIC分布）
    ks_train_val = ks_2samp(df_train[logmic_col], df_val[logmic_col])
    ks_train_test = ks_2samp(df_train[logmic_col], df_test[logmic_col])
    ks_val_test = ks_2samp(df_val[logmic_col], df_test[logmic_col])

    # 2. MolLogP SMD
    if 'smiles' in df_train.columns:
        train_mologp = compute_mologp(df_train['smiles'])
        val_mologp = compute_mologp(df_val['smiles'])
        test_mologp = compute_mologp(df_test['smiles'])

        # 移除NaN
        train_mologp = train_mologp[~np.isnan(train_mologp)]
        val_mologp = val_mologp[~np.isnan(val_mologp)]
        test_mologp = test_mologp[~np.isnan(test_mologp)]

        # SMD = (mean1 - mean2) / sqrt((std1^2 + std2^2) / 2)
        def compute_smd(x1, x2):
            mean1, mean2 = x1.mean(), x2.mean()
            std1, std2 = x1.std(), x2.std()
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        smd_train_val = compute_smd(train_mologp, val_mologp)
        smd_train_test = compute_smd(train_mologp, test_mologp)
    else:
        smd_train_val = None
        smd_train_test = None

    results = {
        'ks_train_val_p': ks_train_val.pvalue,
        'ks_train_test_p': ks_train_test.pvalue,
        'ks_val_test_p': ks_val_test.pvalue,
        'smd_mologp_train_val': smd_train_val,
        'smd_mologp_train_test': smd_train_test
    }

    print(f"\n分布偏移诊断:")
    print(f"  K-S检验（logMIC）:")
    print(f"    Train vs Val:  p={results['ks_train_val_p']:.4e} {'(显著)' if results['ks_train_val_p']<0.05 else '(不显著)'}")
    print(f"    Train vs Test: p={results['ks_train_test_p']:.4e} {'(显著)' if results['ks_train_test_p']<0.05 else '(不显著)'}")
    print(f"    Val vs Test:   p={results['ks_val_test_p']:.4e} {'(显著)' if results['ks_val_test_p']<0.05 else '(不显著)'}")

    if smd_train_val is not None:
        print(f"  MolLogP SMD:")
        print(f"    Train vs Val:  {smd_train_val:.3f} {'(大漂移)' if abs(smd_train_val)>0.5 else '(小漂移)'}")
        print(f"    Train vs Test: {smd_train_test:.3f} {'(大漂移)' if abs(smd_train_test)>0.5 else '(小漂移)'}")

    return results
