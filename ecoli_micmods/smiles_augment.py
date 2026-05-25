#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smiles_augment.py - SMILES随机化增强（分层策略）

核心功能：
1. 生成化学等价的随机SMILES表示
2. 分层增强策略（excellent×4, good×4, moderate×2, poor×0）
3. 为增强样本重新生成4274D特征

创建日期：2025-10-10
版本：v1.0
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from .utils import bin_by_mic


def randomize_smiles(smiles, n_variants=1, random_seed=None):
    """
    生成SMILES的随机化等价表示

    Args:
        smiles: str - 原始SMILES字符串
        n_variants: int - 生成变体数量
        random_seed: int - 随机种子（可选）

    Returns:
        list of str - 随机化SMILES列表（化学等价）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # 设置随机种子（如果提供）
    if random_seed is not None:
        np.random.seed(random_seed)

    random_smiles = []
    for _ in range(n_variants):
        random_smi = Chem.MolToSmiles(mol, doRandom=True)
        random_smiles.append(random_smi)

    return random_smiles


def augment_dataset_stratified(
    df_train,
    task='ecoli',
    smiles_col='smiles',
    feat_start=5,
    feat_count=4274,
    augment_factors={'excellent': 4, 'good': 4, 'moderate': 2, 'poor': 0},
    random_seed=42
):
    """
    分层SMILES增强策略

    优先增强低MIC样本（excellent/good），减少高MIC样本过拟合风险

    Args:
        df_train: pd.DataFrame - 训练集（含SMILES和logMIC）
        task: str - 任务名（用于列名）
        smiles_col: str - SMILES列名
        feat_start: int - 特征起始列索引
        feat_count: int - 特征总数
        augment_factors: dict - 各MIC桶的增强倍数
        random_seed: int - 随机种子

    Returns:
        pd.DataFrame - 增强后训练集（原始 + 增强样本）
    """
    logmic_col = f'{task}_logMIC'

    if logmic_col not in df_train.columns:
        raise ValueError(f"找不到{logmic_col}列")

    # 检查是否已包含特征
    if df_train.shape[1] < feat_start + feat_count:
        raise ValueError(f"训练集缺少特征列，预期{feat_start + feat_count}列，实际{df_train.shape[1]}列")

    # 分桶统计
    df_train['mic_bin'] = df_train[logmic_col].apply(bin_by_mic)
    bin_counts = df_train['mic_bin'].value_counts()

    print(f"\n原始训练集MIC分布:")
    for bin_name in ['excellent', 'good', 'moderate', 'poor']:
        count = bin_counts.get(bin_name, 0)
        factor = augment_factors.get(bin_name, 0)
        print(f"  {bin_name}: {count}样本 -> 增强x{factor} -> {count * (1 + factor)}样本")

    # 导入特征生成函数（延迟导入避免循环依赖）
    try:
        import sys
        import os
        # 尝试多种路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from enhanced_datasets import (
            build_descriptors, build_ecfp4_counts_hashed,
            build_maccs_bits, build_rdk_bits
        )
    except ImportError as e:
        raise ImportError(
            f"无法导入特征生成函数: {e}\n"
            "请确保 enhanced_datasets.py 位于项目根目录。\n"
            "如果暂时无法重算4274D特征，请改用 augment_simple(...)，\n"
            "随后对合并后的CSV统一运行一次特征生成脚本。"
        )

    # 分层增强
    augmented_rows = []
    np.random.seed(random_seed)

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="SMILES增强"):
        smiles_orig = row[smiles_col]
        mic_bin = row['mic_bin']
        factor = augment_factors.get(mic_bin, 0)

        if factor == 0:
            continue  # poor桶不增强

        # 生成随机SMILES
        random_smiles = randomize_smiles(smiles_orig, n_variants=factor, random_seed=random_seed + idx)

        for aug_smi in random_smiles:
            try:
                # 重新生成4274D特征
                desc = build_descriptors([aug_smi])
                ecfp4 = build_ecfp4_counts_hashed([aug_smi], radius=2, hash_mod=2048)
                maccs = build_maccs_bits([aug_smi], bits=167, start_index=0)
                rdk = build_rdk_bits([aug_smi], fp_size=2048, min_path=1, max_path=7)

                # 拼接特征（4274D）
                features = np.hstack([
                    desc.values[0],
                    ecfp4.values[0],
                    maccs.values[0],
                    rdk.values[0]
                ])

                # 构造新行（复制元数据列 + 新特征 + 保留其他列）
                # 策略：复制整行，然后替换SMILES和特征部分
                new_row = row.tolist()

                # 更新SMILES列
                smiles_idx = df_train.columns.get_loc(smiles_col)
                new_row[smiles_idx] = aug_smi

                # 更新特征列（feat_start开始的feat_count列）
                for i, feat_val in enumerate(features):
                    new_row[feat_start + i] = feat_val

                augmented_rows.append(new_row)

            except Exception as e:
                print(f"\n警告: SMILES {aug_smi} 特征生成失败 ({str(e)})，跳过")
                continue

    # 合并原始 + 增强
    df_aug = pd.DataFrame(augmented_rows, columns=df_train.columns)
    df_combined = pd.concat([df_train, df_aug], ignore_index=True)

    # 移除临时列
    df_combined.drop(columns=['mic_bin'], inplace=True, errors='ignore')
    df_train.drop(columns=['mic_bin'], inplace=True, errors='ignore')

    # 最终统计
    final_bin_counts = df_combined[logmic_col].apply(bin_by_mic).value_counts()
    print(f"\n增强后训练集: {len(df_train)} -> {len(df_combined)} 样本 (x{len(df_combined)/len(df_train):.1f})")
    print(f"最终MIC分布:")
    for bin_name in ['excellent', 'good', 'moderate', 'poor']:
        count = final_bin_counts.get(bin_name, 0)
        print(f"  {bin_name}: {count}样本")

    return df_combined


def augment_simple(df_train, smiles_col='smiles', n_aug=3, random_seed=42):
    """
    简单全局增强策略（不分层）

    仅生成随机SMILES，不重新计算特征（需后续调用enhanced_datasets.py）

    Args:
        df_train: pd.DataFrame - 训练集
        smiles_col: str - SMILES列名
        n_aug: int - 每个分子增强倍数
        random_seed: int - 随机种子

    Returns:
        pd.DataFrame - 增强后训练集（仅SMILES列改变，特征列保持原值）
    """
    if n_aug <= 0:
        return df_train.copy()

    augmented_rows = []
    np.random.seed(random_seed)

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="简单SMILES增强"):
        smiles_orig = row[smiles_col]
        random_smiles = randomize_smiles(smiles_orig, n_variants=n_aug, random_seed=random_seed + idx)

        for aug_smi in random_smiles:
            new_row = row.copy()
            new_row[smiles_col] = aug_smi
            augmented_rows.append(new_row)

    df_aug = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([df_train, df_aug], ignore_index=True)

    print(f"\n简单增强: {len(df_train)} -> {len(df_combined)} 样本")
    print("注意: 特征列未更新，需重新运行enhanced_datasets.py")

    return df_combined
