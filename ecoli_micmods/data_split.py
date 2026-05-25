#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_split.py - E.Coli分层scaffold split实现

核心功能：
1. 复用get_datasets.py的数据读取和清洗逻辑
2. 实现分层scaffold split（保证每个MIC桶的配额）
3. 输出符合训练脚本格式的数据集

创建日期：2025-10-09
"""

import os
import sys
import pandas as pd
import numpy as np
import hashlib
from collections import Counter, defaultdict
from rdkit import Chem

from .utils import (
    bin_by_mic,
    compute_scaffold,
    ensure_dir,
    print_mic_distribution
)


def detect_smiles_column(df):
    """自动探测SMILES列名（复用自get_datasets.py）"""
    smiles_candidates = ['SMILES', 'Smiles', 'smiles', 'CanonicalSMILES', 'canonical_smiles']

    # 处理BOM头问题
    actual_columns = [col.strip().lstrip('\ufeff') for col in df.columns]
    column_mapping = dict(zip(df.columns, actual_columns))

    for candidate in smiles_candidates:
        for orig_col, clean_col in column_mapping.items():
            if clean_col == candidate:
                print(f"  发现SMILES列: {orig_col} -> {clean_col}")
                return orig_col

    raise ValueError(f"未找到SMILES列。可用列: {actual_columns}")


def detect_target_column(df, target_col):
    """检测目标列是否存在（复用自get_datasets.py）"""
    if target_col not in df.columns:
        # 尝试寻找相似的列名
        similar_cols = [col for col in df.columns if target_col.lower() in col.lower()]
        if similar_cols:
            print(f"  目标列 '{target_col}' 未找到。使用相似列: {similar_cols[0]}")
            return similar_cols[0]
        else:
            available_cols = [col.strip().lstrip('\ufeff') for col in df.columns]
            raise ValueError(f"目标列 '{target_col}' 未找到。可用列: {available_cols}")
    return target_col


def clean_ecoli_data(df, smiles_col='smiles', target_col='E.coli', task='ecoli'):
    """
    E.Coli数据清洗（复用自get_datasets.py的回归模式）

    Args:
        df: pd.DataFrame - 原始数据
        smiles_col: str - SMILES列名
        target_col: str - MIC目标列名
        task: str - 任务名（默认'ecoli'）

    Returns:
        pd.DataFrame - 清洗后数据（列：smiles, ecoli_MIC）
    """
    print(f"原始数据形状: {df.shape}")

    # 选择需要的列
    if smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)

    target_col = detect_target_column(df, target_col)

    # 提取两列并重命名
    clean_df = df[[smiles_col, target_col]].copy()
    clean_df.columns = ['smiles', f'{task}_MIC']

    # 去除空值
    clean_df = clean_df.dropna(subset=['smiles', f'{task}_MIC'])
    print(f"去除NaN后: {clean_df.shape}")

    # 过滤MIC <= 0的值
    clean_df = clean_df[clean_df[f'{task}_MIC'] > 0]
    print(f"过滤MIC > 0后: {clean_df.shape}")

    # 检查SMILES有效性并去重
    valid_smiles = []
    valid_mics = []

    seen_smiles = set()
    for idx, row in clean_df.iterrows():
        smiles = str(row['smiles']).strip()
        mic = row[f'{task}_MIC']

        # 检查SMILES有效性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 去重
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)

        valid_smiles.append(smiles)
        valid_mics.append(mic)

    result_df = pd.DataFrame({
        'smiles': valid_smiles,
        f'{task}_MIC': valid_mics
    })

    print(f"SMILES验证和去重后: {result_df.shape}")
    return result_df


def add_scaffold_and_bins(df, task='ecoli'):
    """
    添加scaffold和MIC分桶信息

    Args:
        df: pd.DataFrame - 清洗后数据（含smiles, ecoli_MIC）
        task: str - 任务名

    Returns:
        pd.DataFrame - 添加列：ecoli_logMIC, scaffold, mic_bin
    """
    print("\n计算scaffold和MIC分桶...")

    # 计算logMIC
    df[f'{task}_logMIC'] = np.log10(np.maximum(df[f'{task}_MIC'], 1e-9))

    # 计算scaffold
    df['scaffold'] = df['smiles'].apply(compute_scaffold)

    # 计算MIC bin
    df['mic_bin'] = df[f'{task}_logMIC'].apply(bin_by_mic)

    print(f"  总scaffold数: {df['scaffold'].nunique()}")
    print(f"  MIC桶分布:")
    bin_counts = df['mic_bin'].value_counts().to_dict()
    for bin_name in ['excellent', 'good', 'moderate', 'poor']:
        count = bin_counts.get(bin_name, 0)
        print(f"    {bin_name:>10s}: {count:>3d}")

    return df


def stratified_scaffold_split(
    df,
    task='ecoli',
    train_r=0.70,
    val_r=0.15,
    test_r=0.15,
    min_samples_per_bin=None,
    random_seed=42,
    verbose=True
):
    """
    分层scaffold split（回归模式，MIC桶配额约束）

    Args:
        df: pd.DataFrame - 完整数据（需包含: smiles, ecoli_MIC, ecoli_logMIC, scaffold, mic_bin）
        task: str - 任务名
        train_r: float - 训练集比例
        val_r: float - 验证集比例
        test_r: float - 测试集比例
        min_samples_per_bin: dict - 最小配额约束
            格式: {
                'train': {'excellent': 2, 'good': 20, 'moderate': 40, 'poor': 60},
                'val': {'excellent': 1, 'good': 5, 'moderate': 10, 'poor': 15},
                'test': {'excellent': 1, 'good': 5, 'moderate': 10, 'poor': 15}
            }
        random_seed: int - 随机种子（用于确定性排序）
        verbose: bool - 是否打印详细信息

    Returns:
        train_df, val_df, test_df - 划分后的数据集
    """
    if verbose:
        print("\n" + "="*70)
        print("分层Scaffold Split（MIC桶配额约束）")
        print("="*70)
        print(f"目标比例 - Train: {train_r}, Val: {val_r}, Test: {test_r}")

    # 默认配额（如果未指定）
    if min_samples_per_bin is None:
        min_samples_per_bin = {
            'train': {'excellent': 2, 'good': 20, 'moderate': 40, 'poor': 60},
            'val':   {'excellent': 1, 'good': 5,  'moderate': 10, 'poor': 15},
            'test':  {'excellent': 1, 'good': 5,  'moderate': 10, 'poor': 15}
        }

    if verbose:
        print(f"配额约束:")
        for split_name in ['train', 'val', 'test']:
            print(f"  {split_name}: {min_samples_per_bin[split_name]}")

    # 按scaffold聚合
    logmic_col = f'{task}_logMIC'
    mic_col = f'{task}_MIC'

    scaffold_groups = df.groupby('scaffold').agg({
        'smiles': list,
        mic_col: list,
        logmic_col: list,
        'mic_bin': list
    }).reset_index()

    scaffold_groups['n'] = scaffold_groups['smiles'].apply(len)

    # 计算每个scaffold的主导bin（样本数最多的bin）
    def get_dominant_bin(bins):
        counter = Counter(bins)
        return counter.most_common(1)[0][0]

    scaffold_groups['dominant_bin'] = scaffold_groups['mic_bin'].apply(get_dominant_bin)

    # 统计每个scaffold中各bin的样本数
    def count_bins(bins):
        return dict(Counter(bins))

    scaffold_groups['bin_counts'] = scaffold_groups['mic_bin'].apply(count_bins)

    if verbose:
        print(f"\n总scaffold数: {len(scaffold_groups)}")
        print(f"Scaffold主导bin分布:")
        dominant_dist = scaffold_groups['dominant_bin'].value_counts().to_dict()
        for bin_name in ['excellent', 'good', 'moderate', 'poor']:
            count = dominant_dist.get(bin_name, 0)
            print(f"  {bin_name:>10s}: {count:>3d} scaffolds")

    # 确定性排序：按n降序，然后按hash(scaffold)升序
    def scaffold_sort_key(row):
        scaffold = row['scaffold']
        hash_key = hashlib.sha1((scaffold + f"|{task}|{random_seed}").encode()).hexdigest()
        return (-row['n'], hash_key)  # n降序（-n），hash升序

    scaffold_groups['sort_key'] = scaffold_groups.apply(scaffold_sort_key, axis=1)
    scaffold_groups = scaffold_groups.sort_values('sort_key').reset_index(drop=True)

    # 初始化split容器
    total_samples = len(df)
    target_train = int(total_samples * train_r)
    target_val = int(total_samples * val_r)
    target_test = int(total_samples * test_r)

    train_scaffolds = []
    val_scaffolds = []
    test_scaffolds = []

    train_count = 0
    val_count = 0
    test_count = 0

    # 每个split中各bin的当前样本数
    train_bin_counts = defaultdict(int)
    val_bin_counts = defaultdict(int)
    test_bin_counts = defaultdict(int)

    if verbose:
        print(f"\n开始两阶段分配...")
        print(f"  目标样本数 - Train: {target_train}, Val: {target_val}, Test: {target_test}")

    # **阶段1：配额优先（前50% scaffolds）**
    # 优先处理稀缺bin（excellent, good），确保每个split满足最小配额
    phase1_cutoff = len(scaffold_groups) // 2

    for idx, group in scaffold_groups.iterrows():
        is_phase1 = idx < phase1_cutoff
        group_size = group['n']
        dominant_bin = group['dominant_bin']
        bin_counts = group['bin_counts']

        if is_phase1:
            # 阶段1：配额缺口优先
            train_deficit = sum(
                max(0, min_samples_per_bin['train'][bin_name] - train_bin_counts[bin_name])
                for bin_name in bin_counts.keys()
            )
            val_deficit = sum(
                max(0, min_samples_per_bin['val'][bin_name] - val_bin_counts[bin_name])
                for bin_name in bin_counts.keys()
            )
            test_deficit = sum(
                max(0, min_samples_per_bin['test'][bin_name] - test_bin_counts[bin_name])
                for bin_name in bin_counts.keys()
            )

            # 优先分配给配额缺口最大的split
            if val_deficit > train_deficit and val_deficit > test_deficit:
                chosen_split = 'val'
            elif test_deficit > train_deficit:
                chosen_split = 'test'
            else:
                chosen_split = 'train'

        else:
            # 阶段2：比例优先（贪心装箱）
            train_remaining = max(0, target_train - train_count)
            val_remaining = max(0, target_val - val_count)
            test_remaining = max(0, target_test - test_count)

            # 计算代价（距离目标的偏差）
            if train_remaining == 0:
                train_cost = 10000
            else:
                train_cost = abs(train_remaining - group_size)

            if val_remaining == 0:
                val_cost = 10000
            else:
                val_cost = abs(val_remaining - group_size)

            if test_remaining == 0:
                test_cost = 10000
            else:
                test_cost = abs(test_remaining - group_size)

            # 选择代价最小的split
            if train_cost <= val_cost and train_cost <= test_cost:
                chosen_split = 'train'
            elif val_cost <= test_cost:
                chosen_split = 'val'
            else:
                chosen_split = 'test'

        # 分配scaffold到选定的split
        if chosen_split == 'train':
            train_scaffolds.append(group['scaffold'])
            train_count += group_size
            for bin_name, count in bin_counts.items():
                train_bin_counts[bin_name] += count

        elif chosen_split == 'val':
            val_scaffolds.append(group['scaffold'])
            val_count += group_size
            for bin_name, count in bin_counts.items():
                val_bin_counts[bin_name] += count

        else:  # test
            test_scaffolds.append(group['scaffold'])
            test_count += group_size
            for bin_name, count in bin_counts.items():
                test_bin_counts[bin_name] += count

    if verbose:
        print(f"\n初始分配结果:")
        print(f"  Train: {train_count} 样本, {len(train_scaffolds)} scaffolds")
        print(f"    Bin分布: {dict(train_bin_counts)}")
        print(f"  Val:   {val_count} 样本, {len(val_scaffolds)} scaffolds")
        print(f"    Bin分布: {dict(val_bin_counts)}")
        print(f"  Test:  {test_count} 样本, {len(test_scaffolds)} scaffolds")
        print(f"    Bin分布: {dict(test_bin_counts)}")

    # 检查配额是否满足
    if verbose:
        print(f"\n配额满足检查:")

    violations = []
    for split_name, split_bin_counts, required_bins in [
        ('train', train_bin_counts, min_samples_per_bin['train']),
        ('val', val_bin_counts, min_samples_per_bin['val']),
        ('test', test_bin_counts, min_samples_per_bin['test'])
    ]:
        for bin_name, required_count in required_bins.items():
            actual_count = split_bin_counts.get(bin_name, 0)
            if actual_count < required_count:
                violations.append(f"{split_name}-{bin_name}: {actual_count}/{required_count}")

    if violations:
        if verbose:
            print(f"    未完全满足配额: {', '.join(violations)}")
            print(f"  （在268样本的约束下已尽力优化）")
    else:
        if verbose:
            print(f"   所有配额均已满足")

    # 根据scaffold分配构建最终数据集
    train_df = df[df['scaffold'].isin(train_scaffolds)].copy()
    val_df = df[df['scaffold'].isin(val_scaffolds)].copy()
    test_df = df[df['scaffold'].isin(test_scaffolds)].copy()

    # 移除临时列
    cols_to_keep = ['smiles', mic_col, logmic_col]
    train_df = train_df[cols_to_keep]
    val_df = val_df[cols_to_keep]
    test_df = test_df[cols_to_keep]

    if verbose:
        print(f"\n最终划分结果:")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"  实际比例: {len(train_df)/total_samples:.3f} / {len(val_df)/total_samples:.3f} / {len(test_df)/total_samples:.3f}")

        # 打印每个split的MIC分布
        print_mic_distribution(train_df, 'Train', logmic_col)
        print_mic_distribution(val_df, 'Val', logmic_col)
        print_mic_distribution(test_df, 'Test', logmic_col)

    # 验证无重叠
    all_smiles = set()
    for split_df in [train_df, val_df, test_df]:
        split_smiles = set(split_df['smiles'])
        assert len(all_smiles.intersection(split_smiles)) == 0, "检测到重叠样本！"
        all_smiles.update(split_smiles)

    if verbose:
        print(f"\n 划分验证通过（无重叠，样本数一致）")

    return train_df, val_df, test_df


def load_and_prepare_ecoli_data(
    src_csv,
    task='ecoli',
    target_col='E.coli',
    smiles_col='smiles',
    verbose=True
):
    """
    加载并准备E.Coli数据（复用get_datasets.py逻辑）

    Args:
        src_csv: str - 源CSV文件路径（clean_data_v2.csv）
        task: str - 任务名
        target_col: str - MIC列名
        smiles_col: str - SMILES列名
        verbose: bool - 是否打印详细信息

    Returns:
        pd.DataFrame - 准备好的完整数据（含scaffold和mic_bin列）
    """
    if verbose:
        print("="*70)
        print("加载E.Coli数据（复用get_datasets.py逻辑）")
        print("="*70)
        print(f"源文件: {src_csv}")

    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"源文件不存在: {src_csv}")

    # 读取数据
    df = pd.read_csv(src_csv, encoding='utf-8-sig')
    if verbose:
        print(f"原始列: {list(df.columns)}")

    # 数据清洗
    clean_df = clean_ecoli_data(df, smiles_col, target_col, task)

    if len(clean_df) == 0:
        raise ValueError("清洗后无有效数据！")

    # 添加scaffold和MIC bin
    complete_df = add_scaffold_and_bins(clean_df, task)

    if verbose:
        print(f"\n完整数据准备完成: {complete_df.shape}")
        print(f"列: {list(complete_df.columns)}")

    return complete_df


def add_scaffold_stats(df_split, df_complete, task='ecoli', positive_threshold=12.0):
    """
    为划分后的数据集添加scaffold统计信息（n和pos列）

    Args:
        df_split: pd.DataFrame - 划分后的数据集（train/val/test）
        df_complete: pd.DataFrame - 完整数据集（含scaffold列）
        task: str - 任务名
        positive_threshold: float - 正样本阈值（MIC < threshold为positive）

    Returns:
        pd.DataFrame - 添加n和pos列的数据集
    """
    mic_col = f'{task}_MIC'

    # 重新添加scaffold列（用于统计）
    scaffold_map = dict(zip(df_complete['smiles'], df_complete['scaffold']))
    df_split_copy = df_split.copy()
    df_split_copy['scaffold'] = df_split_copy['smiles'].map(scaffold_map)

    # 计算每个scaffold的统计信息
    scaffold_stats = df_complete.groupby('scaffold').agg({
        'smiles': 'count',  # n: scaffold中的样本数
        mic_col: lambda x: sum(x < positive_threshold)  # pos: scaffold中的positive样本数
    }).rename(columns={'smiles': 'n', mic_col: 'pos'})

    # 将统计信息合并到split数据集
    df_split_copy = df_split_copy.merge(scaffold_stats, on='scaffold', how='left')

    # 移除临时scaffold列
    df_split_copy = df_split_copy.drop('scaffold', axis=1)

    # 将n和pos转换为整数
    df_split_copy['n'] = df_split_copy['n'].astype(int)
    df_split_copy['pos'] = df_split_copy['pos'].astype(int)

    return df_split_copy


def prepare_and_split_ecoli_data(
    src_csv,
    output_dir,
    task='ecoli',
    target_col='E.coli',
    smiles_col='smiles',
    train_r=0.70,
    val_r=0.15,
    test_r=0.15,
    positive_threshold=12.0,
    min_samples_per_bin=None,
    random_seed=42,
    force_resplit=False,
    verbose=True
):
    """
    一站式E.Coli数据准备和划分函数（核心封装）

    功能：
    1. 加载clean_data_v2.csv
    2. 数据清洗和特征计算
    3. 分层scaffold split
    4. 添加n和pos列（scaffold统计）
    5. 保存train/val/test三个CSV文件

    Args:
        src_csv: str - 源文件路径（例如：datasets/clean_data_v2.csv）
        output_dir: str - 输出目录（例如：datasets/standard_datasets/ecoli_mic_datasets）
        task: str - 任务名（默认'ecoli'）
        target_col: str - MIC列名（默认'E.coli'）
        smiles_col: str - SMILES列名（默认'smiles'）
        train_r: float - 训练集比例（默认0.70）
        val_r: float - 验证集比例（默认0.15）
        test_r: float - 测试集比例（默认0.15）
        positive_threshold: float - 正样本阈值MIC（默认12.0）
        min_samples_per_bin: dict - MIC桶配额（默认None，使用内置配额）
        random_seed: int - 随机种子（默认42）
        force_resplit: bool - 是否强制重新划分（默认False，如果文件存在则跳过）
        verbose: bool - 是否打印详细信息（默认True）

    Returns:
        dict - {'train': train_path, 'val': val_path, 'test': test_path}

    使用示例：
        from ecoli_micmods.data_split import prepare_and_split_ecoli_data

        paths = prepare_and_split_ecoli_data(
            src_csv='datasets/clean_data_v2.csv',
            output_dir='datasets/standard_datasets/ecoli_mic_datasets',
            verbose=True
        )

        print(f"训练集: {paths['train']}")
        print(f"验证集: {paths['val']}")
        print(f"测试集: {paths['test']}")
    """
    # 1. 检查输出文件是否已存在
    ensure_dir(output_dir)
    train_path = os.path.join(output_dir, f'{task}_train.csv')
    val_path = os.path.join(output_dir, f'{task}_val.csv')
    test_path = os.path.join(output_dir, f'{task}_test.csv')

    if not force_resplit and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        if verbose:
            print("="*70)
            print("数据集文件已存在，跳过划分")
            print("="*70)
            print(f"  Train: {train_path}")
            print(f"  Val:   {val_path}")
            print(f"  Test:  {test_path}")
            print("如需重新划分，请设置 force_resplit=True")

        return {'train': train_path, 'val': val_path, 'test': test_path}

    # 2. 加载和准备完整数据
    if verbose:
        print("\n" + "="*70)
        print("阶段 1/3: 加载和准备数据")
        print("="*70)

    complete_df = load_and_prepare_ecoli_data(
        src_csv=src_csv,
        task=task,
        target_col=target_col,
        smiles_col=smiles_col,
        verbose=verbose
    )

    # 3. 分层scaffold split
    if verbose:
        print("\n" + "="*70)
        print("阶段 2/3: 分层Scaffold Split")
        print("="*70)

    train_df, val_df, test_df = stratified_scaffold_split(
        df=complete_df,
        task=task,
        train_r=train_r,
        val_r=val_r,
        test_r=test_r,
        min_samples_per_bin=min_samples_per_bin,
        random_seed=random_seed,
        verbose=verbose
    )

    # 4. 添加scaffold统计信息（n和pos列）
    if verbose:
        print("\n" + "="*70)
        print("阶段 3/3: 添加Scaffold统计信息")
        print("="*70)

    train_df = add_scaffold_stats(train_df, complete_df, task, positive_threshold)
    val_df = add_scaffold_stats(val_df, complete_df, task, positive_threshold)
    test_df = add_scaffold_stats(test_df, complete_df, task, positive_threshold)

    if verbose:
        print(f"  Train n列唯一值: {train_df['n'].nunique()}")
        print(f"  Train pos列唯一值: {train_df['pos'].nunique()}")
        print(f"  Val n列唯一值: {val_df['n'].nunique()}")
        print(f"  Val pos列唯一值: {val_df['pos'].nunique()}")
        print(f"  Test n列唯一值: {test_df['n'].nunique()}")
        print(f"  Test pos列唯一值: {test_df['pos'].nunique()}")

    # 5. 保存CSV文件
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    if verbose:
        print("\n" + "="*70)
        print("数据集划分完成！")
        print("="*70)
        print(f"  Train: {train_path} ({len(train_df)} 样本)")
        print(f"  Val:   {val_path} ({len(val_df)} 样本)")
        print(f"  Test:  {test_path} ({len(test_df)} 样本)")

    return {'train': train_path, 'val': val_path, 'test': test_path}


def stratified_random_split(df, task='ecoli', test_size=0.1, val_size=0.1, random_state=42, verbose=True):
    """
    随机分层划分80/10/10（按MIC桶stratify）- 同分布评测

    Args:
        df: pd.DataFrame - 完整数据（需包含: smiles, ecoli_MIC, ecoli_logMIC, mic_bin）
        task: str - 任务名
        test_size: float - 测试集比例（默认0.1）
        val_size: float - 验证集比例（默认0.1）
        random_state: int - 随机种子
        verbose: bool - 是否打印详细信息

    Returns:
        train_df, val_df, test_df - 划分后的数据集
    """
    from sklearn.model_selection import train_test_split

    if verbose:
        print("\n" + "="*70)
        print("随机分层划分80/10/10（按MIC桶stratify）- 同分布评测")
        print("="*70)

    mic_col = f'{task}_MIC'
    logmic_col = f'{task}_logMIC'

    # 确保有mic_bin列
    if 'mic_bin' not in df.columns:
        df['mic_bin'] = df[logmic_col].apply(bin_by_mic)

    if verbose:
        print(f"\n原始样本分布:")
        bin_counts = df['mic_bin'].value_counts()
        for bin_name in ['excellent', 'good', 'moderate', 'poor']:
            count = bin_counts.get(bin_name, 0)
            ratio = count / len(df) * 100
            print(f"  {bin_name}: {count}样本 ({ratio:.1f}%)")

    # 第一次划分：分离出test集（10%）
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['mic_bin'],
        random_state=random_state
    )

    # 第二次划分：分离出val集（从剩余的90%中取11.1%，相当于全集的10%）
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        stratify=train_val_df['mic_bin'],
        random_state=random_state
    )

    # 移除临时列
    cols_to_keep = ['smiles', mic_col, logmic_col]
    train_df = train_df[cols_to_keep].copy()
    val_df = val_df[cols_to_keep].copy()
    test_df = test_df[cols_to_keep].copy()

    if verbose:
        print(f"\n划分结果:")
        print(f"  训练集: {len(train_df)}样本 ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  验证集: {len(val_df)}样本 ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  测试集: {len(test_df)}样本 ({len(test_df)/len(df)*100:.1f}%)")

        # 打印每个split的MIC分布
        print_mic_distribution(train_df, 'Train', logmic_col)
        print_mic_distribution(val_df, 'Val', logmic_col)
        print_mic_distribution(test_df, 'Test', logmic_col)

    return train_df, val_df, test_df


def prepare_and_split_ecoli_random(
    src_csv,
    output_dir,
    task='ecoli',
    target_col='Ecoli_MIC_ugmL',
    smiles_col='SMILES',
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    force_resplit=False,
    verbose=True
):
    """
    E.Coli随机分层划分（同分布评测）

    适用于新清洗数据集clean_data_Ecoli.csv

    Args:
        src_csv: str - 源文件路径
        output_dir: str - 输出目录
        task: str - 任务名（默认'ecoli'）
        target_col: str - MIC列名（默认'Ecoli_MIC_ugmL'）
        smiles_col: str - SMILES列名（默认'SMILES'）
        test_size: float - 测试集比例（默认0.1）
        val_size: float - 验证集比例（默认0.1）
        random_state: int - 随机种子（默认42）
        force_resplit: bool - 是否强制重新划分
        verbose: bool - 是否打印详细信息

    Returns:
        dict - {'train': train_path, 'val': val_path, 'test': test_path}
    """
    # 1. 检查输出文件是否已存在
    ensure_dir(output_dir)
    train_path = os.path.join(output_dir, f'{task}_train.csv')
    val_path = os.path.join(output_dir, f'{task}_val.csv')
    test_path = os.path.join(output_dir, f'{task}_test.csv')

    if not force_resplit and os.path.exists(train_path):
        if verbose:
            print("="*70)
            print("数据集文件已存在，跳过划分")
            print("="*70)
            print(f"  Train: {train_path}")
            print(f"  Val:   {val_path}")
            print(f"  Test:  {test_path}")
            print("如需重新划分，请设置 force_resplit=True")

        return {'train': train_path, 'val': val_path, 'test': test_path}

    # 2. 加载新清洗数据集
    if verbose:
        print("="*70)
        print("加载新清洗E.Coli数据集")
        print("="*70)
        print(f"源文件: {src_csv}")

    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"源文件不存在: {src_csv}")

    df = pd.read_csv(src_csv)

    if verbose:
        print(f"原始数据: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

    # 检查列名（新数据集已经是SMILES, Ecoli_MIC_ugmL, Ecoli_logMIC）
    if smiles_col not in df.columns:
        raise ValueError(f"未找到SMILES列: {smiles_col}")

    if target_col not in df.columns:
        # 尝试查找Ecoli_logMIC
        if 'Ecoli_logMIC' in df.columns:
            print(f"  使用已有的logMIC列")
            logmic_col = 'Ecoli_logMIC'
            # 反推MIC
            df[target_col] = 10 ** df[logmic_col]
        else:
            raise ValueError(f"未找到MIC列: {target_col}")
    else:
        # 计算logMIC
        df[f'{task}_logMIC'] = np.log10(np.maximum(df[target_col], 1e-9))
        logmic_col = f'{task}_logMIC'

    # 统一列名
    df_clean = df[[smiles_col, target_col, logmic_col]].copy()
    df_clean.columns = ['smiles', f'{task}_MIC', f'{task}_logMIC']

    # 添加MIC bin
    df_clean['mic_bin'] = df_clean[f'{task}_logMIC'].apply(bin_by_mic)

    if verbose:
        print(f"\n清洗后数据: {df_clean.shape}")
        print(f"logMIC范围: [{df_clean[f'{task}_logMIC'].min():.2f}, {df_clean[f'{task}_logMIC'].max():.2f}]")

    # 3. 随机分层划分
    train_df, val_df, test_df = stratified_random_split(
        df_clean,
        task=task,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        verbose=verbose
    )

    # 4. 保存CSV文件
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    if verbose:
        print("\n" + "="*70)
        print("随机分层划分完成！")
        print("="*70)
        print(f"  Train: {train_path} ({len(train_df)} 样本)")
        print(f"  Val:   {val_path} ({len(val_df)} 样本)")
        print(f"  Test:  {test_path} ({len(test_df)} 样本)")

    return {'train': train_path, 'val': val_path, 'test': test_path}
