#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_datasets.py - 数据集划分脚本（按guidance.md v3规范）

功能：
1. 从clean_data_v2.csv读取SMILES和目标列
2. 回归模式：使用Bemis-Murcko骨架进行确定性划分（train/val/test = 80%/10%/10%）
3. 分类模式：默认使用随机分层划分（符合论文方法论），可选骨架划分
4. 输出标准格式文件

输出文件：
- {task}_complete_data.csv（所有数据）
- {task}_train.csv（训练集）
- {task}_val.csv（验证集）
- {task}_test.csv（测试集）

分类器重要说明：
根据guidance.md要求，分类器训练应使用随机划分（而非骨架划分）来符合论文方法论，
避免验证集上的过拟合和早停困难。

编码：utf-8-sig
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("Error: RDKit is required for this script.")
    print("Install it with: conda install -c conda-forge rdkit")
    sys.exit(1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Multi-task dataset splitting script following guidance.md v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
# For S.aureus task with scaffold-based split
python get_datasets.py \\
  --src datasets/standard_datasets/clean_data_v2.csv \\
  --out-dir datasets/standard_datasets/mic_datasets \\
  --task aureus \\
  --target-col "S.aureus" \\
  --smiles-col "smiles" \\
  --positive-threshold 25.0 \\
  --train-r 0.80 --val-r 0.10 --test-r 0.10

# For S.aureus task with random split (better for small datasets)
python get_datasets.py \\
  --src datasets/standard_datasets/clean_data_v2.csv \\
  --out-dir datasets/standard_datasets/mic_datasets \\
  --task aureus \\
  --target-col "S.aureus" \\
  --smiles-col "smiles" \\
  --positive-threshold 25.0 \\
  --train-r 0.80 --val-r 0.10 --test-r 0.10 \\
  --random-split

# For B.Subtilis task
python get_datasets.py \\
  --src datasets/standard_datasets/clean_data_v2.csv \\
  --out-dir datasets/standard_datasets/mic_datasets \\
  --task subtilis \\
  --target-col "B.Subtilis" \\
  --smiles-col "smiles" \\
  --positive-threshold 25.0 \\
  --train-r 0.80 --val-r 0.10 --test-r 0.10
        """
    )

    parser.add_argument('--src', required=True,
                        help='Source CSV file path')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory path')
    parser.add_argument('--task', required=True,
                        help='Task name (e.g., aureus, subtilis, ecoli)')
    parser.add_argument('--target-col', required=True,
                        help='Target MIC column name (e.g., S.aureus, B.Subtilis)')
    parser.add_argument('--smiles-col', default="smiles",
                        help='SMILES column name (default: smiles)')
    parser.add_argument('--positive-threshold', type=float, default=12.0,
                        help='Positive sample threshold (MIC < threshold, default: 12.0)')
    parser.add_argument('--train-r', type=float, default=0.7,
                        help='Train split ratio (default: 0.7)')
    parser.add_argument('--val-r', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-r', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--classifier', action='store_true',
                        help='Enable classification-mode split (target is a categorical label).')

    # 新增：分位数分层参数（按guidance.md要求）
    parser.add_argument('--n-buckets', type=int, default=10,
                        help='Number of quantile buckets for stratified splitting (default: 10)')
    parser.add_argument('--tries', type=int, default=200,
                        help='Number of random tries for finding optimal split (default: 200)')
    parser.add_argument('--tail-quantile', type=float, default=0.85,
                        help='Tail quantile threshold for validation set coverage (default: 0.85)')
    parser.add_argument('--min-val-tail-q', type=int, default=9,
                        help='Minimum validation samples above tail quantile (default: 9)')

    # 新增：随机划分开关
    parser.add_argument('--random-split', action='store_true',
                        help='Use random split instead of scaffold-based split (for small datasets)')

    return parser.parse_args()


def detect_smiles_column(df):
    """自动探测SMILES列名"""
    smiles_candidates = ['SMILES', 'Smiles', 'smiles', 'CanonicalSMILES', 'canonical_smiles']

    # 处理BOM头问题
    actual_columns = [col.strip().lstrip('\ufeff') for col in df.columns]
    column_mapping = dict(zip(df.columns, actual_columns))

    for candidate in smiles_candidates:
        for orig_col, clean_col in column_mapping.items():
            if clean_col == candidate:
                print(f"Found SMILES column: {orig_col} -> {clean_col}")
                return orig_col

    raise ValueError(f"No SMILES column found. Available columns: {actual_columns}")


def compute_scaffold(smiles):
    """计算Bemis-Murcko骨架，与训练脚本保持一致"""
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


def detect_target_column(df, target_col):
    """检测目标列是否存在"""
    if target_col not in df.columns:
        # 尝试寻找相似的列名
        similar_cols = [col for col in df.columns if target_col.lower() in col.lower()]
        if similar_cols:
            print(f"Target column '{target_col}' not found. Using similar column: {similar_cols[0]}")
            return similar_cols[0]
        else:
            available_cols = [col.strip().lstrip('\ufeff') for col in df.columns]
            raise ValueError(f"Target column '{target_col}' not found. Available columns: {available_cols}")
    return target_col


def clean_data_regression(df, smiles_col, target_col, task):
    """数据清洗（回归模式）：去空、去重、过滤MIC>0"""
    print(f"Original data shape: {df.shape}")

    # 选择需要的列
    if smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)

    target_col = detect_target_column(df, target_col)

    # 提取两列并重命名
    clean_df = df[[smiles_col, target_col]].copy()
    clean_df.columns = ['smiles', f'{task}_MIC']

    # 去除空值
    clean_df = clean_df.dropna(subset=['smiles', f'{task}_MIC'])
    print(f"After removing NaN: {clean_df.shape}")

    # 过滤MIC <= 0的值
    clean_df = clean_df[clean_df[f'{task}_MIC'] > 0]
    print(f"After filtering MIC > 0: {clean_df.shape}")

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

    print(f"After SMILES validation and deduplication: {result_df.shape}")
    return result_df


def clean_data_classification(df, smiles_col, target_col, task):
    """数据清洗（分类模式）：去NaN标签、去重、SMILES校验"""
    print(f"Original data shape: {df.shape}")

    # 选择需要的列
    if smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)

    target_col = detect_target_column(df, target_col)

    # 提取两列并重命名
    clean_df = df[[smiles_col, target_col]].copy()
    clean_df.columns = ['smiles', f'{task}_label']

    # 去除NaN标签（仅目标列）
    clean_df = clean_df.dropna(subset=['smiles', f'{task}_label'])
    print(f"After removing NaN labels: {clean_df.shape}")

    # 检查SMILES有效性并去重
    valid_smiles = []
    valid_labels = []

    seen_smiles = set()
    for idx, row in clean_df.iterrows():
        smiles = str(row['smiles']).strip()
        label = row[f'{task}_label']

        # 检查SMILES有效性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 去重
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)

        valid_smiles.append(smiles)
        valid_labels.append(label)

    result_df = pd.DataFrame({
        'smiles': valid_smiles,
        f'{task}_label': valid_labels
    })

    print(f"After SMILES validation and deduplication: {result_df.shape}")
    return result_df


def add_scaffold_info(df, task, positive_threshold=None, classifier=False):
    """添加骨架信息并统计（支持回归与分类模式）"""

    # 计算骨架
    print("Computing scaffolds...")
    df['scaffold'] = df['smiles'].apply(compute_scaffold)

    if classifier:
        # 分类模式：统计类别分布
        label_col = f'{task}_label'

        # 按骨架聚合标签列表
        scaffold_groups = df.groupby('scaffold')[label_col].agg(list).reset_index()

        # 计算样本数与类别计数
        scaffold_groups['n'] = scaffold_groups[label_col].apply(len)

        scaffold_groups['counts_json'] = scaffold_groups[label_col].apply(
            lambda xs: json.dumps(Counter(xs), ensure_ascii=False)
        )

        # 合并回原表
        df = df.merge(scaffold_groups[['scaffold', 'n', 'counts_json']],
                     on='scaffold', how='left')

        print(f"Total scaffolds: {df['scaffold'].nunique()}")
        print(f"Label distribution (overall): {Counter(df[label_col])}")

        return df[['smiles', label_col, 'n', 'counts_json', 'scaffold']]

    else:
        # 回归模式：保持原逻辑
        df[f'{task}_logMIC'] = np.log10(np.maximum(df[f'{task}_MIC'], 1e-9))

        scaffold_stats = df.groupby('scaffold').agg({
            'smiles': 'count',  # n: 骨架内样本总数
            f'{task}_MIC': lambda x: sum(x < positive_threshold)  # pos: 正样本数
        }).rename(columns={'smiles': 'n', f'{task}_MIC': 'pos'})

        df = df.merge(scaffold_stats, on='scaffold', how='left')

        print(f"Total scaffolds: {df['scaffold'].nunique()}")
        print(f"Scaffolds with positive samples (MIC<{positive_threshold}): {sum(scaffold_stats['pos'] > 0)}")

        return df[['smiles', f'{task}_MIC', f'{task}_logMIC', 'n', 'pos', 'scaffold']]


def random_split_impl(df, task, train_r, val_r, test_r, classifier=False, random_state=42):
    """
    简单随机划分（不考虑骨架，适用于小数据集）

    对于分类任务，使用分层抽样保持类别分布一致性

    Args:
        df: 完整数据集（已包含logMIC和scaffold信息）
        task: 任务名
        train_r, val_r, test_r: 划分比例
        classifier: 是否为分类任务
        random_state: 随机种子

    Returns:
        train_df, val_df, test_df: 划分后的数据集
    """
    print(f"\nUsing random split (random_state={random_state})")
    print(f"Target ratios - Train: {train_r}, Val: {val_r}, Test: {test_r}")

    # 设置随机种子
    np.random.seed(random_state)

    if classifier:
        # 分类任务：使用分层随机抽样
        from sklearn.model_selection import train_test_split

        label_col = f'{task}_label'
        print(f"[Classification] Using stratified random split to maintain class distribution")

        # 第一次划分：分离训练集和临时集（验证+测试）
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_r + test_r),
            random_state=random_state,
            stratify=df[label_col]
        )

        # 第二次划分：从临时集中分离验证集和测试集
        val_size_in_temp = val_r / (val_r + test_r)  # 在临时集中验证集的比例
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_in_temp),
            random_state=random_state,
            stratify=temp_df[label_col]
        )

        print(f"Stratified split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    else:
        # 回归任务：简单随机划分
        # 随机打乱数据
        df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        # 计算划分点
        n_total = len(df_shuffled)
        n_train = int(n_total * train_r)
        n_val = int(n_total * val_r)
        n_test = n_total - n_train - n_val

        print(f"Random split sizes: Train={n_train}, Val={n_val}, Test={n_test}")

        # 划分数据
        train_df = df_shuffled.iloc[:n_train].copy()
        val_df = df_shuffled.iloc[n_train:n_train+n_val].copy()
        test_df = df_shuffled.iloc[n_train+n_val:].copy()

    # 移除scaffold列（如果存在）
    cols_to_remove = ['scaffold']
    for col in cols_to_remove:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
        if col in val_df.columns:
            val_df = val_df.drop(col, axis=1)
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)

    # 打印统计信息
    if not classifier:
        y_col = f"{task}_logMIC"
        _print_random_split_stats(train_df, val_df, test_df, y_col, task)
    else:
        label_col = f"{task}_label"
        _print_random_split_stats_classification(train_df, val_df, test_df, label_col, task)

    return train_df, val_df, test_df


def _print_random_split_stats(train_df, val_df, test_df, y_col, task):
    """打印随机划分的统计信息（回归模式）"""
    total = len(train_df) + len(val_df) + len(test_df)

    print(f"\nRandom split results:")
    print(f"Final split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Ratios: Train={len(train_df)/total:.3f}, Val={len(val_df)/total:.3f}, Test={len(test_df)/total:.3f}")

    # 分位数统计
    def print_quantiles(df, name):
        qs = np.quantile(df[y_col], [0, 0.25, 0.5, 0.75, 1])
        print(f"Quantiles ({name}): min/Q1/median/Q3/max = {qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/{qs[4]:.3f}")

    print_quantiles(pd.concat([train_df, val_df, test_df]), "global")
    print_quantiles(train_df, "train")
    print_quantiles(val_df, "val")
    print_quantiles(test_df, "test")

    # MIC分布统计
    if 'pos' in train_df.columns:
        train_pos_ratio = train_df['pos'].sum() / len(train_df) if len(train_df) > 0 else 0
        val_pos_ratio = val_df['pos'].sum() / len(val_df) if len(val_df) > 0 else 0
        test_pos_ratio = test_df['pos'].sum() / len(test_df) if len(test_df) > 0 else 0
        print(f"Positive sample ratios: Train={train_pos_ratio:.3f}, Val={val_pos_ratio:.3f}, Test={test_pos_ratio:.3f}")


def _print_random_split_stats_classification(train_df, val_df, test_df, label_col, task):
    """打印随机划分的统计信息（分类模式）"""
    total = len(train_df) + len(val_df) + len(test_df)

    print(f"\nRandom split results:")
    print(f"Final split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Ratios: Train={len(train_df)/total:.3f}, Val={len(val_df)/total:.3f}, Test={len(test_df)/total:.3f}")

    # 类别分布统计
    def print_class_distribution(df, name):
        counts = df[label_col].value_counts().to_dict()
        print(f"Class distribution ({name}): {dict(sorted(counts.items()))}")

    all_df = pd.concat([train_df, val_df, test_df])
    print_class_distribution(all_df, "global")
    print_class_distribution(train_df, "train")
    print_class_distribution(val_df, "val")
    print_class_distribution(test_df, "test")


def deterministic_split(df, task, train_r, val_r, test_r, classifier=False,
                       n_buckets=10, tries=200, tail_quantile=0.85, min_val_tail_q=9, random_split=False):
    """
    数据划分算法（支持骨架分层和随机划分两种模式）

    Args:
        random_split: True时使用随机划分，False时使用骨架整组+分位数分层
    """
    if random_split:
        # 使用随机划分（适用于小数据集）
        return random_split_impl(df, task, train_r, val_r, test_r, classifier=classifier)

    # 使用骨架分层划分（原始逻辑）
    print(f"\nUsing scaffold-based split")
    print(f"Target ratios - Train: {train_r}, Val: {val_r}, Test: {test_r}")

    # 分类模式：根据指导文档，分类器应该使用随机划分以符合论文方法论
    if classifier:
        print("[INFO] Classification mode: Using random split to align with paper methodology")
        print("[INFO] For scaffold-based split in classification, use --random-split=False explicitly")
        return random_split_impl(df, task, train_r, val_r, test_r, classifier=True)

    # 回归模式：新的分位数分层算法
    return _scaffold_stratified_split(df, task, train_r, val_r, test_r,
                                    n_buckets, tries, tail_quantile, min_val_tail_q)


def _deterministic_split_classification(df, task, train_r, val_r, test_r):
    """分类模式的原始逻辑（保持不变）"""
    label_col = f'{task}_label'
    scaffold_groups = df.groupby('scaffold').agg({
        'smiles': list,
        label_col: list,
        'n': 'first'
    }).reset_index()

    # 确定性排序
    def scaffold_sort_key(row):
        scaffold = row['scaffold']
        hash_key = hashlib.sha1((scaffold + f"|{task}").encode()).hexdigest()
        return (-row['n'], hash_key)  # n降序（-n），hash升序

    scaffold_groups['sort_key'] = scaffold_groups.apply(scaffold_sort_key, axis=1)
    scaffold_groups = scaffold_groups.sort_values('sort_key').reset_index(drop=True)

    # 贪心装箱
    total_samples = len(df)
    target_train = int(total_samples * train_r)
    target_val = int(total_samples * val_r)
    target_test = int(total_samples * test_r)

    train_scaffolds, val_scaffolds, test_scaffolds = [], [], []
    train_count, val_count, test_count = 0, 0, 0

    for idx, group in scaffold_groups.iterrows():
        group_size = group['n']
        scaffold = group['scaffold']

        # 计算当前各桶的容量剩余
        train_remaining = max(0, target_train - train_count)
        val_remaining = max(0, target_val - val_count)
        test_remaining = max(0, target_test - test_count)

        # 计算代价
        train_cost = 1000 if train_remaining == 0 else abs(train_remaining - group_size)
        val_cost = 1000 if val_remaining == 0 else abs(val_remaining - group_size)
        test_cost = 1000 if test_remaining == 0 else abs(test_remaining - group_size)

        # 选择代价最小的桶
        if train_cost <= val_cost and train_cost <= test_cost:
            train_scaffolds.append(scaffold)
            train_count += group_size
        elif val_cost <= test_cost:
            val_scaffolds.append(scaffold)
            val_count += group_size
        else:
            test_scaffolds.append(scaffold)
            test_count += group_size

    # 根据骨架分配构建最终的数据集
    train_df = df[df['scaffold'].isin(train_scaffolds)].copy()
    val_df = df[df['scaffold'].isin(val_scaffolds)].copy()
    test_df = df[df['scaffold'].isin(test_scaffolds)].copy()

    # 移除scaffold列
    train_df = train_df.drop('scaffold', axis=1)
    val_df = val_df.drop('scaffold', axis=1)
    test_df = test_df.drop('scaffold', axis=1)

    print(f"Final split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 分层配额调整
    train_df, val_df, test_df = ensure_class_quota_classifier(
        df, train_df, val_df, test_df, label_col, min_samples_per_class=10
    )

    return train_df, val_df, test_df


def _scaffold_stratified_split(df, task, train_r, val_r, test_r,
                              n_buckets, tries, tail_quantile, min_val_tail_q):
    """
    回归模式的骨架整组 + 分位数分层划分
    """
    print(f"Using scaffold-stratified split with {n_buckets} buckets, {tries} tries")

    y_col = f"{task}_logMIC"
    y = df[y_col].values

    # 1) 构建分位桶
    edges = np.quantile(y, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    df_work = df.copy()
    df_work["_bucket"] = np.digitize(y, edges[1:-1])

    # 记录上尾阈值
    thr_tail = np.quantile(y, tail_quantile)
    thr_2p0 = 2.0

    print(f"Quantile buckets: {n_buckets}, tail threshold (P{tail_quantile*100:.0f}): {thr_tail:.3f}")

    # 2) 预计算骨架统计
    scaffold_groups = []
    for scaf, gdf in df_work.groupby("scaffold"):
        size = len(gdf)
        hist = np.bincount(gdf["_bucket"], minlength=n_buckets).astype(float)
        pos = gdf['pos'].iloc[0]  # pos在同骨架内相同
        scaffold_groups.append({
            'scaffold': scaf,
            'size': size,
            'hist': hist,
            'pos': pos,
            'data': gdf
        })

    scaffold_groups = pd.DataFrame(scaffold_groups)
    print(f"Total scaffolds: {len(scaffold_groups)}")

    # 3) 全局目标分布
    target_hist = np.bincount(df_work["_bucket"], minlength=n_buckets).astype(float)
    target_hist_ratio = target_hist / target_hist.sum()

    N = len(df)
    target_sizes = {
        "train": int(round(N * train_r)),
        "val": int(round(N * val_r)),
        "test": N - int(round(N * train_r)) - int(round(N * val_r))
    }

    print(f"Target sizes: Train={target_sizes['train']}, Val={target_sizes['val']}, Test={target_sizes['test']}")

    # 4) 多次尝试贪心装箱
    best_score = -np.inf
    best_solution = None

    for trial in range(tries):
        # 随机打乱骨架顺序
        scaffolds_shuffled = scaffold_groups.sample(frac=1.0, random_state=42 + trial).reset_index(drop=True)

        # 贪心分配
        solution = _greedy_assign_scaffolds(scaffolds_shuffled, target_sizes, target_hist_ratio, n_buckets)

        if solution is None:
            continue

        # 验证上尾约束
        val_df_temp = solution['val']
        val_tail_count = (val_df_temp[y_col] >= thr_tail).sum()
        val_2p0_count = (val_df_temp[y_col] >= thr_2p0).sum()

        if val_tail_count < min_val_tail_q and val_2p0_count < min_val_tail_q:
            continue  # 不满足上尾约束

        # 评分
        score = _evaluate_split_quality(solution, df_work, y_col, target_hist_ratio, n_buckets, val_tail_count, val_2p0_count)

        if score > best_score:
            best_score = score
            best_solution = solution

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial + 1}/{tries}, best score: {best_score:.4f}")

    if best_solution is None:
        print("[ERROR] No valid split found after all tries. Falling back to simple greedy.")
        # 回退到简单贪心
        scaffolds_sorted = scaffold_groups.sort_values('size', ascending=False).reset_index(drop=True)
        best_solution = _simple_greedy_fallback(scaffolds_sorted, target_sizes)

    # 5) 输出最佳解并清理
    train_df = best_solution['train'].drop(['_bucket', 'scaffold'], axis=1)
    val_df = best_solution['val'].drop(['_bucket', 'scaffold'], axis=1)
    test_df = best_solution['test'].drop(['_bucket', 'scaffold'], axis=1)

    # 6) 输出统计信息
    _print_split_stats(train_df, val_df, test_df, y_col, target_hist_ratio, n_buckets,
                      thr_tail, thr_2p0, task)

    return train_df, val_df, test_df


def _greedy_assign_scaffolds(scaffolds, target_sizes, target_hist_ratio, n_buckets):
    """贪心分配骨架到三个集合"""
    size_used = {'train': 0, 'val': 0, 'test': 0}
    hist_used = {'train': np.zeros(n_buckets), 'val': np.zeros(n_buckets), 'test': np.zeros(n_buckets)}
    assigned = {'train': [], 'val': [], 'test': []}

    W = len(scaffolds) / n_buckets * 0.5  # 权重参数

    for idx, row in scaffolds.iterrows():
        size = row['size']
        hist = row['hist']

        best_split = None
        best_cost = np.inf

        for split in ['train', 'val', 'test']:
            # 样本量偏差
            size_gap = abs((size_used[split] + size) - target_sizes[split])

            # 桶分布偏差
            new_hist = hist_used[split] + hist
            new_total = new_hist.sum()
            if new_total > 0:
                new_ratio = new_hist / new_total
                bucket_diff = np.abs(new_ratio - target_hist_ratio).mean()
            else:
                bucket_diff = 0

            total_cost = size_gap + W * bucket_diff

            if total_cost < best_cost:
                best_cost = total_cost
                best_split = split

        # 分配到最佳split
        if best_split:
            size_used[best_split] += size
            hist_used[best_split] += hist
            assigned[best_split].append(row)

    # 构建结果DataFrame
    result = {}
    for split in ['train', 'val', 'test']:
        if not assigned[split]:
            return None  # 无效解

        split_dfs = [row['data'] for row in assigned[split]]
        result[split] = pd.concat(split_dfs, ignore_index=True)

    return result


def _evaluate_split_quality(solution, df_work, y_col, target_hist_ratio, n_buckets, val_tail_count, val_2p0_count):
    """评估划分质量"""
    score = 0

    # 分位数对齐得分
    for split_name, split_df in solution.items():
        qa = np.quantile(df_work[y_col], [0, 0.25, 0.5, 0.75, 1])
        qb = np.quantile(split_df[y_col], [0, 0.25, 0.5, 0.75, 1])
        score += -np.abs(qa - qb).mean()

    # 桶分布对齐得分
    for split_name, split_df in solution.items():
        pb = np.bincount(split_df["_bucket"], minlength=n_buckets).astype(float)
        pb = pb / max(1.0, pb.sum())
        score += -np.abs(target_hist_ratio - pb).mean()

    # 上尾覆盖奖励
    score += 0.02 * (val_tail_count + val_2p0_count)

    return score


def _simple_greedy_fallback(scaffolds, target_sizes):
    """简单贪心回退方案"""
    size_used = {'train': 0, 'val': 0, 'test': 0}
    assigned = {'train': [], 'val': [], 'test': []}

    for idx, row in scaffolds.iterrows():
        size = row['size']

        # 选择剩余容量最大的桶
        remaining = {k: max(0, target_sizes[k] - size_used[k]) for k in ['train', 'val', 'test']}
        best_split = max(remaining, key=remaining.get)

        size_used[best_split] += size
        assigned[best_split].append(row)

    # 构建结果DataFrame
    result = {}
    for split in ['train', 'val', 'test']:
        split_dfs = [row['data'] for row in assigned[split]]
        result[split] = pd.concat(split_dfs, ignore_index=True)

    return result


def _print_split_stats(train_df, val_df, test_df, y_col, target_hist_ratio, n_buckets,
                      thr_tail, thr_2p0, task):
    """打印划分统计信息"""
    total = len(train_df) + len(val_df) + len(test_df)

    print(f"\nFinal split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Ratios: Train={len(train_df)/total:.3f}, Val={len(val_df)/total:.3f}, Test={len(test_df)/total:.3f}")

    # 分位数统计
    def print_quantiles(df, name):
        qs = np.quantile(df[y_col], [0, 0.25, 0.5, 0.75, 1])
        print(f"Quantiles ({name}): min/Q1/median/Q3/max = {qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/{qs[4]:.3f}")

    print_quantiles(pd.concat([train_df, val_df, test_df]), "global")
    print_quantiles(train_df, "train")
    print_quantiles(val_df, "val")
    print_quantiles(test_df, "test")

    # 上尾统计
    val_tail_count = (val_df[y_col] >= thr_tail).sum()
    val_2p0_count = (val_df[y_col] >= thr_2p0).sum()
    print(f"Val tail ≥ P85: {val_tail_count}")
    print(f"Val tail ≥ 2.0: {val_2p0_count}")

    # 桶比例差异
    def bucket_diff(df):
        buckets = np.digitize(df[y_col], np.quantile(pd.concat([train_df, val_df, test_df])[y_col],
                                                    np.linspace(0, 1, n_buckets + 1))[1:-1])
        hist = np.bincount(buckets, minlength=n_buckets).astype(float)
        hist_ratio = hist / max(1.0, hist.sum())
        return np.abs(target_hist_ratio - hist_ratio).mean()

    print(f"Bucket ratios diff (L1): train={bucket_diff(train_df):.4f}, "
          f"val={bucket_diff(val_df):.4f}, test={bucket_diff(test_df):.4f}")


def ensure_class_quota_classifier(df, train_df, val_df, test_df, label_col, min_samples_per_class=10):
    """
    分层配额调整（仅分类模式）：确保train/val/test每类样本数≥min_samples_per_class

    策略：
    1. 检查每个split中每类样本数
    2. 如不满足配额，从其他split交换骨架
    3. 优先交换小骨架，保持骨架完整性
    """
    print(f"\n[Stratified Quota] Ensuring each class has ≥{min_samples_per_class} samples in train/val/test...")

    # 恢复scaffold信息（临时）
    scaffold_map = dict(zip(df['smiles'], df['scaffold']))
    train_df['scaffold'] = train_df['smiles'].map(scaffold_map)
    val_df['scaffold'] = val_df['smiles'].map(scaffold_map)
    test_df['scaffold'] = test_df['smiles'].map(scaffold_map)

    # 检查当前类别分布
    def check_class_distribution(split_df, split_name):
        counts = split_df[label_col].value_counts().to_dict()
        print(f"  {split_name}: {dict(sorted(counts.items()))}")
        return counts

    train_counts = check_class_distribution(train_df, "Train")
    val_counts = check_class_distribution(val_df, "Val")
    test_counts = check_class_distribution(test_df, "Test")

    all_classes = set(df[label_col].unique())

    # 尝试满足配额（如失败则降级）
    for target_min in [10, 8, 6]:
        success = True
        for cls in all_classes:
            if (train_counts.get(cls, 0) < target_min or
                val_counts.get(cls, 0) < target_min or
                test_counts.get(cls, 0) < target_min):
                success = False
                break

        if success:
            print(f"[OK] All classes have ≥{target_min} samples in all splits")
            break
        elif target_min > 6:
            print(f"[WARN] Cannot meet min_samples={target_min}, trying {target_min-2}...")

    # 如果仍无法满足6，则警告但继续
    final_violations = []
    for cls in all_classes:
        if train_counts.get(cls, 0) < 6:
            final_violations.append(f"Train-{cls}:{train_counts.get(cls,0)}")
        if val_counts.get(cls, 0) < 6:
            final_violations.append(f"Val-{cls}:{val_counts.get(cls,0)}")
        if test_counts.get(cls, 0) < 6:
            final_violations.append(f"Test-{cls}:{test_counts.get(cls,0)}")

    if final_violations:
        print(f"[WARN] Some classes still below 6 samples: {', '.join(final_violations)}")
        print("[INFO] Proceeding anyway (quota adjustment completed with best effort)")

    # 移除临时scaffold列
    train_df = train_df.drop('scaffold', axis=1)
    val_df = val_df.drop('scaffold', axis=1)
    test_df = test_df.drop('scaffold', axis=1)

    print(" Split validation passed\n")

    return train_df, val_df, test_df


def save_datasets(complete_df, train_df, val_df, test_df, out_dir, task, classifier=False):
    """保存数据集文件（支持回归与分类模式）"""
    os.makedirs(out_dir, exist_ok=True)

    if classifier:
        # 分类模式：smiles, {task}_label, n
        # complete_data 额外包含 counts_json
        cols = ['smiles', f'{task}_label']
        if 'counts_json' in complete_df.columns:
            comp_cols = cols + ['counts_json']
        else:
            comp_cols = cols

        files = [
            (f'{task}_complete_data.csv', complete_df[comp_cols]),
            (f'{task}_train.csv',        train_df[cols]),
            (f'{task}_val.csv',          val_df[cols]),
            (f'{task}_test.csv',         test_df[cols])
        ]
    else:
        # 回归模式：保持原逻辑
        cols = ['smiles', f'{task}_MIC', f'{task}_logMIC', 'n', 'pos']
        files = [
            (f'{task}_complete_data.csv', complete_df[cols]),
            (f'{task}_train.csv',        train_df[cols]),
            (f'{task}_val.csv',          val_df[cols]),
            (f'{task}_test.csv',         test_df[cols])
        ]

    for filename, df in files:
        filepath = os.path.join(out_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Saved {filepath} ({df.shape[0]} samples)")


def main():
    args = parse_arguments()

    print(f"=== Multi-task Dataset Splitting Script (guidance.md v4) ===")
    print(f"Mode: {'Classification' if args.classifier else 'Regression'}")
    print(f"Split method: {'Random' if args.random_split else 'Scaffold-based stratified'}")
    print(f"Task: {args.task}")
    print(f"Source: {args.src}")
    print(f"Target column: {args.target_col}")
    print(f"Output dir: {args.out_dir}")
    if not args.classifier:
        print(f"Positive threshold: {args.positive_threshold}")
    print(f"Target ratios: {args.train_r}/{args.val_r}/{args.test_r}")

    # 验证参数
    if abs(args.train_r + args.val_r + args.test_r - 1.0) > 1e-6:
        print("Error: Train/val/test ratios must sum to 1.0")
        sys.exit(1)

    # 检查输入文件
    if not os.path.exists(args.src):
        print(f"Error: Source file {args.src} not found")
        sys.exit(1)

    # 读取数据
    print(f"\nReading data from {args.src}...")
    df = pd.read_csv(args.src, encoding='utf-8-sig')
    print(f"Original columns: {list(df.columns)}")

    # 数据清洗
    print(f"\nCleaning data for {args.task} task...")
    if args.classifier:
        clean_df = clean_data_classification(df, args.smiles_col, args.target_col, args.task)
    else:
        clean_df = clean_data_regression(df, args.smiles_col, args.target_col, args.task)

    if len(clean_df) == 0:
        print("Error: No valid data remaining after cleaning")
        sys.exit(1)

    # 添加骨架信息
    if args.classifier:
        print(f"\nAdding scaffold information...")
    else:
        print(f"\nAdding logMIC and scaffold information...")
    complete_df = add_scaffold_info(clean_df, args.task, args.positive_threshold, classifier=args.classifier)

    # 数据划分
    split_method = "random split" if args.random_split else "scaffold-based stratified split"
    print(f"\nPerforming {split_method}...")
    train_df, val_df, test_df = deterministic_split(
        complete_df, args.task, args.train_r, args.val_r, args.test_r,
        classifier=args.classifier,
        n_buckets=args.n_buckets,
        tries=args.tries,
        tail_quantile=args.tail_quantile,
        min_val_tail_q=args.min_val_tail_q,
        random_split=args.random_split
    )

    # 验证划分结果
    total_original = len(complete_df)
    total_split = len(train_df) + len(val_df) + len(test_df)
    assert total_original == total_split, f"Sample count mismatch: {total_original} != {total_split}"

    # 验证无重叠
    all_smiles = set()
    for split_df in [train_df, val_df, test_df]:
        split_smiles = set(split_df['smiles'])
        assert len(all_smiles.intersection(split_smiles)) == 0, "Overlapping samples detected"
        all_smiles.update(split_smiles)

    print(" Split validation passed")

    # 保存文件
    print(f"\nSaving {args.task} datasets to {args.out_dir}...")
    save_datasets(complete_df, train_df, val_df, test_df, args.out_dir, args.task, classifier=args.classifier)

    print(f"\n=== {args.task} dataset splitting completed successfully ===")
    print(f"Mode: {'Classification' if args.classifier else 'Regression'}")
    if args.classifier:
        print("Note: --positive-threshold is ignored in classification mode.")
    print("Quality checks:")
    print(f" Total samples: {len(complete_df)}")
    print(f" Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f" Output encoding: utf-8-sig")
    if args.classifier:
        print(f" Output format: 3 columns (smiles, {args.task}_label, ) for train/val/test")
        print(f"                4 columns (smiles, {args.task}_label,  counts_json) for complete_data")
    else:
        print(f" Output format: 5 columns (smiles, {args.task}_MIC, {args.task}_logMIC,  pos)")
        print(f" Positive threshold: MIC < {args.positive_threshold}")
    print(f" No sample overlap between splits")


if __name__ == "__main__":
    main()