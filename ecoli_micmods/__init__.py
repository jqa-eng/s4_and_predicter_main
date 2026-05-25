#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecoli_micmods - E.Coli MIC回归器专用模块（论文对齐版）

完整功能：
1. 分层scaffold split（解决MIC分布偏移问题）已实现
2. 样本重加权（MIC桶 + MolLogP漂移校正）已实现
3. SMILES随机化增强（分层策略：excellent×4, good×4, moderate×2, poor×0）已实现
4. 两阶段特征选择（Stage1过滤 + Stage2 SelectKBest(mutual_info) 160→n_select，默认48维）已实现
5. GBDT模型选择（XGBoost → LightGBM → HGBR优先级）已实现

模块组成：
- data_split.py: 数据划分（分层scaffold split）
- data_io.py: 数据加载与样本重加权
- smiles_augment.py: SMILES随机化增强
- feat_select.py: 两阶段特征选择
- models.py: GBDT模型选择与训练
- utils.py: 工具函数（MIC分桶、scaffold计算等）

快速使用（数据划分）：
    from ecoli_micmods import prepare_and_split_ecoli_data

    paths = prepare_and_split_ecoli_data(
        src_csv='datasets/clean_data_v2.csv',
        output_dir='datasets/standard_datasets/ecoli_mic_datasets'
    )

快速使用（完整训练）：
    见 train_ecoli_paperalign.py

作者：Claude Code Assistant
创建日期：2025-10-09
最后更新：2025-10-10
"""

__version__ = '2.0.0'

# 数据划分
from .data_split import (
    prepare_and_split_ecoli_data,
    stratified_scaffold_split,
    load_and_prepare_ecoli_data,
    add_scaffold_stats
)

# 数据加载与重加权
from .data_io import (
    load_csv_with_logmic,
    compute_mologp,
    compute_sample_weights_stratified,
    diagnose_distribution_shift
)

# SMILES增强
from .smiles_augment import (
    randomize_smiles,
    augment_dataset_stratified,
    augment_simple
)

# 特征选择
from .feat_select import (
    stage1_filter,
    stage2_rfecv,
    select_features_pipeline,
    calculate_feature_stability
)

# 种子扫描（实验性功能，主流程已改用确定性特征选择+锁定机制，不推荐使用）
# from .seed_sweep import (
#     sweep_random_states,
#     select_best_seed_for_features,
#     find_best_seed_for_training
# )

# 模型
from .models import (
    get_best_gbdt_model,
    train_with_early_stopping,
    evaluate_regression_model,
    check_gpu_availability,
    get_feature_importance
)

# 工具函数
from .utils import (
    bin_by_mic,
    compute_scaffold,
    ensure_dir
)

__all__ = [
    # 数据划分
    'prepare_and_split_ecoli_data',
    'stratified_scaffold_split',
    'load_and_prepare_ecoli_data',
    'add_scaffold_stats',

    # 数据加载与重加权
    'load_csv_with_logmic',
    'compute_mologp',
    'compute_sample_weights_stratified',
    'diagnose_distribution_shift',

    # SMILES增强
    'randomize_smiles',
    'augment_dataset_stratified',
    'augment_simple',

    # 特征选择
    'stage1_filter',
    'stage2_rfecv',
    'select_features_pipeline',
    'calculate_feature_stability',

    # 种子扫描（已下线，不对外暴露）
    # 'sweep_random_states',
    # 'select_best_seed_for_features',
    # 'find_best_seed_for_training',

    # 模型
    'get_best_gbdt_model',
    'train_with_early_stopping',
    'evaluate_regression_model',
    'check_gpu_availability',
    'get_feature_importance',

    # 工具函数
    'bin_by_mic',
    'compute_scaffold',
    'ensure_dir'
]
