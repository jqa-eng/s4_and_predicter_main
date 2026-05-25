#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ecoli_paperalign.py - E.Coli MIC回归器训练
核心策略（确定性特征选择版）：
1. SMILES随机化增强（分层：excellent×4, good×4, moderate×2, poor×0）
2. 样本重加权（MIC桶权重 + MolLogP漂移校正）
3. 两阶段确定性特征选择（Stage1过滤 + Stage2 SelectKBest(f_regression)预筛160 → RFE(RandomForest)精选到n_select）
4. 特征锁定机制（首次选择后永久锁定，后续训练直接读取不再重选）
5. GBDT优先（XGBoost → LightGBM → HGBR）
6. 单模型训练（全量训练集，无CV集成）
7. 目标y的z-score标准化（改善树模型幅度学习）
8. 闸门验收（Val R²>-0.05, Test R²≥0.0 [relaxed模式]）

训练流程：
  Phase 1: 数据增强 + 特征选择（或读取锁定特征）
  Phase 2: 最终模型训练 + 验收
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

# 导入ecoli_micmods模块
from ecoli_micmods.data_io import (
    load_csv_with_logmic,
    compute_sample_weights_stratified,
    diagnose_distribution_shift
)
from ecoli_micmods.smiles_augment import augment_dataset_stratified
from ecoli_micmods.feat_select import select_features_pipeline
from ecoli_micmods.models import (
    get_best_gbdt_model,
    train_with_early_stopping,
    evaluate_regression_model,
    check_gpu_availability
)
from ecoli_micmods.utils import ensure_dir


def plot_learning_curve_generic(train_history, val_history, output_dir, metric='RMSE'):
    """绘制通用学习曲线（兼容不同模型）"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(train_history, label=f'Train {metric}', alpha=0.7)
    ax.plot(val_history, label=f'Val {metric}', alpha=0.7, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{metric} (logMIC)')
    ax.set_title(f'Learning Curve ({metric})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', 'learning_curve.png')
    ensure_dir(os.path.dirname(plot_path))
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  学习曲线已保存: {plot_path}")


def train_ecoli_paperalign(args):
    """论文对齐版E.Coli训练主流程"""

    print("=" * 80)
    print("E.Coli MIC 回归器训练（论文对齐版）")
    print("=" * 80)
    print(f"训练集: {args.train}")
    print(f"验证集: {args.val}")
    print(f"测试集: {args.test}")
    print(f"任务: {args.task}")
    print(f"SMILES增强: {'是' if args.augment else '否'} (分层策略)")
    print(f"样本重加权: {'是' if args.reweight else '否'} (MIC桶+MolLogP)")
    print(f"目标特征数: {args.n_features}")
    print(f"最大迭代数: {args.num_boost_round}")
    print(f"模型优先级: {args.model_prefer}")

    # ==================== Phase 1: 数据准备 ====================
    print("\n" + "=" * 80)
    print("Phase 1: 数据加载与诊断")
    print("=" * 80)

    # 1.1 加载数据
    df_train = load_csv_with_logmic(args.train, task=args.task)
    df_val = load_csv_with_logmic(args.val, task=args.task)
    df_test = load_csv_with_logmic(args.test, task=args.task)

    print(f"  训练集: {df_train.shape}")
    print(f"  验证集: {df_val.shape}")
    print(f"  测试集: {df_test.shape}")

    # 1.2 分布诊断
    diagnose_distribution_shift(df_train, df_val, df_test, task=args.task)

    # ==================== Phase 2: SMILES增强 ====================
    if args.augment:
        print("\n" + "=" * 80)
        print("Phase 2: SMILES随机化增强（分层策略）")
        print("=" * 80)

        df_train = augment_dataset_stratified(
            df_train,
            task=args.task,
            smiles_col='smiles',
            feat_start=args.feat_start,
            feat_count=args.feat_count,
            augment_factors={'excellent': 4, 'good': 4, 'moderate': 2, 'poor': 0},
            random_seed=42
        )
        print(f"  增强后训练集: {df_train.shape}")
    else:
        print("\n跳过SMILES增强（--no-augment）")

    # ==================== Phase 3: 特征与标签提取（黑名单 + 列名对齐） ====================
    print("\n" + "=" * 80)
    print("Phase 3: 特征与标签提取（黑名单过滤 + 列名对齐）")
    print("=" * 80)

    target_col = f'{args.task}_logMIC'
    BLACKLIST = {
        f'{args.task}_MIC', f'{args.task}_logMIC',
        'pos', 'n', 'smiles', 'split', 'fold', 'bucket', 'weight', 'augment'
    }

    all_cols_train = df_train.columns.tolist()
    # 1) 先从 train 的特征窗内拿"列名候选"
    feature_window_names = [c for c in all_cols_train[args.feat_start: args.feat_start + args.feat_count]]
    candidate_feat_names = [c for c in feature_window_names if c not in BLACKLIST]

    # 2) 与 val/test 求交集，保证三份表都有
    common_feat_names = [c for c in candidate_feat_names if c in df_val.columns and c in df_test.columns]

    # 3) 诊断：如果数量对不上，给出提示
    missing_in_val  = [c for c in candidate_feat_names if c not in df_val.columns]
    missing_in_test = [c for c in candidate_feat_names if c not in df_test.columns]
    print(f"  训练候选特征数: {len(candidate_feat_names)}  -> 三集共同特征数: {len(common_feat_names)}")
    if missing_in_val:
        print(f"  [WARN] 有 {len(missing_in_val)} 列在 Val 缺失（示例）: {missing_in_val[:5]}")
    if missing_in_test:
        print(f"  [WARN] 有 {len(missing_in_test)} 列在 Test 缺失（示例）: {missing_in_test[:5]}")

    assert len(common_feat_names) > 0, "共同特征为空，请检查三份CSV的特征生成与列名一致性！"

    # 4) 用"列名"取矩阵（顺序由 common_feat_names 明确控制）
    X_train = df_train[common_feat_names].values
    y_train = df_train[target_col].values

    X_val = df_val[common_feat_names].values
    y_val = df_val[target_col].values

    X_test = df_test[common_feat_names].values
    y_test = df_test[target_col].values

    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}, 范围: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # ==================== Phase 3.5: 目标y的z-score标准化（GPT-5 v4：改善幅度学习） ====================
    print("\n" + "=" * 80)
    print("Phase 3.5: 目标y标准化（z-score）")
    print("=" * 80)

    # 计算训练集的均值和标准差
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 1e-8 else 1.0

    # 对所有集合做z-score标准化
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    print(f"  原始 y_train: 均值={y_mean:.3f}, 标准差={y_std:.3f}")
    print(f"  标准化后 y_train_z: 均值={y_train_z.mean():.3f}, 标准差={y_train_z.std():.3f}")
    print(f"  GPT-5 v4: 树模型在标准化目标上更容易学习幅度")

    # 保存用于训练的目标（z-score）
    y_train_fit = y_train_z
    y_val_fit = y_val_z
    y_test_fit = y_test_z

    # 定义反变换函数（推理时用）
    def inv_z_score(pred_z):
        """将z-score预测值反变换回原始尺度"""
        return pred_z * y_std + y_mean

    # ==================== Phase 4: 样本重加权 ====================
    if args.reweight:
        print("\n" + "=" * 80)
        print("Phase 4: 样本重加权（MIC桶 + MolLogP漂移校正）")
        print("=" * 80)

        sample_weights = compute_sample_weights_stratified(
            df_train, df_val,
            task=args.task,
            mic_桶_weights={'excellent': 30, 'good': 8, 'moderate': 3, 'poor': 1},
            mologp_correction=True,
            mologp_bandwidth=0.5
        )
    else:
        print("\n跳过样本重加权（--no-reweight）")
        sample_weights = None

    # ==================== Phase 4.5: 特征锁定检测 ====================
    print("\n" + "=" * 80)
    print("Phase 4.5: 检测已锁定的特征规范")
    print("=" * 80)

    output_dir = args.output_dir if args.output_dir else f"models_predicter/{args.task}_predictor_paperalign"
    feature_spec_path = os.path.join(output_dir, 'feature_spec.json')

    feature_locked = False
    locked_feature_names = None

    if os.path.exists(feature_spec_path):
        print(f"  发现特征规范文件: {feature_spec_path}")
        try:
            with open(feature_spec_path, 'r', encoding='utf-8') as f:
                existing_spec = json.load(f)

            if existing_spec.get('locked', False):
                feature_locked = True
                locked_feature_names = existing_spec.get('selected_names', [])
                print(f"  [LOCKED] 特征已锁定！")
                print(f"  锁定特征数: {len(locked_feature_names)}")
                print(f"  前5个特征: {locked_feature_names[:5]}")
                print(f"  将跳过特征选择，直接使用锁定特征")
            else:
                print(f"  特征规范存在但未锁定，将重新进行特征选择")
        except Exception as e:
            print(f"  警告: 读取特征规范失败: {e}")
            print(f"  将重新进行特征选择")
    else:
        print(f"  未发现特征规范文件，将进行首次特征选择")

    # ==================== Phase 5: 两阶段特征选择（或使用锁定特征） ====================
    print("\n" + "=" * 80)
    print("Phase 5: 特征选择")
    print("=" * 80)

    if feature_locked:
        # 使用锁定的特征
        print(f"  使用已锁定的{len(locked_feature_names)}个特征")

        # 验证特征列是否存在
        missing_features = [f for f in locked_feature_names if f not in common_feat_names]
        if missing_features:
            raise ValueError(f"锁定的特征在数据中不存在: {missing_features[:5]}")

        # 获取特征索引
        final_idx = [common_feat_names.index(f) for f in locked_feature_names]

        # 构造特征字典
        X_dict = {
            'train': X_train[:, final_idx],
            'val': X_val[:, final_idx],
            'test': X_test[:, final_idx]
        }

        rfecv = None  # 不需要selector对象
        print(f"  特征提取完成: {X_train.shape[1]} -> {len(final_idx)}")

    else:
        # 执行特征选择
        print(f"  执行两阶段特征选择（Stage1过滤 + Stage2 RFE）")

        X_dict, final_idx, rfecv = select_features_pipeline(
            X_train, y_train_z,  # 使用z-score的y
            X_val=X_val, X_test=X_test,
            n_select=args.n_features,
            random_state=42,  # 固定种子确保可重复
            sample_weight=sample_weights
        )

        print(f"  特征选择完成，使用的random_state=42（固定）")

    # 最终特征数和样本/特征比
    print(f"  样本/特征比: {X_dict['train'].shape[0]}/{len(final_idx)} = {X_dict['train'].shape[0]/len(final_idx):.2f}")

    # 断言: final_idx是相对common_feat_names的索引
    assert len(final_idx) == X_dict['train'].shape[1], \
        f"final_idx长度({len(final_idx)})与选择后特征数({X_dict['train'].shape[1]})不一致"
    print(f"  [验证通过] final_idx映射: {len(final_idx)}个索引 → {X_dict['train'].shape[1]}维特征")

    # ==================== Phase 5.5: 特征标准化（GPT-5建议：树模型跳过） ====================
    print("\n" + "=" * 80)
    print("Phase 5.5: 特征标准化判断（树模型跳过，线性模型使用StandardScaler）")
    print("=" * 80)

    # 6.1 提前获取模型名称（用于判断是否需要标准化）
    model, model_name, supports_weight = get_best_gbdt_model(
        prefer=args.model_prefer,
        task='regression',
        random_state=42,
        n_estimators=args.num_boost_round
    )

    # GPT-5建议：树模型（XGBoost/LightGBM/HGBR）不需要标准化
    if model_name in {'xgboost', 'lightgbm', 'hgbr'}:
        print(f"  检测到树模型({model_name})，跳过标准化（GPT-5建议：避免对树分割阈值的扰动）")
        X_train_sel = X_dict['train']
        X_val_sel = X_dict['val']
        X_test_sel = X_dict['test']
        scaler = None  # 不使用标准化
    else:
        print(f"  检测到非树模型({model_name})，使用StandardScaler标准化")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_sel = scaler.fit_transform(X_dict['train'])
        X_val_sel = scaler.transform(X_dict['val'])
        X_test_sel = scaler.transform(X_dict['test'])

        print(f"  标准化完成")
        print(f"  均值范围: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
        print(f"  标准差范围: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

    # 特征维度验证（对齐目标维度）
    expected_feats = args.n_features
    print(f"\n  [{expected_feats}维特征验证]")
    print(f"    X_train_sel.shape: {X_train_sel.shape}")
    print(f"    X_val_sel.shape: {X_val_sel.shape}")
    print(f"    X_test_sel.shape: {X_test_sel.shape}")

    # 验证是否都符合目标维度
    all_ok = (X_train_sel.shape[1] == expected_feats and
              X_val_sel.shape[1] == expected_feats and
              X_test_sel.shape[1] == expected_feats)

    if all_ok:
        print(f"    特征维度验证: [PASS] - 所有数据集均为{expected_feats}维特征")
    else:
        print(f"    特征维度验证: [FAIL] - 特征维度不一致")
        print(f"      期望: (*, {expected_feats}), 实际: Train={X_train_sel.shape[1]}, Val={X_val_sel.shape[1]}, Test={X_test_sel.shape[1]}")

    # ==================== Phase 6: 模型训练 ====================
    print("\n" + "=" * 80)
    print("Phase 6: GBDT模型训练（全量训练集）")
    print("=" * 80)

    # 模型已在Phase 5.5获取，直接使用
    print(f"  使用已选择的模型: {model_name}")

    # 6.2 训练（带早停）
    print(f"\n  开始训练{model_name}模型...")

    # 使用z-score标准化的目标进行训练
    model, best_iter = train_with_early_stopping(
        model, model_name,
        X_train_sel, y_train_fit,  # 使用z-score目标
        X_val_sel, y_val_fit,      # 使用z-score目标
        sample_weight=sample_weights,
        early_stopping_rounds=args.early_stopping_rounds
    )

    print(f"  训练完成: best_iteration={best_iter}")

    # ==================== Phase 7: 模型评估 ====================
    print("\n" + "=" * 80)
    print("Phase 7: 模型评估（Val + Test）")
    print("=" * 80)

    # 7.0 Baseline RMSE（预测训练集均值）
    yhat_mean_val = np.full_like(y_val, y_train.mean())
    baseline_rmse_val = np.sqrt(((y_val - yhat_mean_val)**2).mean())
    yhat_mean_test = np.full_like(y_test, y_train.mean())
    baseline_rmse_test = np.sqrt(((y_test - yhat_mean_test)**2).mean())
    print(f"Baseline RMSE (predict train-mean):")
    print(f"  Val: {baseline_rmse_val:.3f}")
    print(f"  Test: {baseline_rmse_test:.3f}")
    print(f"  (模型RMSE应显著低于此值，说明真正学到了信号)\n")

    # 7.1 获取z-score预测并反变换到原始尺度
    from ecoli_micmods.models import predict_with_best_iteration

    y_pred_train_z = predict_with_best_iteration(model, model_name, X_train_sel)
    y_pred_val_z = predict_with_best_iteration(model, model_name, X_val_sel)
    y_pred_test_z = predict_with_best_iteration(model, model_name, X_test_sel)

    # 反变换到原始尺度
    y_pred_train = inv_z_score(y_pred_train_z)
    y_pred_val = inv_z_score(y_pred_val_z)
    y_pred_test = inv_z_score(y_pred_test_z)

    # 7.2 幅度诊断（斜率/截距）+ 线性校准参数计算
    from sklearn.linear_model import LinearRegression

    def slope_intercept(y_true, y_pred):
        lr = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
        return float(lr.coef_[0]), float(lr.intercept_)

    s_train, b_train = slope_intercept(y_train, y_pred_train)
    s_val, b_val = slope_intercept(y_val, y_pred_val)
    s_test, b_test = slope_intercept(y_test, y_pred_test)

    print(f"\n[GPT-5 v4幅度诊断] Train: slope={s_train:.3f}, intercept={b_train:.3f}")
    print(f"[GPT-5 v4幅度诊断]  Val : slope={s_val:.3f}, intercept={b_val:.3f}")
    print(f"[GPT-5 v4幅度诊断] Test: slope={s_test:.3f}, intercept={b_test:.3f}")
    print(f"  (目标: slope≈1, intercept≈0 说明幅度学习正常)\n")

    # 7.2.5 线性校准参数（使用验证集）- 用于推理时校准
    # 根据guidance.md，使用验证集的斜率和截距作为校准参数
    a_val = s_val  # 校准系数（斜率）
    b_val_calibration = b_val  # 校准截距
    print(f"[线性校准参数] a_val={a_val:.4f}, b_val={b_val_calibration:.4f}")
    print(f"  推理时将应用: y_calibrated = a_val * y_pred + b_val")
    print(f"  这些参数将保存到meta.json供推理使用\n")

    # 7.3 手动计算指标（使用原始尺度的预测值）
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import pearsonr

    # 训练集
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_pearson, _ = pearsonr(y_train, y_pred_train)
    print(f"Train R2={train_r2:.3f}, RMSE={train_rmse:.3f}, Pearson={train_pearson:.3f}")

    # 验证集
    val_r2 = r2_score(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_pearson, _ = pearsonr(y_val, y_pred_val)
    print(f"Val R2={val_r2:.3f}, RMSE={val_rmse:.3f}, Pearson={val_pearson:.3f}")

    val_metrics = {'r2': val_r2, 'rmse': val_rmse, 'pearson': val_pearson}

    # 测试集
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_pearson, _ = pearsonr(y_test, y_pred_test)
    print(f"Test R2={test_r2:.3f}, RMSE={test_rmse:.3f}, Pearson={test_pearson:.3f}")

    test_metrics = {'r2': test_r2, 'rmse': test_rmse, 'pearson': test_pearson}

    # 7.3 闸门检查
    print(f"\n  闸门检查（{args.gate}）:")

    if args.gate == 'strict':
        val_pass = (val_metrics['r2'] > 0.0) and (val_metrics['pearson'] > 0.35)
        test_pass = (test_metrics['r2'] >= 0.20) and (test_metrics['pearson'] >= 0.50)
        print(f"    Val R2 > 0 且 Pearson > 0.35: {'通过' if val_pass else '失败'}")
        print(f"      实际值: R2={val_metrics['r2']:.3f}, Pearson={val_metrics['pearson']:.3f}")
        print(f"    Test R2 >= 0.20 且 Pearson >= 0.50: {'通过' if test_pass else '失败'}")
        print(f"      实际值: R2={test_metrics['r2']:.3f}, Pearson={test_metrics['pearson']:.3f}")
    elif args.gate == 'relaxed':
        val_pass = (val_metrics['r2'] > -0.05)  # 允许轻微负漂移
        test_pass = (test_metrics['r2'] >= 0.0)  # 关键：Test R²不为负
        print(f"    Val R2 > -0.05: {'通过' if val_pass else '失败'}")
        print(f"      实际值: R2={val_metrics['r2']:.3f}")
        print(f"    Test R2 >= 0.0: {'通过' if test_pass else '失败'}")
        print(f"      实际值: R2={test_metrics['r2']:.3f}")
    else:  # none
        val_pass = test_pass = True
        print(f"    闸门检查已禁用")

    gate_status = 'PASS' if (val_pass and test_pass) else 'FAIL'
    print(f"\n  最终闸门: {gate_status}")

    # ==================== Phase 7.5: 绘制学习曲线（Train/Val 二曲线，不使用Test避免数据泄漏） ====================
    print("\n" + "=" * 80)
    print("Phase 7.5: 绘制学习曲线（Train/Val，不包含Test以避免数据泄漏）")
    print("=" * 80)

    output_dir = args.output_dir if args.output_dir else f"models_predicter/{args.task}_predictor_paperalign"
    ensure_dir(os.path.join(output_dir, "plots"))

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def compute_rmse_histories(model, model_name,
                               X_tr, y_tr_fit,
                               X_va, y_va_fit,
                               y_std,
                               best_iter, stride=1):
        """逐迭代计算 z-score 空间的 RMSE，再乘 y_std 还原到原始 logMIC 尺度。

        注意：不包含测试集以避免数据泄漏！测试集仅在最终评估时使用。
        """
        train_hist, val_hist = [], []

        if model_name == 'xgboost':
            # sklearn API 支持 iteration_range
            # 注意：我们对 y 的训练目标是 z-score，因此预测也是 z-space
            for t in range(1, best_iter + 1, stride):
                y_tr_pred_z = model.predict(X_tr, iteration_range=(0, t))
                y_va_pred_z = model.predict(X_va, iteration_range=(0, t))

                # z-space RMSE -> 乘 y_std -> 原始 logMIC 尺度
                train_hist.append(_rmse(y_tr_fit, y_tr_pred_z) * y_std)
                val_hist.append(_rmse(y_va_fit, y_va_pred_z) * y_std)

        elif model_name == 'lightgbm':
            booster = model.booster_
            for t in range(1, best_iter + 1, stride):
                y_tr_pred_z = booster.predict(X_tr, num_iteration=t)
                y_va_pred_z = booster.predict(X_va, num_iteration=t)

                train_hist.append(_rmse(y_tr_fit, y_tr_pred_z) * y_std)
                val_hist.append(_rmse(y_va_fit, y_va_pred_z) * y_std)
        else:
            print("  HGBR 不支持逐迭代曲线，跳过")
            return [], []

        return train_hist, val_hist

    # 选择"步长"防止循环过慢：最多取 400 个点
    best_iter_eff = int(best_iter) if best_iter is not None else args.num_boost_round
    stride = max(1, best_iter_eff // 400)

    print(f"  逐迭代评估：best_iter={best_iter_eff}, stride={stride}")

    try:
        train_hist, val_hist = compute_rmse_histories(
            model, model_name,
            X_train_sel, y_train_fit,
            X_val_sel,   y_val_fit,
            y_std,
            best_iter=best_iter_eff,
            stride=stride
        )

        if len(val_hist) > 0:
            # 画二条曲线（原始 logMIC 尺度）- 不包含测试集
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            xs = list(range(1, best_iter_eff + 1, stride))
            ax.plot(xs, train_hist, label='Train RMSE', alpha=0.8)
            ax.plot(xs, val_hist,   label='Val RMSE',   alpha=0.9, linewidth=2)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('RMSE (logMIC)')
            ax.set_title('Learning Curves: Train / Val (Test excluded to avoid data leakage)')
            ax.grid(alpha=0.3)
            ax.legend()
            plt.tight_layout()

            out_png = os.path.join(output_dir, 'plots', 'learning_curve_train_val.png')
            plt.savefig(out_png, dpi=150)
            plt.close()

            print(f"  学习曲线已保存: {out_png}")
            print(f"    末尾值(原始尺度): Train={train_hist[-1]:.3f}, Val={val_hist[-1]:.3f}")
            print(f"    [数据泄漏修复] 测试集已从学习曲线中移除")
        else:
            print("  跳过：无有效历史记录（或模型不支持）")
            train_hist, val_hist = [], []
    except Exception as e:
        print(f"  学习曲线绘制失败: {e}")
        import traceback
        traceback.print_exc()
        train_hist, val_hist = [], []

    # ==================== Phase 8: 保存模型和元数据 ====================
    print("\n" + "=" * 80)
    print("Phase 8: 保存模型和元数据")
    print("=" * 80)

    output_dir = args.output_dir if args.output_dir else f"models_predicter/{args.task}_predictor_paperalign"
    ensure_dir(output_dir)

    # 8.1 保存模型
    if model_name == 'lightgbm':
        model_path = os.path.join(output_dir, 'model.txt')
        model.booster_.save_model(model_path)
    elif model_name == 'xgboost':
        model_path = os.path.join(output_dir, 'model.json')
        model.save_model(model_path)
    else:  # HGBR
        import joblib
        model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, model_path)

    print(f"  模型已保存: {model_path}")

    # 8.2 保存feature_spec.json (修复final_idx映射)
    # final_idx是相对common_feat_names的索引,需映射回列名
    try:
        selected_feat_names = [common_feat_names[i] for i in final_idx]
        print(f"  [列名映射验证] 前3个: {selected_feat_names[:3]}")
    except IndexError as e:
        print(f"  [ERROR] final_idx映射越界: {e}")
        print(f"    common_feat_names长度: {len(common_feat_names)}")
        print(f"    final_idx范围: [{min(final_idx)}, {max(final_idx)}]")
        raise

    feature_spec = {
        'locked': True,  # 特征选择后永久锁定，后续训练直接读取不再重选
        'selector': {
            'type': 'TwoStage_SelectKBest_RFE',  # 实际使用SelectKBest预筛 + RFE精选
            'stage1': {
                'var_thresh_ratio': 0.10,
                'spearman_th': 0.98,
                'target_corr_min': 0.0,
                'top_k_corr': 256
            },
            'stage2': {
                'method': 'SelectKBest(f_regression) -> RFE(RandomForest)',
                'prefilter_k': 160,
                'n_select': len(final_idx),
                'rfe_estimator': 'RandomForest(n_estimators=200, max_depth=12, bootstrap=False)',
                'random_state': 42
            },
            'topk': len(final_idx)
        },
        'scaler': {
            'type': 'None' if scaler is None else 'StandardScaler',
            'mean': scaler.mean_.tolist() if scaler is not None else [],
            'scale': scaler.scale_.tolist() if scaler is not None else []
        },
        'selected_indices': final_idx,
        'selected_names': selected_feat_names,
        'task': args.task,
        'feat_start': args.feat_start,
        'feat_count': args.feat_count
    }

    spec_path = os.path.join(output_dir, 'feature_spec.json')
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(feature_spec, f, indent=2, ensure_ascii=False)
    print(f"  特征规范已保存: {spec_path}")

    # 8.3 保存meta.json
    meta = {
        'task': args.task,
        'strategy': 'paper_aligned',
        'model_type': model_name,
        'target_normalization': {  # GPT-5 v4新增：目标标准化参数
            'enabled': True,
            'method': 'z-score',
            'y_mean': float(y_mean),
            'y_std': float(y_std)
        },
        'calibration': {  # 新增：线性校准参数（用于推理）
            'enabled': True,
            'method': 'linear_regression_on_validation',
            'a_val': float(a_val),  # 校准系数（验证集斜率）
            'b_val': float(b_val_calibration),  # 校准截距（验证集截距）
            'description': 'Apply as: y_final = a_val * y_pred_logMIC + b_val after inverse z-transform'
        },
        'augmentation': {
            'enabled': args.augment,
            'strategy': 'stratified',
            'factors': {'excellent': 4, 'good': 4, 'moderate': 2, 'poor': 0}
        },
        'reweighting': {
            'enabled': args.reweight,
            'mic_bucket_weights': {'excellent': 30, 'good': 8, 'moderate': 3, 'poor': 1},
            'mologp_correction': True
        },
        'feature_selection': {
            'method': 'TwoStage_SelectKBest_RFE',
            'final_n_features': len(final_idx),
            'stage1': 'zero_variance + spearman_correlation_filter + collinearity_removal',
            'stage2': 'SelectKBest(f_regression, k=160) -> RFE(RandomForest, n_select=24)',
            'random_state': 42,
            'locked': True  # 特征选择后永久锁定（与feature_spec.json一致）
        },
        'training': {
            'n_estimators': args.num_boost_round,
            'best_iteration': int(best_iter) if best_iter is not None else None,
            'early_stopping_rounds': args.early_stopping_rounds,
            'history': {
                'train_rmse': train_hist if isinstance(train_hist, list) else [],
                'val_rmse': val_hist if isinstance(val_hist, list) else []
                # 注意：已移除test_rmse以避免数据泄漏
            }
        },
        'val_metrics': {
            'r2': float(val_metrics['r2']),
            'rmse': float(val_metrics['rmse']),
            'pearson': float(val_metrics['pearson'])
        },
        'test_metrics': {
            'r2': float(test_metrics['r2']),
            'rmse': float(test_metrics['rmse']),
            'pearson': float(test_metrics['pearson'])
        },
        'amplitude_diagnosis': {  # GPT-5 v4新增：幅度诊断
            'train': {'slope': s_train, 'intercept': b_train},
            'val': {'slope': s_val, 'intercept': b_val},
            'test': {'slope': s_test, 'intercept': b_test}
        },
        'gate_status': gate_status,
        'gate_config': {
            'strategy': args.gate,
            'val_pass': bool(val_pass),
            'test_pass': bool(test_pass)
        },
        'gate_criteria': {
            'strategy_description': f'{args.gate} strategy',
            'val_criteria': f'Val R2 > {-0.05 if args.gate == "relaxed" else 0.0}' + (' and Pearson > 0.35' if args.gate == 'strict' else ''),
            'test_criteria': f'Test R2 >= {0.0 if args.gate == "relaxed" else 0.20}' + (' and Pearson >= 0.50' if args.gate == 'strict' else ''),
            'val_actual_values': f'R2={val_metrics["r2"]:.3f}, Pearson={val_metrics["pearson"]:.3f}',
            'test_actual_values': f'R2={test_metrics["r2"]:.3f}, Pearson={test_metrics["pearson"]:.3f}'
        },
        'data': {
            'n_train': int(X_train_sel.shape[0]),
            'n_val': int(X_val_sel.shape[0]),
            'n_test': int(X_test_sel.shape[0])
        }
    }

    meta_path = os.path.join(output_dir, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  元数据已保存: {meta_path}")

    # ==================== 完成 ====================
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"模型目录: {output_dir}")
    print(f"  - model: {model_name}模型文件")
    print(f"  - feature_spec.json: 特征选择规范")
    print(f"  - meta.json: 训练元数据与指标")
    print(f"\n闸门状态: {gate_status}")

    if gate_status == 'FAIL':
        print("\n建议: 模型未通过闸门，建议:")
        print("  1. 检查数据质量与分布")
        print("  2. 调整增强/重加权参数")
        print("  3. 考虑暂停E.Coli模块，专注Aureus+Toxicity")

    return 0 if gate_status == 'PASS' else 1


# ============ CLI 入口 ============

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="E.Coli MIC 回归器训练（论文对齐版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--train', required=True, help='训练集 CSV 路径')
    parser.add_argument('--val', required=True, help='验证集 CSV 路径')
    parser.add_argument('--test', required=True, help='测试集 CSV 路径')
    parser.add_argument('--task', default='ecoli', help='任务名（默认 ecoli）')
    parser.add_argument('--feat-start', type=int, default=3, help='特征起始列索引（默认5）')
    parser.add_argument('--feat-count', type=int, default=4274, help='特征总数（默认4274）')
    parser.add_argument('--n-features', type=int, default=24,
                        help='目标特征数（默认24维）')
    parser.add_argument('--num-boost-round', type=int, default=3000,
                        help='最大迭代次数（GPT-5 v4: 默认3000，观察早停点）')
    parser.add_argument('--early-stopping-rounds', type=int, default=200,
                        help='早停轮数（GPT-5 v4: 默认200，快速暴露问题）')
    parser.add_argument('--model-prefer', default='xgboost',
                        choices=['lightgbm', 'hgbr', 'xgboost'],
                        help='模型优先级（默认xgboost）')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='启用SMILES增强（默认关闭）')
    parser.add_argument('--reweight', action='store_true', default=False,
                        help='启用样本重加权（默认关闭）')
    parser.add_argument('--gate', choices=['strict', 'relaxed', 'none'], default='relaxed',
                        help='闸门策略（strict=严格, relaxed=宽松[默认], none=不检查）')
    parser.add_argument('--output-dir', default=None,
                        help='模型输出目录（默认自动生成）')

    return parser.parse_args()


def main():
    args = parse_arguments()
    return train_ecoli_paperalign(args)


if __name__ == "__main__":
    sys.exit(main())
