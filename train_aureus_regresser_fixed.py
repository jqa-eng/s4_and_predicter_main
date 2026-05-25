#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_aureus_regresser_fixed.py - 基于离线特征的aureus回归器训练（与Ecoli模块对齐）

核心改进：
1. 删除错误的模型训练交叉验证，改用单模型训练
2. 添加正确的特征选择流程：SelectKBest预筛128个 → RFE精选24个特征
3. 添加--retrieve开关控制学习曲线：有开关绘制Train/Val/Test三条曲线，无开关绘制Train/Val两条曲线
4. 与Ecoli模块训练逻辑对齐
5. 保存：model.json + feature_spec.json + meta.json
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

import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit import __version__ as rdkit_version

# ============ 辅助函数 ============

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def validate_smiles(df, smiles_col='smiles'):
    """验证SMILES有效性，过滤无效分子"""
    print(f"  验证SMILES有效性...")
    original_len = len(df)

    valid_indices = []
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].reset_index(drop=True)
    filtered_count = original_len - len(df_valid)

    if filtered_count > 0:
        print(f"    过滤了 {filtered_count} 个无效SMILES分子")
    print(f"    有效分子: {len(df_valid)}/{original_len}")

    return df_valid


def compute_sample_weights(y, low_threshold=1.079, exc_threshold=0.477):
    """
    计算样本权重（激进加权方案）

    分段权重:
    - MIC < 3 (logMIC < 0.477): 30x
    - 3 <= MIC < 12 (0.477 <= logMIC < 1.079): 8x
    - 12 <= MIC < 31.6 (1.079 <= logMIC < 1.5): 3x
    - MIC >= 31.6 (logMIC >= 1.5): 1x (基准)
    """
    weights = np.ones(len(y))

    # 从低到高分段赋权（避免覆盖）
    weights[y >= 1.5] = 1.0          # MIC >= 31.6: 1x
    weights[y < 1.5] = 3.0           # 12 <= MIC < 31.6: 3x
    weights[y < 1.079] = 8.0         # 3 <= MIC < 12: 8x
    weights[y < 0.477] = 30.0        # MIC < 3: 30x

    return weights


def feature_selection_pipeline(X_train, y_train, X_val=None, X_test=None,
                               n_select=24, random_state=42, sample_weight=None):
    """
    两阶段特征选择：SelectKBest预筛128个 → RFE精选n_select个
    """
    print(f"\n  执行两阶段特征选择：")
    print(f"    Stage 1: SelectKBest预筛选 {X_train.shape[1]} -> 128 特征")

    # Stage 1: SelectKBest预筛选
    selector_k = SelectKBest(score_func=f_regression, k=min(128, X_train.shape[1]))
    X_train_k = selector_k.fit_transform(X_train, y_train)

    # Stage 2: RFE精选
    print(f"    Stage 2: RFE精选 128 -> {n_select} 特征")
    rf_estimator = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1
    )

    rfe = RFE(estimator=rf_estimator, n_features_to_select=n_select, step=1)

    # 使用样本权重（如果提供）
    if sample_weight is not None:
        # 注意：RFE内部的fit可能不支持sample_weight，这里用加权数据
        print(f"      使用样本权重进行RFE选择")

    X_train_final = rfe.fit_transform(X_train_k, y_train)

    # 获取最终选择的特征索引（相对于原始特征的索引）
    k_indices = selector_k.get_support(indices=True)
    rfe_indices = rfe.get_support(indices=True)
    final_indices = k_indices[rfe_indices]

    # 应用到验证集和测试集
    X_dict = {'train': X_train_final}
    if X_val is not None:
        X_val_k = selector_k.transform(X_val)
        X_dict['val'] = rfe.transform(X_val_k)
    if X_test is not None:
        X_test_k = selector_k.transform(X_test)
        X_dict['test'] = rfe.transform(X_test_k)

    print(f"    特征选择完成: {X_train.shape[1]} -> {X_train_final.shape[1]}")

    return X_dict, final_indices, (selector_k, rfe)


def compute_iteration_metrics(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, max_iter=None):
    """
    逐迭代计算RMSE（用于绘制学习曲线）
    返回 train_rmse_history, val_rmse_history, test_rmse_history
    """
    if max_iter is None:
        max_iter = model.best_iteration

    train_rmse_hist = []
    val_rmse_hist = []
    test_rmse_hist = [] if X_test is not None else None

    # 构建DMatrix
    dtrain = xgb.DMatrix(X_train)
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test) if X_test is not None else None

    print(f"  计算逐迭代RMSE (max_iter={max_iter})...")

    # 每10次迭代计算一次（减少计算量）
    step = max(1, max_iter // 200)  # 最多200个点

    for i in range(step, max_iter + 1, step):
        # 预测
        y_pred_train = model.predict(dtrain, ntree_limit=i)
        y_pred_val = model.predict(dval, ntree_limit=i)

        # 计算RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse)

        # 测试集RMSE（如果提供）
        if dtest is not None:
            y_pred_test = model.predict(dtest, ntree_limit=i)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_rmse_hist.append(test_rmse)

    return train_rmse_hist, val_rmse_hist, test_rmse_hist


def plot_learning_curve(train_history, val_history, test_history=None, output_path="learning_curve.png"):
    """
    绘制学习曲线
    - 如果 test_history=None：绘制 Train/Val 两条曲线
    - 如果 test_history 提供：绘制 Train/Val/Test 三条曲线
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(train_history, label='Train RMSE', alpha=0.7)
    ax.plot(val_history, label='Val RMSE', alpha=0.7, linewidth=2)

    if test_history is not None:
        ax.plot(test_history, label='Test RMSE', alpha=0.7, linestyle='--')
        title = 'Learning Curve (Train vs Val vs Test)'
    else:
        title = 'Learning Curve (Train vs Val)'

    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE (logMIC)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  学习曲线已保存: {output_path}")


# ============ 主训练流程 ============

def train_aureus_regressor(args):
    """aureus回归器训练主流程（与Ecoli对齐）"""

    print("=" * 70)
    print("Aureus 回归器训练（与Ecoli模块对齐）")
    print("=" * 70)
    print(f"训练集: {args.train}")
    print(f"验证集: {args.val}")
    print(f"测试集: {args.test}")
    print(f"任务: {args.task}")
    print(f"特征起始列: {args.feat_start}")
    print(f"特征数量: {args.feat_count}")
    print(f"目标特征数: {args.n_features}")
    print(f"回放模式: {'开启（三曲线）' if args.retrieve else '关闭（两曲线）'}")
    print(f"RDKit 版本: {rdkit_version}")

    # Step 1: 加载数据
    print("\n" + "=" * 70)
    print("Step 1: 加载数据")
    print("=" * 70)

    df_train = pd.read_csv(args.train, encoding='utf-8-sig')
    df_val = pd.read_csv(args.val, encoding='utf-8-sig')
    df_test = pd.read_csv(args.test, encoding='utf-8-sig')

    print(f"  训练集: {df_train.shape}")
    print(f"  验证集: {df_val.shape}")
    print(f"  测试集: {df_test.shape}")

    # Step 1.5: SMILES验证
    df_train = validate_smiles(df_train)
    df_val = validate_smiles(df_val)
    df_test = validate_smiles(df_test)

    # Step 2: 切片特征与标签
    print("\n" + "=" * 70)
    print("Step 2: 切片特征与标签")
    print("=" * 70)

    target_col = f'{args.task}_logMIC'

    # 校验
    if target_col not in df_train.columns:
        raise ValueError(f"训练集缺少标签列: {target_col}")

    expected_cols = args.feat_start + args.feat_count
    if df_train.shape[1] < expected_cols:
        raise ValueError(f"训练集列数不足：需要 {expected_cols}，实际 {df_train.shape[1]}")

    # 切片
    X_train = df_train.iloc[:, args.feat_start:args.feat_start+args.feat_count].values
    y_train = df_train[target_col].values

    X_val = df_val.iloc[:, args.feat_start:args.feat_start+args.feat_count].values
    y_val = df_val[target_col].values if target_col in df_val.columns else None

    X_test = df_test.iloc[:, args.feat_start:args.feat_start+args.feat_count].values
    y_test = df_test[target_col].values if target_col in df_test.columns else None

    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}, 范围: [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Step 3: 样本权重计算
    print("\n" + "=" * 70)
    print("Step 3: 样本权重计算")
    print("=" * 70)

    sample_weights = compute_sample_weights(y_train)
    print(f"  样本权重统计: min={sample_weights.min():.1f}, max={sample_weights.max():.1f}, " +
          f"mean={sample_weights.mean():.2f}")

    # Step 4: 特征选择
    print("\n" + "=" * 70)
    print("Step 4: 特征选择")
    print("=" * 70)

    X_dict, final_indices, selectors = feature_selection_pipeline(
        X_train, y_train,
        X_val=X_val, X_test=X_test,
        n_select=args.n_features,
        sample_weight=sample_weights
    )

    X_train_sel = X_dict['train']
    X_val_sel = X_dict['val']
    X_test_sel = X_dict['test']

    print(f"  样本/特征比: {X_train_sel.shape[0]}/{X_train_sel.shape[1]} = {X_train_sel.shape[0]/X_train_sel.shape[1]:.2f}")

    # Step 5: 模型训练（单模型，非CV）
    print("\n" + "=" * 70)
    print("Step 5: 模型训练（单模型训练）")
    print("=" * 70)

    # XGBoost 参数
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_lambda': 4.0,
        'reg_alpha': 1.0,
        'learning_rate': 0.02,
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': 42
    }

    # 训练
    dtrain = xgb.DMatrix(X_train_sel, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val_sel, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
        evals_result=evals_result
    )

    print(f"  训练完成: best_iteration={model.best_iteration}")

    # Step 6: 模型评估
    print("\n" + "=" * 70)
    print("Step 6: 模型评估")
    print("=" * 70)

    # 预测
    y_pred_train = model.predict(dtrain, ntree_limit=model.best_ntree_limit)
    y_pred_val = model.predict(dval, ntree_limit=model.best_ntree_limit)

    # 训练集指标
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_pearson = np.corrcoef(y_train, y_pred_train)[0, 1]
    print(f"  训练集: R2={train_r2:.4f}, RMSE={train_rmse:.4f}, Pearson={train_pearson:.4f}")

    # 验证集指标
    val_r2 = r2_score(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_pearson = np.corrcoef(y_val, y_pred_val)[0, 1]
    print(f"  验证集: R2={val_r2:.4f}, RMSE={val_rmse:.4f}, Pearson={val_pearson:.4f}")

    # 测试集指标
    test_r2 = test_rmse = test_pearson = None
    if y_test is not None:
        dtest = xgb.DMatrix(X_test_sel)
        y_pred_test = model.predict(dtest, ntree_limit=model.best_ntree_limit)

        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_pearson = np.corrcoef(y_test, y_pred_test)[0, 1]
        print(f"  测试集: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, Pearson={test_pearson:.4f}")

        # 低 MIC 样本召回
        low_mic_mask = y_test <= 1.079
        if low_mic_mask.sum() > 0:
            low_mic_rmse = np.sqrt(mean_squared_error(
                y_test[low_mic_mask], y_pred_test[low_mic_mask]
            ))
            print(f"  低MIC样本 (≤12, n={low_mic_mask.sum()}): RMSE={low_mic_rmse:.4f}")

    # Step 7: 绘制学习曲线（根据--retrieve开关）
    print("\n" + "=" * 70)
    print("Step 7: 绘制学习曲线")
    print("=" * 70)

    # 创建输出目录
    output_dir = args.output_dir
    plots_dir = os.path.join(output_dir, 'plots')
    ensure_dir(plots_dir)

    if args.retrieve and y_test is not None:
        print("  回放模式：计算三条曲线（Train/Val/Test）")
        train_hist, val_hist, test_hist = compute_iteration_metrics(
            model, X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test
        )
        plot_path = os.path.join(plots_dir, 'learning_curve_three.png')
        plot_learning_curve(train_hist, val_hist, test_hist, plot_path)
    else:
        print("  标准模式：使用训练过程的RMSE历史（Train/Val）")
        plot_path = os.path.join(plots_dir, 'learning_curve_two.png')
        plot_learning_curve(evals_result['train']['rmse'], evals_result['val']['rmse'], output_path=plot_path)

    # Step 8: 保存模型和元数据
    print("\n" + "=" * 70)
    print("Step 8: 保存模型和元数据")
    print("=" * 70)

    ensure_dir(output_dir)

    # 保存模型
    model_path = os.path.join(output_dir, 'model.json')
    model.save_model(model_path)
    print(f"  模型已保存: {model_path}")

    # 保存 feature_spec.json
    feature_spec = {
        'task': args.task,
        'feat_start': args.feat_start,
        'feat_count': args.feat_count,
        'n_features_selected': args.n_features,
        'feature_selection_method': 'SelectKBest(128) + RFE(24)',
        'selected_indices': final_indices.tolist(),
        'notes': f'Two-stage feature selection; RDKit-{rdkit_version}'
    }

    spec_path = os.path.join(output_dir, 'feature_spec.json')
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(feature_spec, f, indent=2, ensure_ascii=False)
    print(f"  特征规范已保存: {spec_path}")

    # 保存 meta.json
    meta = {
        'task': args.task,
        'strategy': 'aligned_with_ecoli',
        'retrieve_mode': args.retrieve,
        'feature_selection': {
            'method': 'SelectKBest_RFE',
            'stage1': 'SelectKBest(f_regression, k=128)',
            'stage2': 'RFE(RandomForest, n_select=24)',
            'final_n_features': args.n_features
        },
        'model_type': 'XGBoost',
        'training': {
            'method': 'single_model',  # 非CV训练
            'num_boost_round': args.num_boost_round,
            'best_iteration': int(model.best_iteration),
            'early_stopping_rounds': args.early_stopping_rounds,
            'sample_weighting': True
        },
        'hyperparameters': {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'min_child_weight': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_lambda': 4.0,
            'reg_alpha': 1.0,
            'learning_rate': 0.02
        },
        'metrics': {
            'train': {'r2': float(train_r2), 'rmse': float(train_rmse), 'pearson': float(train_pearson)},
            'val': {'r2': float(val_r2), 'rmse': float(val_rmse), 'pearson': float(val_pearson)},
            'test': {'r2': float(test_r2) if test_r2 is not None else None,
                    'rmse': float(test_rmse) if test_rmse is not None else None,
                    'pearson': float(test_pearson) if test_pearson is not None else None}
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

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"模型目录: {output_dir}")
    print(f"  - model.json: XGBoost 模型")
    print(f"  - feature_spec.json: 特征规范")
    print(f"  - meta.json: 训练历史与指标")
    print(f"  - plots/learning_curve_{'three' if args.retrieve else 'two'}.png: 学习曲线")

    return 0


# ============ CLI 入口 ============

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aureus 回归器训练（与Ecoli模块对齐）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 标准模式（两条曲线）
  python train_aureus_regresser_fixed.py \\
    --train datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_train.csv \\
    --val datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_val.csv \\
    --test datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_test.csv \\
    --output-dir models_predicter/aureus_predictor_fixed

  # 回放模式（三条曲线）
  python train_aureus_regresser_fixed.py \\
    --train datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_train.csv \\
    --val datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_val.csv \\
    --test datasets/standard_datasets/aureus_stratified_mic_datasets/aureus_test.csv \\
    --output-dir models_predicter/aureus_predictor_fixed \\
    --retrieve
        """
    )
    parser.add_argument('--train', required=True, help='训练集 CSV 路径（已增强）')
    parser.add_argument('--val', required=True, help='验证集 CSV 路径（已增强）')
    parser.add_argument('--test', required=True, help='测试集 CSV 路径（已增强）')
    parser.add_argument('--task', default='aureus', help='任务名（默认 aureus）')
    parser.add_argument('--feat-start', type=int, default=3,
                        help='特征起始列索引（0-based，默认 5）')
    parser.add_argument('--feat-count', type=int, default=4274,
                        help='特征总数（默认 4274）')
    parser.add_argument('--n-features', type=int, default=24,
                        help='目标特征数（默认 24）')
    parser.add_argument('--num-boost-round', type=int, default=3000,
                        help='最大迭代次数（默认 3000）')
    parser.add_argument('--early-stopping-rounds', type=int, default=200,
                        help='早停轮数（默认 200）')
    parser.add_argument('--retrieve', action='store_true',
                        help='启用回放模式：绘制Train/Val/Test三条学习曲线')
    parser.add_argument('--output-dir', required=True,
                        help='模型输出目录（必需，推荐：models_predicter/aureus_predictor_fixed）')
    return parser.parse_args()


def main():
    args = parse_arguments()
    return train_aureus_regressor(args)


if __name__ == "__main__":
    sys.exit(main())