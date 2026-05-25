#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_sweep.py - 种子扫描模块（特征选择稳定性）

目的：扫描多个random_state，找到在Val集上表现最好的特征组合
原则：只用Train做特征选择，只用Val做评估，Test完全不参与决策

创建日期：2025-10-12
版本：v1.0
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

from .feat_select import select_features_pipeline
from .models import get_best_gbdt_model, train_with_early_stopping, predict_with_best_iteration


def sweep_random_states(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    common_feat_names,
    seeds,
    n_features=24,
    model_prefer='xgboost',
    num_boost_round=3000,
    early_stopping_rounds=200,
    verbose=True
):
    """
    扫描多个random_state，找到Val集上最优的特征组合

    Args:
        X_train: (N, D) 训练集特征矩阵
        y_train: (N,) 训练集目标（原始尺度）
        X_val: (M, D) 验证集特征矩阵
        y_val: (M,) 验证集目标（原始尺度）
        X_test: (K, D) 测试集特征矩阵（仅用于记录，不参与选择）
        y_test: (K,) 测试集目标（原始尺度）
        common_feat_names: list - 特征名列表（用于映射final_idx到名称）
        seeds: list of int - 待扫描的random_state列表
        n_features: 目标特征数
        model_prefer: 模型优先级
        num_boost_round: 最大迭代数
        early_stopping_rounds: 早停轮数
        verbose: 是否打印详细信息

    Returns:
        results_df: DataFrame包含每个seed的结果
        best_seed: int - Val R2最优的seed
        best_features: list - 最优特征名列表
    """

    # z-score标准化目标
    y_mean = y_train.mean()
    y_std = y_train.std() if y_train.std() > 1e-8 else 1.0
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    def inv_z_score(pred_z):
        return pred_z * y_std + y_mean

    results = []

    for seed in seeds:
        if verbose:
            print(f"\n{'='*80}")
            print(f"扫描 random_state={seed}")
            print(f"{'='*80}")

        try:
            # 1) 特征选择（使用当前seed）
            X_dict, final_idx, _ = select_features_pipeline(
                X_train, y_train_z,
                X_val=X_val, X_test=X_test,
                n_select=n_features,
                random_state=seed,  # 关键：使用当前seed
                sample_weight=None
            )

            # 获取选中的特征名
            selected_feat_names = [common_feat_names[i] for i in final_idx]

            # 2) 训练模型
            model, model_name, _ = get_best_gbdt_model(
                prefer=model_prefer,
                task='regression',
                random_state=seed,
                n_estimators=num_boost_round
            )

            X_train_sel = X_dict['train']
            X_val_sel = X_dict['val']
            X_test_sel = X_dict['test']

            model, best_iter = train_with_early_stopping(
                model, model_name,
                X_train_sel, y_train_z,
                X_val_sel, y_val_z,
                sample_weight=None,
                early_stopping_rounds=early_stopping_rounds
            )

            # 3) 预测并反变换
            y_pred_val_z = predict_with_best_iteration(model, model_name, X_val_sel)
            y_pred_test_z = predict_with_best_iteration(model, model_name, X_test_sel)

            y_pred_val = inv_z_score(y_pred_val_z)
            y_pred_test = inv_z_score(y_pred_test_z)

            # 4) 计算指标
            val_r2 = r2_score(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_pearson, _ = pearsonr(y_val, y_pred_val)

            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_pearson, _ = pearsonr(y_test, y_pred_test)

            if verbose:
                print(f"\n结果:")
                print(f"  Val  R2={val_r2:.3f}, RMSE={val_rmse:.3f}, Pearson={val_pearson:.3f}")
                print(f"  Test R2={test_r2:.3f}, RMSE={test_rmse:.3f}, Pearson={test_pearson:.3f}")
                print(f"  best_iter={best_iter}")
                print(f"  选中特征(前5个): {selected_feat_names[:5]}")

            results.append({
                'seed': seed,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_pearson': val_pearson,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_pearson': test_pearson,
                'best_iter': best_iter,
                'selected_features': selected_feat_names
            })

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            results.append({
                'seed': seed,
                'val_r2': -999,
                'val_rmse': 999,
                'val_pearson': -999,
                'test_r2': -999,
                'test_rmse': 999,
                'test_pearson': -999,
                'best_iter': -1,
                'selected_features': []
            })

    # 转为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_sorted = results_df.sort_values('val_r2', ascending=False)

    # 获取最优配置
    best_row = results_sorted.iloc[0]
    best_seed = int(best_row['seed'])
    best_features = best_row['selected_features']

    if verbose:
        print(f"\n{'='*80}")
        print(f"扫描完成！")
        print(f"{'='*80}")
        print("\n按Val R2排序:")
        print(results_sorted[['seed', 'val_r2', 'val_rmse', 'val_pearson', 'test_r2', 'best_iter']].to_string())
        print(f"\n最优 random_state: {best_seed}")
        print(f"  Val R2={best_row['val_r2']:.3f}")
        print(f"  Test R2={best_row['test_r2']:.3f} (仅供参考，不参与决策)")
        print(f"  best_iter={best_row['best_iter']}")

    return results_df, best_seed, best_features


def select_best_seed_for_features(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    common_feat_names,
    seeds=None,
    n_features=24,
    output_csv=None
):
    """
    便捷函数：扫描种子并返回最优特征配置（推荐用于训练脚本）

    Args:
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集（仅记录）
        common_feat_names: 特征名列表
        seeds: random_state列表（默认为[0,1,2,13,29,37,42,97,123]）
        n_features: 目标特征数
        output_csv: 可选，保存扫描结果的CSV路径

    Returns:
        dict: {
            'best_seed': int,
            'selected_names': list,
            'val_metrics': dict,
            'results_df': DataFrame
        }
    """
    if seeds is None:
        seeds = [0, 1, 2, 13, 29, 37, 42, 97, 123]

    results_df, best_seed, best_features = sweep_random_states(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        common_feat_names,
        seeds=seeds,
        n_features=n_features,
        verbose=True
    )

    # 保存结果
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n扫描结果已保存: {output_csv}")

    # 获取最优行的Val指标
    best_row = results_df.sort_values('val_r2', ascending=False).iloc[0]

    return {
        'best_seed': best_seed,
        'selected_names': best_features,
        'val_metrics': {
            'r2': float(best_row['val_r2']),
            'rmse': float(best_row['val_rmse']),
            'pearson': float(best_row['val_pearson'])
        },
        'results_df': results_df
    }


def find_best_seed_for_training(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    common_feat_names,
    n_features=24,
    seeds=None,
    save_dir=None
):
    """
    【推荐】训练前调用：自动找到最优random_state，用于后续训练

    用途：
        在正式训练前，扫描多个random_state，找到在Val集上表现最好的种子。
        训练脚本可以直接调用此函数，获取最优种子后再进行特征选择和训练。

    Args:
        X_train: (N, D) 训练集特征矩阵（原始维度）
        y_train: (N,) 训练集目标（原始尺度，函数内部会做z-score）
        X_val: (M, D) 验证集特征矩阵
        y_val: (M,) 验证集目标
        X_test: (K, D) 测试集特征矩阵（仅记录，不参与决策）
        y_test: (K,) 测试集目标
        common_feat_names: list - 特征名列表（长度=D）
        n_features: 目标特征数（默认24）
        seeds: list of int - 待扫描的种子列表（默认[0,1,2,13,29,37,42,97,123]）
        save_dir: str - 可选，保存扫描结果的目录（默认不保存）

    Returns:
        dict: {
            'best_seed': int - 最优种子
            'best_features': list - 最优特征名列表（长度=n_features）
            'val_r2': float - 验证集R²
            'val_rmse': float - 验证集RMSE
            'test_r2': float - 测试集R²（仅参考）
            'best_iter': int - 最优迭代数
            'all_results': DataFrame - 所有种子的结果表
        }

    示例：
        # 在训练脚本中调用
        result = find_best_seed_for_training(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            common_feat_names,
            n_features=24,
            save_dir='seed_sweep_output'
        )

        # 使用最优配置训练
        best_seed = result['best_seed']
        X_dict, final_idx, _ = select_features_pipeline(
            X_train, y_train,
            X_val=X_val, X_test=X_test,
            n_select=24,
            random_state=best_seed  # 使用最优种子
        )
    """
    if seeds is None:
        seeds = [0, 1, 2, 13, 29, 37, 42, 97, 123]

    print("\n" + "=" * 80)
    print("种子扫描：查找最优random_state用于特征选择")
    print("=" * 80)
    print(f"扫描种子列表: {seeds}")
    print(f"目标特征数: {n_features}")
    print(f"数据规模: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"决策依据: 仅使用Val R²（Test不参与决策）")

    # 调用核心扫描函数
    results_df, best_seed, best_features = sweep_random_states(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        common_feat_names,
        seeds=seeds,
        n_features=n_features,
        verbose=True
    )

    # 保存结果到文件
    if save_dir:
        from .utils import ensure_dir
        ensure_dir(save_dir)

        csv_path = f"{save_dir}/seed_sweep_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n扫描结果已保存: {csv_path}")

        # 保存最优配置
        import json
        best_row = results_df.sort_values('val_r2', ascending=False).iloc[0]
        config = {
            'best_seed': int(best_seed),
            'selected_features': best_features,
            'val_metrics': {
                'r2': float(best_row['val_r2']),
                'rmse': float(best_row['val_rmse']),
                'pearson': float(best_row['val_pearson'])
            },
            'test_metrics': {
                'r2': float(best_row['test_r2']),
                'rmse': float(best_row['test_rmse']),
                'pearson': float(best_row['test_pearson'])
            },
            'best_iter': int(best_row['best_iter'])
        }

        json_path = f"{save_dir}/best_seed_config.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"最优配置已保存: {json_path}")

    # 获取最优种子的指标
    best_row = results_df.sort_values('val_r2', ascending=False).iloc[0]

    return {
        'best_seed': int(best_seed),
        'best_features': best_features,
        'val_r2': float(best_row['val_r2']),
        'val_rmse': float(best_row['val_rmse']),
        'test_r2': float(best_row['test_r2']),
        'best_iter': int(best_row['best_iter']),
        'all_results': results_df
    }
