#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py - GBDT模型选择与训练（保守策略）

优先级：LightGBM > HGBR > XGBoost
核心原则：强正则化 + 早停 + GPU加速（如可用）

创建日期：2025-10-10
版本：v1.0
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_best_gbdt_model(prefer='lightgbm', task='regression', random_state=42, n_estimators=3000):
    """
    按优先级返回最优GBDT模型

    Args:
        prefer: str - 优先选择的模型（'lightgbm', 'hgbr', 'xgboost'）
        task: str - 任务类型（'regression', 'classification'）
        random_state: int - 随机种子
        n_estimators: int - 最大迭代次数（默认3000）

    Returns:
        tuple: (model_obj, model_name, supports_sample_weight)
    """
    models_priority = []

    if prefer == 'lightgbm':
        models_priority = ['lightgbm', 'hgbr', 'xgboost']
    elif prefer == 'hgbr':
        models_priority = ['hgbr', 'lightgbm', 'xgboost']
    elif prefer == 'xgboost':
        models_priority = ['xgboost', 'lightgbm', 'hgbr']
    else:
        models_priority = ['lightgbm', 'hgbr', 'xgboost']

    # GPU检测（XGBoost + LightGBM）
    xgb_tree_method = check_gpu_availability()
    lgb_device =  'cpu'

    for model_name in models_priority:
        try:
            if model_name == 'lightgbm':
                import lightgbm as lgb


                lgbm_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'num_leaves': 15,
                    'min_child_samples': 8,
                    'subsample': 0.8,
                    'colsample_bytree': 0.6,
                    'reg_lambda': 6.0,
                    'reg_alpha': 0.5,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }

                # 尝试添加device=cpu参数（仅LightGBM 4.x支持）
                try:
                    test_model = lgb.LGBMRegressor(n_estimators=1, device='cpu')
                    lgbm_params['device'] = 'cpu'
                    device_info = "device=cpu"
                except TypeError:
                    # LightGBM 3.x不支持device参数
                    device_info = "device=cpu (3.x版本)"

                if task == 'regression':
                    model = lgb.LGBMRegressor(**lgbm_params)
                else:
                    model = lgb.LGBMClassifier(**lgbm_params)

                print(f"已选择模型: LightGBM ({device_info}, n_estimators={n_estimators})")
                return model, 'lightgbm', True

        except (ImportError, Exception) as e:
            if isinstance(e, ImportError):
                print(f"LightGBM不可用，尝试下一个...")
            else:
                print(f"LightGBM初始化失败 ({e})，尝试下一个...")
            continue

        if model_name == 'hgbr':
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
                if task == 'regression':
                    model = HistGradientBoostingRegressor(
                        max_iter=n_estimators,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_leaf=8,
                        l2_regularization=6.0,
                        random_state=random_state,
                        early_stopping=True,
                        n_iter_no_change=200,
                        validation_fraction=0.1
                    )
                else:
                    model = HistGradientBoostingClassifier(
                        max_iter=n_estimators,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_leaf=8,
                        l2_regularization=6.0,
                        random_state=random_state,
                        early_stopping=True,
                        n_iter_no_change=200,
                        validation_fraction=0.1
                    )
                print(f"已选择模型: HistGradientBoosting (sklearn, max_iter={n_estimators})")
                return model, 'hgbr', True

            except Exception as e:
                print(f"HGBR不可用 ({e})，尝试下一个...")
                continue

        if model_name == 'xgboost':
            try:
                import xgboost as xgb
                if task == 'regression':
                    # GPT-5 v4配方：可学幅度、但不过拟合（目标：best_iter 400-1200）
                    model = xgb.XGBRegressor(
                        n_estimators=n_estimators,  # CLI传入（建议8000）
                        learning_rate=0.03,          # v4: 0.02→0.03（适中步长）
                        max_depth=6,                 # v4: 5→6（放开容量，学习幅度）
                        min_child_weight=2,          # v4: 5→2（关键：更易学幅度）
                        gamma=0.2,                   # v4: 0.5→0.2（轻分裂代价）
                        subsample=0.80,              # v4: 保持0.8（温和采样）
                        colsample_bytree=0.70,       # v4: 保持0.7
                        reg_lambda=8.0,              # v4: 12.0→8.0（轻正则）
                        reg_alpha=0.5,               # v4: 2.0→0.5（轻L1）
                        base_score=0.0,              # v4: 配合y的z-score，基准置0
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method=xgb_tree_method,
                        verbosity=0
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=0.03,
                        max_depth=6,
                        min_child_weight=2,
                        gamma=0.2,
                        subsample=0.80,
                        colsample_bytree=0.70,
                        reg_lambda=8.0,
                        reg_alpha=0.5,
                        base_score=0.0,
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method=xgb_tree_method,
                        verbosity=0
                    )
                print(f"已选择模型: XGBoost (tree_method={xgb_tree_method}, n_estimators={n_estimators}, GPT-5 v4配方: 可学幅度+温和正则)")
                return model, 'xgboost', True

            except ImportError:
                print(f"XGBoost不可用，尝试下一个...")
                continue

    raise RuntimeError("无可用GBDT库 (LightGBM/HGBR/XGBoost均不可用)")


def train_with_early_stopping(model, model_name, X_train, y_train, X_val, y_val,
                               sample_weight=None, early_stopping_rounds=200):
    """
    统一的GBDT训练接口（带早停）

    Args:
        model: GBDT模型对象
        model_name: str - 模型名称（'lightgbm', 'hgbr', 'xgboost'）
        X_train: np.array - 训练特征
        y_train: np.array - 训练标签
        X_val: np.array - 验证特征
        y_val: np.array - 验证标签
        sample_weight: np.array - 样本权重（可选）
        early_stopping_rounds: int - 早停轮数

    Returns:
        model: 训练后模型
        best_iter: int - 最优迭代次数
    """
    if model_name == 'lightgbm':
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_val, y_val)],  # 包含训练集历史
            eval_names=['train', 'valid'],  # 显式指定键名
            eval_metric='rmse',
            callbacks=[
                __import__('lightgbm').early_stopping(early_stopping_rounds),
                __import__('lightgbm').log_evaluation(period=0)
            ]
        )
        best_iter = model.best_iteration_

    elif model_name == 'hgbr':
        # HGBR内置早停（通过validation_fraction）
        model.fit(X_train, y_train, sample_weight=sample_weight)
        best_iter = model.n_iter_

    elif model_name == 'xgboost':
        # 记录两条曲线：validation_0=Train, validation_1=Val
        model.set_params(early_stopping_rounds=early_stopping_rounds, eval_metric='rmse')
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_train, y_train), (X_val, y_val)],  # 同时记录train和val
            verbose=False
        )
        # 获取最佳迭代（兼容不同版本）
        best_iter = getattr(model, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(model, 'best_ntree_limit', 0)

    else:
        raise ValueError(f"未知模型类型: {model_name}")

    return model, best_iter


def predict_with_best_iteration(model, model_name, X):
    """
    使用最佳迭代次数进行预测（避免过拟合）

    Args:
        model: 训练好的GBDT模型
        model_name: str - 模型类型
        X: np.array - 预测特征

    Returns:
        np.array: 预测值
    """
    if model_name == 'xgboost':
        # XGBoost 1.7.6+: 使用iteration_range参数
        best_iter = getattr(model, 'best_iteration', None)
        if best_iter is not None:
            return model.predict(X, iteration_range=(0, best_iter + 1))

        # 兜底：使用ntree_limit（旧版本）
        best_ntree = getattr(model, 'best_ntree_limit', None)
        if best_ntree is not None and best_ntree > 0:
            return model.predict(X, ntree_limit=best_ntree)

    elif model_name == 'lightgbm':
        # LightGBM: 使用num_iteration参数
        best_iter = getattr(model, 'best_iteration_', None)
        if best_iter is not None:
            return model.predict(X, num_iteration=best_iter)

    # HGBR或无best_iteration属性：使用默认预测
    return model.predict(X)


def evaluate_regression_model(model, model_name, X, y, prefix=''):
    """
    评估回归模型性能（使用最佳迭代）

    Args:
        model: 训练好的模型
        model_name: str - 模型类型（'lightgbm', 'hgbr', 'xgboost'）
        X: np.array - 特征
        y: np.array - 真实标签
        prefix: str - 输出前缀（如'Train', 'Val', 'Test'）

    Returns:
        dict: {'r2', 'rmse', 'pearson'}
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score, mean_squared_error

    # 使用最佳迭代进行预测（避免过拟合）
    y_pred = predict_with_best_iteration(model, model_name, X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    pearson, _ = pearsonr(y, y_pred)

    print(f"{prefix} R2={r2:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}")

    return {
        'r2': r2,
        'rmse': rmse,
        'pearson': pearson
    }


def check_gpu_availability():
    """
    检测GPU是否可用（XGBoost专用）

    Returns:
        str: 'gpu_hist' if available, else 'hist'
    """
    try:
        import xgboost as xgb
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("检测到NVIDIA GPU，XGBoost将使用gpu_hist")
            return 'gpu_hist'
    except:
        pass

    print("未检测到GPU或XGBoost GPU支持，使用CPU")
    return 'hist'


def get_feature_importance(model, model_name, feature_names=None):
    """
    获取特征重要性

    Args:
        model: 训练好的GBDT模型
        model_name: str - 模型类型
        feature_names: list - 特征名列表（可选）

    Returns:
        np.array: 特征重要性分数（与特征顺序对应）
    """
    if model_name in ['lightgbm', 'xgboost']:
        importance = model.feature_importances_
    elif model_name == 'hgbr':
        # HGBR也有feature_importances_属性
        importance = model.feature_importances_
    else:
        importance = np.ones(len(feature_names)) if feature_names else np.array([])

    return importance
