#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aureus_reward_module.py - S. aureus MIC预测奖励模块（特征对齐版本）

基于最新训练的模型实现：
- 模型: models_predicter/aureus_regresser/model.json (单模型)
- 特征: 4274D -> 24D（通过索引选择，无PCA）
- 特征对齐: 与训练CSV的4274D表头做有序比对，确保一致性（enhanced_datasets.py负责生成特征）
- 混合奖励: 排序型(60%) + 阈值型(40%)
"""

from .base_reward_module import BaseRewardModule
import numpy as np
import os
import json
import math


class AureusRewardModule(BaseRewardModule):
    """
    S. aureus MIC预测奖励模块（特征对齐版本）

    流程: SMILES -> 4274D特征（enhanced_datasets.py）-> 与训练CSV表头对齐校验 -> 24D索引选择 -> XGBoost -> 混合奖励
    """

    def __init__(self):
        super().__init__()
        self.name = "aureus"

        # 硬编码路径（新模型）
        self.model_dir = os.path.join("models_predicter", "aureus_regresser")
        self.model_path = os.path.join(self.model_dir, "model.json")
        self.feature_spec_path = os.path.join(self.model_dir, "feature_spec.json")

        # 训练集路径（用于特征对齐校验）
        self.train_csv = os.path.join("datasets", "standard_datasets",
                                      "aureus_random_mic_datasets", "aureus_train.csv")

        # 加载统一的回归奖励配置
        self._load_reward_config()

        # 加载模型和特征规格
        self.load_models()

        # 加载训练期特征名列表（用于对齐校验）
        self._load_training_feature_names()

    def _load_reward_config(self):
        """加载统一的回归奖励配置"""
        config_path = os.path.join(os.path.dirname(__file__), "regression_reward_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        reward_config = config["regression_reward_config"]

        # logMIC范围
        self.logmic_min = reward_config["logmic_range"]["logmic_min"]
        self.logmic_max = reward_config["logmic_range"]["logmic_max"]

        # 门限值（直接使用MIC值）
        self.mic_12_threshold = reward_config["bonus_thresholds"]["mic_12_threshold"]
        self.mic_3_threshold = reward_config["bonus_thresholds"]["mic_3_threshold"]

        # 加成值
        self.bonus12 = reward_config["bonus_thresholds"]["bonus12"]
        self.bonus3 = reward_config["bonus_thresholds"]["bonus3"]

        # 底分上限（为加成预留头间隙）
        self.base_max = reward_config["base_reward"]["base_max"]

    def load_models(self):
        """加载XGBoost模型和特征规格"""
        import xgboost as xgb

        # 加载特征选择规格
        with open(self.feature_spec_path, 'r', encoding='utf-8') as f:
            self.feature_spec = json.load(f)

        # 提取关键信息（确保索引为int类型）
        self.selected_indices = list(map(int, self.feature_spec["selected_indices"]))
        self.n_features_selected = self.feature_spec["n_features_selected"]
        self.total_features = self.feature_spec["feat_count"]

        # 验证特征规格
        assert len(self.selected_indices) == self.n_features_selected == 24
        assert self.total_features == 4274

        # 索引越界检查
        assert max(self.selected_indices) < self.total_features, \
            f"索引越界: max({max(self.selected_indices)}) >= {self.total_features}"

        # 读取标准化参数（如果训练时做了缩放）
        self.scalers = self.feature_spec.get("scalers")

        # 加载XGBoost模型
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)

    def _load_training_feature_names(self):
        """从训练CSV中加载特征名列表"""
        import pandas as pd

        # 只读取表头
        train_df = pd.read_csv(self.train_csv, nrows=0, encoding='utf-8-sig')
        feat_start = self.feature_spec["feat_start"]
        feat_count = self.feature_spec["feat_count"]

        # 提取4274维特征列名，存为列表
        self.train_feature_names = list(train_df.columns[feat_start : feat_start + feat_count])

    def _empty_result(self):
        """返回空结果结构"""
        return {
            'rewards': [],
            'metadata': {
                'n_samples': 0,
                'note': 'empty input; skipped scoring'
            }
        }

    def compute_reward(self, smiles_list):
        """
        计算MIC奖励（个别分子跳过版本）

        流程: SMILES -> 4274D -> 特征对齐校验 -> 24D选择 -> 个别分子检查 -> XGBoost -> 混合奖励
        """
        from enhanced_datasets import get_full_features
        import xgboost as xgb

        # 空输入守卫
        if not smiles_list or len(smiles_list) == 0:
            return self._empty_result()

        n_samples = len(smiles_list)

        # 1. 生成4274维特征
        X_full, feature_ids, meta = get_full_features(smiles_list)

        # 2. 特征对齐校验（宽松：按列名重排 + 缺列报错）
        feat_idx = {name: i for i, name in enumerate(feature_ids)}
        missing = [nm for nm in self.train_feature_names if nm not in feat_idx]
        if missing:
            raise ValueError(f"feature mismatch: missing {len(missing)} cols (e.g. {missing[:5]})")
        reorder = [feat_idx[nm] for nm in self.train_feature_names]
        X_full = X_full[:, reorder]
        print("S.aureus_regresser特征已重排对齐")

        # 3. 维度校验
        assert X_full.shape[1] == self.total_features == len(self.train_feature_names) == 4274

        # 4. 按索引选择24维特征
        X_selected = X_full[:, self.selected_indices]
        assert X_selected.shape[1] == self.n_features_selected == 24

        # 5. 应用训练时的标准化（如果有）
        if self.scalers is not None:
            for i, orig_idx in enumerate(self.selected_indices):
                scaler_info = self.scalers.get(str(orig_idx))
                if scaler_info:
                    mean_val = scaler_info["mean"]
                    std_val = scaler_info["std"]
                    X_selected[:, i] = (X_selected[:, i] - mean_val) / (std_val + 1e-8)

        # 6. 个别分子检查，构建有效样本索引
        valid_indices = []
        for i in range(n_samples):
            # 检查1：RDKit解析失败（通过检查原始特征中的NaN来判断）
            if np.isnan(X_full[i, :11]).any():
                continue

            # 检查2：选择后的特征包含NaN/Inf
            feat_row = X_selected[i, :]
            if not np.isfinite(feat_row).all():
                continue

            valid_indices.append(i)

        # 7. 批量预测有效样本
        if not valid_indices:
            # 所有分子都无效，返回全None奖励
            return {
                'rewards': [None] * n_samples,
                'metadata': {
                    'model_type': 'single_xgboost',
                    'feature_method': 'index_selection',
                    'note': 'all samples invalid'
                }
            }

        # 提取有效样本并预测
        X_valid = X_selected[valid_indices]
        dmatrix = xgb.DMatrix(X_valid.astype(np.float32))
        predictions_valid = self.model.predict(dmatrix)
        rewards_valid = self._compute_unified_reward(predictions_valid)

        # 8. 回填结果到原始索引
        rewards = [None] * n_samples  # 使用None而非0.0作为占位值
        mic_list = [None] * n_samples

        # 同时计算并回填 MIC 原值
        logmic_valid = np.asarray(predictions_valid, dtype=float)
        mic_valid = (10.0 ** logmic_valid).astype(float)

        for i, orig_idx in enumerate(valid_indices):
            rewards[orig_idx] = float(rewards_valid[i])
            mic_list[orig_idx] = float(mic_valid[i])

        return {
            'rewards': rewards,
            'mic': mic_list,  # ← 新增：硬口径诊断需要
            'metadata': {
                'model_type': 'single_xgboost',
                'feature_method': 'index_selection',
                'reward_function': 'unified_regression',
                'logmic_range': [self.logmic_min, self.logmic_max],
                'mic_thresholds': [self.mic_3_threshold, self.mic_12_threshold],
                'bonus_values': [self.bonus3, self.bonus12],
                'base_max': self.base_max,
                'selected_features': len(self.selected_indices),
                'n_valid': len(valid_indices),
                'n_total': n_samples
            }
        }

    def _compute_unified_reward(self, predictions):
        """
        统一的回归奖励函数（与E.coli模块完全对齐）

        设计：连续底分(0.80上限) + 阶梯加成(最多0.20) = 总奖励∈[0,1]
        无需最终裁剪，避免饱和问题

        Args:
            predictions: np.array (N,) - logMIC预测值

        Returns:
            rewards: np.array (N,) - 奖励值数组
        """
        rewards = []

        for logmic_pred in predictions:
            # 1. 连续底分（为加成预留头间隙）
            # Base = base_max * clip((logmic_max - y) / (logmic_max - logmic_min), 0, 1)
            base_raw = (self.logmic_max - logmic_pred) / (self.logmic_max - self.logmic_min)
            base_clipped = np.clip(base_raw, 0.0, 1.0)
            base_reward = self.base_max * base_clipped

            # 2. 阶梯加成（基于MIC值判断）
            mic_pred = 10 ** logmic_pred
            bonus = 0.0

            if mic_pred < self.mic_12_threshold:  # MIC < 12
                bonus += self.bonus12
            if mic_pred < self.mic_3_threshold:   # MIC < 3（额外加成）
                bonus += self.bonus3

            # 3. 总奖励（添加0-1上限保护，防未来调参越界）
            final_reward = min(1.0, max(0.0, base_reward + bonus))
            rewards.append(float(final_reward))

        return np.array(rewards)
