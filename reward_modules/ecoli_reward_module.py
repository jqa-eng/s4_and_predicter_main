#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecoli_reward_module.py - E. coli MIC预测奖励模块（特征对齐版本）

基于指导文档实现：
- 模型: models_predicter/ecoli_predictor_random_24d/model.json (单模型)
- 特征: 4274D -> 24D（通过索引选择，无PCA）
- 特征对齐: 与训练CSV的4274D表头做有序比对，确保一致性（enhanced_datasets.py负责生成特征）
- 奖励换算: 与aureus模块保持一致的阈值和计算方式
- 输出结构: 符合指导文档的简化结构，支持None值表示跳过
"""

from .base_reward_module import BaseRewardModule
import numpy as np
import os
import json
import math


class EcoliRewardModule(BaseRewardModule):
    """
    E. coli MIC预测奖励模块（特征对齐版本）

    流程: SMILES -> 4274D特征（enhanced_datasets.py）-> 与训练CSV表头对齐校验 -> 24D索引选择 -> XGBoost -> 奖励转换
    """

    def __init__(self):
        super().__init__()
        self.name = "ecoli"

        # 硬编码路径（使用重训练的模型）
        self.model_dir = os.path.join("models_predicter", "ecoli_regresser")
        self.model_path = os.path.join(self.model_dir, "model.json")
        self.feature_spec_path = os.path.join(self.model_dir, "feature_spec.json")
        self.meta_path = os.path.join(self.model_dir, "meta.json")

        # 训练集路径（用于特征对齐校验）
        self.train_csv = os.path.join("datasets", "standard_datasets",
                                      "ecoli_mic_datasets", "ecoli_train.csv")

        # 加载统一的回归奖励配置
        self._load_reward_config()

        # 内部状态
        self.model = None
        self.feature_spec = None
        self.meta = None
        self.selected_indices = None
        self.train_feature_names = None

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

        # 加载元数据
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        # 提取关键信息（确保索引为int类型）
        self.selected_indices = list(map(int, self.feature_spec["selected_indices"]))
        self.n_features_selected = len(self.selected_indices)
        self.total_features = self.feature_spec["feat_count"]

        # 提取z-score反变换和线性校准参数（修复致命问题）
        target_norm = self.meta.get('target_normalization', {})
        self.y_mean = target_norm.get('y_mean', 0.0)
        self.y_std = target_norm.get('y_std', 1.0)
        self.z_score_enabled = target_norm.get('enabled', False)

        calibration = self.meta.get('calibration', {})
        self.a_val = calibration.get('a_val', 1.0)
        self.b_val = calibration.get('b_val', 0.0)
        self.calibration_enabled = calibration.get('enabled', False)

        print(f"  目标标准化: enabled={self.z_score_enabled}, y_mean={self.y_mean:.4f}, y_std={self.y_std:.4f}")
        print(f"  线性校准: enabled={self.calibration_enabled}, a_val={self.a_val:.4f}, b_val={self.b_val:.4f}")

        # 验证特征规格
        assert self.n_features_selected == 24, f"期望24个特征，实际{self.n_features_selected}个"
        assert self.total_features == 4274, f"期望4274维特征，实际{self.total_features}维"

        # 索引越界检查
        assert max(self.selected_indices) < self.total_features, \
            f"索引越界: max({max(self.selected_indices)}) >= {self.total_features}"

        # 加载XGBoost模型
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)

        print(f"E.coli模型加载成功: {self.n_features_selected}D特征, 模型路径: {self.model_path}")

    def _load_training_feature_names(self):
        """从训练CSV中加载特征名列表"""
        import pandas as pd

        # 只读取表头
        train_df = pd.read_csv(self.train_csv, nrows=0, encoding='utf-8-sig')
        feat_start = self.feature_spec["feat_start"]
        feat_count = self.feature_spec["feat_count"]

        # 提取4274维特征列名，存为列表
        self.train_feature_names = list(train_df.columns[feat_start : feat_start + feat_count])

    def compute_reward(self, smiles_list):
        """
        计算E.coli MIC奖励（按照指导文档要求实现）

        流程:
        1. SMILES -> 4274D特征
        2. 特征对齐校验（宽松：按列名重排 + 缺列报错）
        3. 24D索引选择
        4. 异常处理（RDKit解析失败、NaN/Inf特征）
        5. XGBoost预测
        6. 奖励转换

        Returns:
            dict: {
                "rewards": [0.12, None, 0.34],  # None 表示该分子被跳过
                "metadata": {
                    "label_order": null,  # 回归任务无标签顺序
                    "n_skipped": 1,
                    "skipped_indices": [1],
                    "skipped_reasons": {
                        "parse_error": [1],
                        "invalid_features": []
                    }
                }
            }
        """
        from enhanced_datasets import get_full_features

        # 空输入守卫
        if not smiles_list or len(smiles_list) == 0:
            return {
                "rewards": [],
                "metadata": {
                    "label_order": None,
                    "n_skipped": 0,
                    "skipped_indices": [],
                    "skipped_reasons": {
                        "parse_error": [],
                        "invalid_features": []
                    }
                }
            }

        # 初始化结果结构
        n_samples = len(smiles_list)
        rewards = [None] * n_samples
        mic_full = [None] * n_samples
        skipped_indices = []
        parse_error_indices = []
        invalid_features_indices = []

        try:
            # 1. 生成4274维特征
            X_full, feature_ids, meta = get_full_features(smiles_list)

            # 2. 特征对齐校验（宽松：按列名重排 + 缺列报错）
            # 用列名字典重排到训练期顺序；缺列才报错
            feat_idx = {name: i for i, name in enumerate(feature_ids)}
            missing = [nm for nm in self.train_feature_names if nm not in feat_idx]
            if missing:
                raise ValueError(f"特征对齐失败：缺少列 {missing[:5]} 等共{len(missing)}列")
            reorder = [feat_idx[nm] for nm in self.train_feature_names]
            X_full = X_full[:, reorder]
            print("特征已重排对齐")

            # 3. 维度校验
            assert X_full.shape[1] == self.total_features == len(self.train_feature_names) == 4274
            assert X_full.shape[0] == n_samples

            # 4. 按索引选择24维特征
            X_selected = X_full[:, self.selected_indices]
            assert X_selected.shape[1] == self.n_features_selected == 24

            # 5. 逐个分子检查（复用toxicity模块的逻辑）
            valid_indices = []

            for i, smi in enumerate(smiles_list):
                # 检查1：RDKit解析失败（通过检查原始特征中的NaN来判断）
                # 描述符在X_full的前11列，如果解析失败会是NaN
                if np.isnan(X_full[i, :11]).any():
                    parse_error_indices.append(i)
                    skipped_indices.append(i)
                    continue

                # 检查2：选择后的特征包含NaN/Inf
                feat_row = X_selected[i, :]
                if not np.isfinite(feat_row).all():
                    invalid_features_indices.append(i)
                    skipped_indices.append(i)
                    continue

                valid_indices.append(i)

            # 6. 批量预测有效样本
            if valid_indices:
                import xgboost as xgb

                X_valid = X_selected[valid_indices]
                dmatrix = xgb.DMatrix(X_valid.astype(np.float32))
                predictions_z = self.model.predict(dmatrix)  # z-score标准化后的预测值

                # 7. 应用反z变换和线性校准（修复致命问题！）
                if self.z_score_enabled:
                    # 步骤1：反z变换 (z-score -> 原始logMIC尺度)
                    predictions_logMIC = predictions_z * self.y_std + self.y_mean
                else:
                    predictions_logMIC = predictions_z

                if self.calibration_enabled:
                    # 步骤2：线性校准 (使用验证集的校准参数)
                    predictions_calibrated = self.a_val * predictions_logMIC + self.b_val
                else:
                    predictions_calibrated = predictions_logMIC

                # 8. 计算 MIC 原值并回填结果
                mic_valid = (10.0 ** predictions_calibrated).astype(float)

                for pred_idx, orig_idx in enumerate(valid_indices):
                    logmic_pred = float(predictions_calibrated[pred_idx])
                    reward = self._compute_reward_from_logmic(logmic_pred)
                    rewards[orig_idx] = reward
                    mic_full[orig_idx] = float(mic_valid[pred_idx])

        except ValueError as e:
            # 特征对齐失败应该重新抛出，不应该被捕获
            if "特征对齐失败" in str(e):
                raise e
            else:
                # 其他ValueError当作解析错误处理
                parse_error_indices = list(range(n_samples))
                skipped_indices = list(range(n_samples))
        except Exception as e:
            # 如果是RDKit解析失败等，将所有样本标记为解析错误
            parse_error_indices = list(range(n_samples))
            skipped_indices = list(range(n_samples))

        # 构建返回结果
        return {
            "rewards": rewards,
            "mic": mic_full,  # ← 新增：硬口径诊断需要
            "metadata": {
                "label_order": None,  # 回归任务无标签顺序
                "n_skipped": len(skipped_indices),
                "skipped_indices": skipped_indices,
                "skipped_reasons": {
                    "parse_error": parse_error_indices,
                    "invalid_features": invalid_features_indices
                }
            }
        }

    def _compute_reward_from_logmic(self, logmic_pred):
        """
        统一的回归奖励函数（与Aureus模块完全对齐）

        设计：连续底分(0.80上限) + 阶梯加成(最多0.20) = 总奖励∈[0,1]
        无需最终裁剪，避免饱和问题

        Args:
            logmic_pred: float - logMIC预测值

        Returns:
            float - 奖励值，范围[0, 1]
        """
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

        return float(final_reward)