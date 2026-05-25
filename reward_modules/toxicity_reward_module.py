#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toxicity_reward_module.py - 细胞毒性分类奖励模块（重构版）

重构要点：
- 以 model.classes_ 为唯一真值进行奖励对齐
- 空输入短路返回
- 严格特征对齐校验 + 极简输出
- 解析/特征异常样本跳过机制
- 统一 NaN 策略为跳过处理
- 仅保留单模型逻辑
"""

from .base_reward_module import BaseRewardModule
import numpy as np
import json
import joblib


class ToxicityRewardModule(BaseRewardModule):
    """
    细胞毒性分类奖励模块（重构版）

    流程: SMILES -> 验证 -> 4274D特征 -> 特征对齐校验 -> 异常跳过 -> 单模型预测 -> 概率期望奖励
    """

    def __init__(self):
        super().__init__()
        self.name = "toxicity"

        # 固定路径（符合既定规范）
        self.model_path = "models_predicter/toxicity_classifier_4274d/toxicity_classifier.pkl"
        self.meta_path = "models_predicter/toxicity_classifier_4274d/meta.json"

        # =============================================
        # 奖励权重配置 - 用户可在此处修改
        # =============================================
        self.label_rewards = {
            "低毒": 1.0,
            "微毒": 0.8,
            "中毒": -0.2,
            "高毒": -0.6
        }

        # 加载模型
        self.load_models()

    def load_models(self):
        """加载单模型分类器 + 元信息，以model.classes_为唯一真值进行奖励对齐"""
        # 加载模型
        self.model = joblib.load(self.model_path)

        # 加载元信息
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        # 获取标签映射和特征名
        self.label_mapping = self.meta["label_mapping"]
        self.feature_names_train = self.meta["features"]["feature_names"]

        # 以model.classes_为唯一真值构建标签顺序
        model_classes = self.model.classes_  # 模型的真实类别顺序
        self.label_order = [self.label_mapping[str(cls)] for cls in model_classes]

        # 构建与model.classes_严格对齐的奖励权重向量
        self.weights_aligned = np.array([self.label_rewards[label] for label in self.label_order])

    def compute_reward(self, smiles_list):
        """
        计算毒性奖励（重构版）

        流程:
        1. 空输入短路返回
        2. 生成4274D特征
        3. 严格特征对齐校验
        4. 异常样本跳过处理
        5. 单模型预测
        6. 概率期望奖励
        """
        # 1. 空输入短路返回
        if not smiles_list or len(smiles_list) == 0:
            return {
                'rewards': [],
                'metadata': {
                    'label_order': self.label_order,
                    'n_skipped': 0,
                    'skipped_indices': [],
                    'skipped_reasons': {'parse_error': [], 'invalid_features': []}
                }
            }

        # 2. 生成4274D特征
        X, feature_names_pred = self._generate_features(smiles_list)

        # 3. 特征对齐校验（宽松：按列名重排 + 缺列报错）
        feat_idx = {n: i for i, n in enumerate(feature_names_pred)}
        missing = [n for n in self.feature_names_train if n not in feat_idx]
        if missing:
            raise ValueError(f"toxicity features missing cols: {missing[:5]} ...")

        reorder = [feat_idx[n] for n in self.feature_names_train]
        X = X[:, reorder]
        print("细胞毒性分类器特征已重排对齐")

        # 4. 异常样本跳过处理
        X_clean, valid_indices, skipped_reasons = self._skip_invalid_samples(X, smiles_list)

        # 5. 单模型预测（仅对有效样本）
        if len(valid_indices) == 0:
            # 全部样本都被跳过
            return self._create_all_skipped_result(smiles_list, skipped_reasons)

        # 对有效样本进行预测
        proba_clean = self.model.predict_proba(X_clean)  # (N_valid, 4) 与 model.classes_ 同序

        # 低毒最好、微毒略差、中毒更差、高毒最差 —— 严重度 ∈ [0,1]
        severity_base = {"低毒": 0.0, "微毒": 0.1, "中毒": 0.5, "高毒": 1.0}
        # 与 model.classes_ → self.label_order 对齐到列顺序
        severity_w = np.array([severity_base[label] for label in self.label_order], dtype=float)  # shape=(4,)

        sev = (proba_clean * severity_w.reshape(1, -1)).sum(axis=1)  # 期望严重度 ∈[0,1]
        rewards_clean = 1.0 - sev                                    # R_tox = 安全度 ∈[0,1]

        # === 规范顺序重排（中毒, 低毒, 微毒, 高毒）并生成规范标签编码 ===
        canonical_order = ["中毒", "低毒", "微毒", "高毒"]
        # 计算从规范顺序到当前模型列顺序的索引映射
        idx_map = [self.label_order.index(name) for name in canonical_order]
        proba_canon = proba_clean[:, idx_map]  # (N_valid, 4) 现在是规范列序
        labels_canon = np.argmax(proba_canon, axis=1)  # 0..3 对应 canonical_order

        # 6. 回填到原顺序（rewards / labels / probs 都要回填）
        N = len(smiles_list)
        rewards_full = [None] * N
        labels_full  = [None] * N        # 规范编码：0中毒,1低毒,2微毒,3高毒
        probs_full   = [None] * N        # 每项为长度4的list（规范列序）

        for i, vidx in enumerate(valid_indices):
            rewards_full[vidx] = float(rewards_clean[i])
            labels_full[vidx]  = int(labels_canon[i])
            probs_full[vidx]   = proba_canon[i].astype(float).tolist()

        # 计算统计信息
        skip_count = len(smiles_list) - len(valid_indices)

        return {
            'rewards': rewards_full,
            'labels':  labels_full,   # ← 新增：硬口径诊断需要（规范编码）
            'probs':   probs_full,    # ← 新增：可用于安全概率与软门
            'metadata': {
                'label_order': self.label_order,
                'n_skipped': skip_count,
                'skipped_indices': [i for i in range(len(smiles_list)) if i not in valid_indices],
                'skipped_reasons': skipped_reasons
            }
        }

    def _generate_features(self, smiles_list):
        """生成4274维特征并返回特征名称"""
        from enhanced_datasets import get_full_features

        # 使用enhanced_datasets的get_full_features函数
        X, feature_names, _ = get_full_features(smiles_list)

        return X, feature_names

    def _skip_invalid_samples(self, X, smiles_list):
        """
        跳过异常样本：解析失败或特征包含NaN/Inf
        一次遍历完成，避免二次解析

        Returns:
            X_clean: 有效样本的特征矩阵
            valid_indices: 有效样本的原始索引列表
            skipped_reasons: 结构化跳过原因 {'parse_error': [...], 'invalid_features': [...]}
        """
        from rdkit import Chem

        valid_indices = []
        skipped_reasons = {'parse_error': [], 'invalid_features': []}

        for i, (smi, feat_row) in enumerate(zip(smiles_list, X)):
            # 检查1：RDKit解析失败
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped_reasons['parse_error'].append(i)
                continue

            # 检查2：特征包含NaN/Inf
            if not np.isfinite(feat_row).all():
                skipped_reasons['invalid_features'].append(i)
                continue

            valid_indices.append(i)

        # 提取有效样本的特征
        if len(valid_indices) > 0:
            X_clean = X[valid_indices]
        else:
            X_clean = np.empty((0, X.shape[1]))

        return X_clean, valid_indices, skipped_reasons

    def _create_all_skipped_result(self, smiles_list, skipped_reasons):
        """创建全部样本被跳过时的返回结果"""
        n_samples = len(smiles_list)
        return {
            'rewards': [None] * n_samples,
            'metadata': {
                'label_order': self.label_order,
                'n_skipped': n_samples,
                'skipped_indices': list(range(n_samples)),
                'skipped_reasons': skipped_reasons
            }
        }

    def _backfill_results(self, rewards_clean, valid_indices, total_samples):
        """将有效样本的结果回填到原始顺序"""
        # 使用list避免numpy object数组问题
        rewards_full = [None] * total_samples

        # 回填有效样本的结果
        for i, valid_idx in enumerate(valid_indices):
            rewards_full[valid_idx] = float(rewards_clean[i])

        return rewards_full

