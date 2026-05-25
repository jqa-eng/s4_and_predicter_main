#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_reward_module.py - 奖励模块抽象基类

定义所有奖励模块的统一接口，确保模块化系统的一致性。
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseRewardModule(ABC):
    """
    抽象基类：定义所有奖励模块的统一接口

    所有奖励模块必须继承此类并实现抽象方法。
    """

    def __init__(self):
        self.name = "base"           # 模块名称
        self.enabled = False         # 是否启用
        self.model_path = None       # 模型路径（子类设置）
        self.pca_path = None         # PCA路径（可选）
        self.lcb_coef = 1.0          # LCB系数

    @abstractmethod
    def load_models(self):
        """
        加载预测器模型（子类实现）

        子类应该在此方法中：
        1. 加载训练好的模型文件
        2. 加载特征降维对象（如PCA）
        3. 加载校准器（如有）
        4. 打印加载信息
        """
        pass

    @abstractmethod
    def compute_reward(self, smiles_list):
        """
        计算奖励

        Args:
            smiles_list: List[str] - SMILES字符串列表

        Returns:
            dict: {
                'rewards': np.array (N,) - 奖励值,
                'predictions': np.array (N,) or (N, C) - 原始预测,
                'lcb': np.array (N,) - 下置信界（可选）,
                'std': np.array (N,) - 标准差（可选）,
                'metadata': dict - 其他信息（如类别概率、阈值等）
            }

        Notes:
            - rewards: 最终奖励值，范围通常在[0, 1]或[-1, 1]
            - predictions: 原始预测值（回归任务为logMIC，分类任务为概率）
            - lcb: 下置信界（用于不确定性估计）
            - std: 标准差（集成模型时计算）
            - metadata: 模块特定的附加信息
        """
        pass

    def _validate_smiles(self, smiles_list):
        """
        验证SMILES有效性（通用方法）

        Args:
            smiles_list: List[str] - SMILES字符串列表

        Returns:
            valid_smiles: List[str] - 有效的SMILES
            valid_indices: List[int] - 有效SMILES的索引
        """
        from rdkit import Chem
        valid_smiles = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_indices.append(i)
        return valid_smiles, valid_indices
