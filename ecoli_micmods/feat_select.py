#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feat_select.py - 两阶段特征选择（SelectKBest + RFE）

Pipeline:
  Stage 1: 零方差过滤 + 相关性预筛(Top-256) + 共线性剔除(ρ>0.98)
  Stage 2: SelectKBest(f_regression)快速预筛到160 → RFE递归消除到n_select

关键特性：
  - Stage1: 确定性过滤（零方差、共线性、弱相关）
  - Stage2.1: SelectKBest快速降维（4000+ → 160，加速RFE）
  - Stage2.2: RFE实质性特征选择（160 → 24，考虑特征交互）
  - RFE使用RandomForest作为基模型，每次删除1个特征
  - 特征选择结果固定（给定random_state）

创建日期：2025-10-10
最后更新：2025-10-12（改用RFE作为Stage2主方法）
版本：v2.2
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import KFold


def stage1_filter(X, y, var_thresh_ratio=0.10, spearman_th=0.98, target_corr_min=0.0, top_k_corr=256):
    """
    Stage 1: 零方差+目标相关性预筛+共线性过滤（稳健版）

    Args:
        X: (N, D) 原始特征矩阵
        y: (N,) 目标值
        var_thresh_ratio: 方差阈值比例（用于分位数，默认0.10）
        spearman_th: Spearman相关系数阈值（共线性，放宽至0.98）
        target_corr_min: 与目标最小相关性阈值（默认0.0，关闭弱相关过滤）
        top_k_corr: 按相关性预筛到Top-K（默认256）

    Returns:
        X_filtered: (N, D') 过滤后特征
        selected_idx: 保留的原始索引
    """
    print(f"\n[Stage 1] 零方差+相关性预筛+共线性过滤")
    print(f"  输入特征数: {X.shape[1]}")

    # 1) 零方差过滤（直接删除常数列）
    var = X.var(axis=0)
    keep_var = var > 1e-8  # 几乎为零的方差视为常数
    X1 = X[:, keep_var]
    idx1 = np.where(keep_var)[0]
    print(f"  零方差过滤后: {X1.shape[1]} 特征（删除 {X.shape[1] - X1.shape[1]} 个零方差特征）")

    # 2) 与y的相关性计算（用于排序和过滤）
    rho = np.array([abs(spearmanr(X1[:, j], y)[0]) for j in range(X1.shape[1])])
    rho = np.nan_to_num(rho)  # NaN替换为0

    # 2.1) 按相关性预筛到Top-K（大幅降维）
    if X1.shape[1] > top_k_corr:
        top_k_idx = np.argsort(rho)[::-1][:top_k_corr]
        X2 = X1[:, top_k_idx]
        idx2 = idx1[top_k_idx]
        rho2 = rho[top_k_idx]
        print(f"  相关性预筛后: {X2.shape[1]} 特征（Top-{top_k_corr}）")
    else:
        X2 = X1
        idx2 = idx1
        rho2 = rho
        print(f"  相关性预筛: 跳过（特征数 <= {top_k_corr}）")

    # 2.2) 弱相关过滤（删除与目标几乎无关的特征）
    keep_corr = rho2 >= target_corr_min
    X3 = X2[:, keep_corr]
    idx3 = idx2[keep_corr]
    rho3 = rho2[keep_corr]
    print(f"  弱相关过滤后: {X3.shape[1]} 特征（|ρ| >= {target_corr_min}）")

    # 3) 共线性剔除（保留与y相关性更高的）
    keep = np.ones(X3.shape[1], dtype=bool)
    for i in range(X3.shape[1]):
        if not keep[i]:
            continue
        for k in range(i + 1, X3.shape[1]):
            if not keep[k]:
                continue
            r, _ = spearmanr(X3[:, i], X3[:, k])
            if not np.isnan(r) and abs(r) > spearman_th:
                # 保留与y相关性更高的
                if rho3[i] >= rho3[k]:
                    keep[k] = False
                else:
                    keep[i] = False
                    break  # i已被删除，跳出内循环

    X_out = X3[:, keep]
    idx_out = idx3[keep]

    print(f"  共线性过滤后: {X_out.shape[1]} 特征（|ρ| > {spearman_th}）")
    print(f"  Stage1总计: {X.shape[1]} -> {X_out.shape[1]}")

    return X_out, idx_out


def stage2_rfecv(
    X, y,
    n_select=24,  # 默认值，实际由CLI参数传入
    random_state=42,
    sample_weight=None,
    *,
    prefilter_k=160,
    step=0.2,         # RFE块删除比例（暂不使用）
    rf_n_estimators=200,
    rf_max_depth=12
):
    """
    Stage 2: SelectKBest快速预筛 + RFE递归消除

    两步流程：
    1) SelectKBest(f_regression): 快速降维到160（加速后续RFE）
    2) RFE(RandomForest): 递归特征消除到n_select（实质性特征选择）

    RFE优势：
    - 考虑特征交互作用（不是单独评估每个特征）
    - 递归消除最不重要的特征
    - 使用RandomForest作为基模型，更稳健

    Args:
        X: (N, D) Stage1过滤后特征
        y: (N,) 目标值
        n_select: 目标特征数（默认24）
        random_state: 随机种子（控制RandomForest的随机性）
        sample_weight: 样本权重（可选，传递给RFE）
        prefilter_k: SelectKBest预筛特征数（默认160）
        step: RFE每次删除特征数（默认1，更精细）
        rf_n_estimators: RandomForest树数量（默认200）
        rf_max_depth: RandomForest最大深度（默认12）

    Returns:
        X_selected: (N, n_select) 选择后特征
        selected_idx: 保留的Stage1后索引
        rfe_obj: RFE对象（包含特征排名信息）
    """
    print(f"\n[Stage 2] SelectKBest预筛 + RFE精选 → Top-{n_select}")
    print(f"  输入特征数: {X.shape[1]}")

    # --- 1) 快速预筛到160（使用F统计量，加速RFE）
    k_pre = min(prefilter_k, X.shape[1])
    if k_pre < X.shape[1]:
        print(f"  Step 1: SelectKBest(f_regression) -> {k_pre}特征（快速预筛）")
        skb = SelectKBest(score_func=f_regression, k=k_pre)
        X_pre = skb.fit_transform(X, y)
        kept_idx_pre = np.where(skb.get_support())[0]
        print(f"    完成: {X.shape[1]} -> {X_pre.shape[1]}")
    else:
        print(f"  Step 1: 跳过（特征数 <= {prefilter_k}）")
        X_pre = X
        kept_idx_pre = np.arange(X.shape[1])

    # --- 2) RFE递归消除到n_select（实质性特征选择）
    n_select_eff = min(n_select, X_pre.shape[1])
    print(f"  Step 2: RFE(RandomForest) -> {n_select_eff}特征（递归消除）")

    # 使用RandomForest作为RFE的基模型（确保确定性）
    rf_estimator = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
        bootstrap=False,  # 关闭bootstrap，确保确定性
        max_features=1.0,  # 使用全部特征，避免随机采样
        n_jobs=-1
    )

    # RFE递归特征消除
    rfe = RFE(
        estimator=rf_estimator,
        n_features_to_select=n_select_eff,
        step=1  # 每次删除1个特征（更精细）
    )

    X_sel = rfe.fit_transform(X_pre, y, sample_weight=sample_weight)
    kept_idx_rfe = np.where(rfe.support_)[0]

    print(f"    完成: {X_pre.shape[1]} -> {X_sel.shape[1]}")

    # 断言：确保精确等于目标特征数
    assert X_sel.shape[1] == n_select_eff, \
        f"RFE失败: 期望{n_select_eff}个特征，实际得到{X_sel.shape[1]}个"

    # --- 3) 映射回原始索引
    final_idx_local = kept_idx_pre[kept_idx_rfe]
    print(f"  Stage2 总计: {X.shape[1]} -> {X_sel.shape[1]}")

    return X_sel, final_idx_local, rfe


def select_features_pipeline(
    X_train, y_train,
    X_val=None, X_test=None,
    n_select=48,
    random_state=42,
    sample_weight=None
):
    """
    两阶段确定性特征选择（完全可重复）：
      Stage1：零方差过滤 + 相关性预筛(Top-256) + 共线性剔除(ρ>0.98)
      Stage2：SelectKBest(f_regression)快速预筛到160 → RFE(RandomForest)递归消除到n_select

    Args:
        X_train: (N, 4274) 训练集原始特征
        y_train: (N,) 训练集目标
        X_val: (M, 4274) 验证集（可选）
        X_test: (K, 4274) 测试集（可选）
        n_select: 目标特征数（默认48，由CLI --n-features传入）
        random_state: 随机种子（固定为42，确保RFE完全确定性）
        sample_weight: 样本权重（可选，传递给RFE）

    Returns:
        X_dict: {'train': X_train_sel, 'val': X_val_sel, 'test': X_test_sel}
        final_idx: 最终选择的原始特征索引（长度=n_select）
        rfe_obj: RFE对象（包含特征排名信息）
    """
    print(f"\n========== 两阶段特征选择 ==========")
    print(f"目标特征数: {n_select}")

    # Stage 1: 确定性过滤（零方差、共线性、弱相关）
    X_s1, idx_s1 = stage1_filter(X_train, y_train)

    # Stage 2: SelectKBest预筛 + RFE实质性特征选择
    X_s2, idx_s2_local, sel_obj = stage2_rfecv(
        X_s1, y_train,
        n_select=n_select,
        random_state=random_state,
        sample_weight=sample_weight,
        prefilter_k=160,         # 快速预筛到160（加速RFE）
        step=1,                  # RFE每次删除1个特征（更精细）
        rf_n_estimators=200,
        rf_max_depth=12
    )

    final_idx = [int(idx_s1[i]) for i in idx_s2_local]
    print(f"\n最终选择: {len(final_idx)} 特征 (原始索引)")
    X_dict = {'train': X_train[:, final_idx]}
    if X_val is not None:  X_dict['val']  = X_val[:, final_idx]
    if X_test is not None: X_dict['test'] = X_test[:, final_idx]
    print(f"========== 特征选择完成 ==========\n")
    return X_dict, final_idx, sel_obj


def apply_feature_spec(X_all, feature_spec, feature_names=None):
    """
    根据保存的特征规范应用特征选择和标准化（用于预测/推理阶段）

    重要：推理阶段必须按列名映射特征，而不是按索引！
    原因：selected_indices是相对Stage-1过滤后的索引，不是原始窗口切片的索引

    Args:
        X_all: (N, D) 原始特征矩阵（通常D=4274）
        feature_spec: dict - 从feature_spec.json加载的特征规范
        feature_names: list - 特征列名列表（长度=D），如果为None则回退到索引模式（不推荐）

    Returns:
        X_processed: (N, topk) 处理后特征矩阵
    """
    print(f"\n应用特征规范: {X_all.shape[1]} -> {feature_spec['selector']['topk']} 特征")

    # 1) 优先使用列名映射（推荐，训练/推理一致）
    if feature_names is not None and 'selected_names' in feature_spec:
        selected_names = feature_spec['selected_names']
        print(f"  使用列名映射: {len(selected_names)}个特征")

        # 验证列名存在
        missing_cols = [name for name in selected_names if name not in feature_names]
        if missing_cols:
            raise ValueError(f"特征规范中的列名在数据中不存在: {missing_cols[:5]}")

        # 按列名映射
        selected_indices = [feature_names.index(name) for name in selected_names]
        X_selected = X_all[:, selected_indices]
        print(f"  特征提取完成: {X_all.shape[1]} -> {X_selected.shape[1]}")

    # 2) 回退到索引模式（不推荐，仅用于旧版兼容）
    else:
        print(f"  警告：回退到索引模式（不推荐），推理结果可能不正确！")
        print(f"  建议：传入feature_names参数并使用selected_names进行列名映射")

        # 特征切片（feat_start:feat_start+feat_count）
        feat_start = feature_spec['feat_start']
        feat_count = feature_spec['feat_count']
        X_slice = X_all[:, feat_start:feat_start+feat_count]
        print(f"  特征切片: {X_all.shape[1]} -> {X_slice.shape[1]}")

        # 应用选择的特征索引（高风险：索引可能错位！）
        selected_indices = feature_spec['selected_indices']
        X_selected = X_slice[:, selected_indices]
        print(f"  特征选择: {X_slice.shape[1]} -> {X_selected.shape[1]}")

    # 3) 应用标准化
    if 'scaler' in feature_spec and feature_spec['scaler']['type'] == 'StandardScaler':
        mean = np.array(feature_spec['scaler']['mean'])
        scale = np.array(feature_spec['scaler']['scale'])

        # 防除零处理
        scale = np.where(scale < 1e-8, 1e-8, scale)

        X_scaled = (X_selected - mean) / scale
        print(f"  标准化完成: 均值范围[{mean.min():.3f}, {mean.max():.3f}], 标准差范围[{scale.min():.3f}, {scale.max():.3f}]")
    else:
        X_scaled = X_selected
        print(f"  跳过标准化")

    # 4) 一致性断言
    expected_features = feature_spec['selector']['topk']
    assert X_scaled.shape[1] == expected_features, \
        f"特征数不一致: 期望{expected_features}个，实际{X_scaled.shape[1]}个"

    print(f"  特征规范应用完成: {X_all.shape} -> {X_scaled.shape}")
    return X_scaled


def load_model_from_directory(model_dir):
    """
    从模型目录加载feature_spec.json和meta.json

    Args:
        model_dir: str - 模型目录路径

    Returns:
        tuple: (feature_spec, meta_info)
    """
    import json
    import os

    feature_spec_path = os.path.join(model_dir, 'feature_spec.json')
    meta_path = os.path.join(model_dir, 'meta.json')

    if not os.path.exists(feature_spec_path):
        raise FileNotFoundError(f"特征规范文件不存在: {feature_spec_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

    with open(feature_spec_path, 'r', encoding='utf-8') as f:
        feature_spec = json.load(f)

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_info = json.load(f)

    print(f"已加载模型配置: {model_dir}")
    print(f"  模型类型: {meta_info.get('model_type', 'unknown')}")
    print(f"  目标特征数: {feature_spec['selector']['topk']}")
    print(f"  实际特征数: {len(feature_spec['selected_indices'])}")

    return feature_spec, meta_info


def calculate_feature_stability(feature_sets):
    """
    计算多折特征选择的Jaccard稳定性

    Args:
        feature_sets: list of set - 每折选择的特征索引集合

    Returns:
        float: 平均Jaccard相似度（0-1，越高越稳定）
    """
    if len(feature_sets) < 2:
        return 1.0

    jaccard_scores = []
    n_sets = len(feature_sets)

    for i in range(n_sets):
        for j in range(i + 1, n_sets):
            set_i = feature_sets[i]
            set_j = feature_sets[j]

            intersection = len(set_i & set_j)
            union = len(set_i | set_j)

            if union > 0:
                jaccard = intersection / union
                jaccard_scores.append(jaccard)

    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0

    print(f"特征稳定性（Jaccard）: {avg_jaccard:.3f}")

    return avg_jaccard
