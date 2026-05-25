#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genmol_v1_fixed.py - 终极生成器分子批量生成与筛选脚本（修复版 + 严格过滤）

修复和改进内容：
1. 加载meta.json以获取z-score反变换和线性校准参数
2. 在predict_logmic中正确应用反变换和校准
3. 与mol_prediction.py和ecoli_reward_module.py的逻辑完全对齐
4. 严格化学合法性检查（RDKit sanitize=True）
5. SMILES规范化去重（canonical SMILES）
6. 三唑门：仅保留含三唑环且不含四唑环的分子

目标：从 reinforced_s4/final_mol_gen_v1 载入已强化的 S4 生成器，批量生成 N=5000 条 SMILES，
经过化学合法性检查、去重、三唑筛选后，用三通道预测器得到 aureus_logMIC / ecoli_logMIC / toxicity_label，
转换并输出 MIC 口径到 molecules.csv

表头固定：smiles, toxicity, aureus_MIC, ecoli_MIC, chem_valid（注意：MIC 为 10^logMIC）
"""

import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from tqdm import tqdm

# 新增：RDKit 用于化学合法性检查
from rdkit import Chem

# S4 生成器载入
from s4dd.s4_for_denovo_design import S4forDenovoDesign
from enhanced_datasets import get_full_features

# 三唑和四唑的SMARTS模式（来自mol_filter.py）
TRIAZOLE_PATTERNS = (
    "c1n[nH]nc1",  # 1,2,4-triazole (aromatic)
    "c1nc[nH]n1",  # 1,2,4-triazole (tautomer)
    "c1c[nH]nn1",  # 1,2,3-triazole (aromatic)
    "c1[nH]nnc1",  # 1,2,3-triazole (tautomer)
    "n1ncnc1",     # 1,2,4-triazole (deprotonated)
    "n1nccn1",     # 1,2,3-triazole (deprotonated)
    "c1nnnc1",
# === 新增：非芳香、显式双键三唑（兼容 C2=C(N)NN=N2 这种写法）===
    "C2=C(N)NN=N2",
)

TETRAZOLE_PATTERNS = (
    "c1nnn[nH]1",  # tetrazole aromatic
    "c1nn[nH]n1",  # tetrazole tautomer
    "c1[nH]nnn1",  # tetrazole tautomer 2
    "n1nnnn1",     # deprotonated tetrazole
    #以下部分为额外排除的结构
    "[*]-N1N=NC2N=NCN2N1",
    "C1=NN2C([*])N=NC2S1",
    "[*]-N1N=CN=N1",
    "[*]-C1N=NN=N1",
    "[*]-C1N=NC2N=NCN12",
    #以下去除重复的异吲哚啉酮基本骨架
    'C1(=O)Nc2ccccc2C1',
    'c1ccc2c(c1)CN(C2=O)',
    'c1ccc2c(c1)C(*)N(*)C2=O',
    #以下为实验显示不希望出现的结构
    '[CX2]#[CX2]'
)

# === 13ap-like 结构（soft preference，用于候选优先级，不是硬约束） ===
FAVOR_13AP_SMARTS = Chem.MolFromSmarts(
    "O=C(c1ccc(Br)cc1)c2nn[nH]c2N(Cc3ccccc3)Cc4ccccc4"
)

# === 实验成功结构的必要子结构（REQUIRE，用于收缩生成空间） ===
# 第二轮：使用“关键连接关系”的结构约束（溴苯甲酰基）
REQUIRE_SMARTS = (
    # Chem.MolFromSmarts("O=C(c1ccccc1)c2nn[nH]c2N"),
    # Chem.MolFromSmarts("c1nnnc1"),
)

# 预编译SMARTS模式
_TRIAZOLE_SMARTS = tuple(
    smarts
    for smarts in (Chem.MolFromSmarts(pattern) for pattern in TRIAZOLE_PATTERNS)
    if smarts is not None
)
_TETRAZOLE_SMARTS = tuple(
    smarts
    for smarts in (Chem.MolFromSmarts(pattern) for pattern in TETRAZOLE_PATTERNS)
    if smarts is not None
)


def try_parse_and_canonical(smiles: str):
    """
    sanitize=True 解析；成功则返回 (mol, 规范SMILES)，失败返回 (None, None)。

    Args:
        smiles: SMILES字符串

    Returns:
        tuple: (mol对象, 规范SMILES) 或 (None, None)
    """
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=True)  # 等价于显式 SanitizeMol
        if m is None:
            return None, None
        return m, Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None, None


def is_triazole_not_tetrazole(mol) -> bool:
    """
    检查分子是否包含三唑环且不含四唑环

    Args:
        mol: RDKit mol对象

    Returns:
        bool: 是否为符合条件的三唑分子
    """
    if mol is None:
        return False

    # 检查是否含有三唑环
    has_triazole = any(mol.HasSubstructMatch(pattern) for pattern in _TRIAZOLE_SMARTS)
    if not has_triazole:
        return False

    # 检查是否含有四唑环（需要排除）
    has_tetrazole = any(mol.HasSubstructMatch(pattern) for pattern in _TETRAZOLE_SMARTS)
    return not has_tetrazole


def load_xgb_regressor(model_dir):
    """
    载入XGBoost回归器模型（修复版：同时加载meta.json）

    Args:
        model_dir: 模型目录路径

    Returns:
        tuple: (booster, selected_indices, feat_start, feat_count, scalers, meta)
    """
    # 读取 feature_spec.json
    with open(os.path.join(model_dir, "feature_spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)

    selected = list(map(int, spec["selected_indices"]))
    feat_start = spec["feat_start"]
    feat_count = spec["feat_count"]
    scalers = spec.get("scalers")

    # 加载 meta.json（用于z-score反变换和线性校准）
    meta_path = os.path.join(model_dir, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  Loaded meta.json from {model_dir}")
    else:
        print(f"  WARNING: meta.json not found in {model_dir}, will skip z-score and calibration transforms")

    # 加载 xgb 模型
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.json"))

    return booster, selected, feat_start, feat_count, scalers, meta


def align_features(X_full, feature_ids, train_csv_path, feat_start, feat_count):
    """
    特征对齐：按训练期列名顺序重排特征

    Args:
        X_full: 完整特征矩阵
        feature_ids: 特征ID列表
        train_csv_path: 训练CSV路径
        feat_start: 特征起始列
        feat_count: 特征数量

    Returns:
        tuple: (对齐后特征矩阵, 训练特征名列表)
    """
    # 读取训练期列名（只读表头）
    train_cols = pd.read_csv(train_csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    train_feat_names = train_cols[feat_start: feat_start + feat_count]

    # 构建重排索引
    feat_idx = {nm: i for i, nm in enumerate(feature_ids)}
    missing = [nm for nm in train_feat_names if nm not in feat_idx]
    if missing:
        raise ValueError(f"feature mismatch: missing={missing[:5]} ... total={len(missing)}")

    reorder = [feat_idx[nm] for nm in train_feat_names]
    X_full = X_full[:, reorder]

    return X_full, train_feat_names


def select_and_scale(X_full, selected, scalers=None):
    """
    特征选择和标准化

    Args:
        X_full: 完整特征矩阵
        selected: 选择的特征索引
        scalers: 标准化器字典

    Returns:
        选择并标准化后的特征矩阵
    """
    X_sel = X_full[:, selected]

    if scalers:
        for i, orig_idx in enumerate(selected):
            sc = scalers.get(str(orig_idx))
            if sc:
                X_sel[:, i] = (X_sel[:, i] - sc["mean"]) / (sc["std"] + 1e-8)

    return X_sel


def predict_logmic(smiles_batch, booster, selected, feat_start, feat_count, train_csv_path, scalers=None, meta=None):
    """
    预测logMIC值（修复版：应用z-score反变换和线性校准）

    Args:
        smiles_batch: SMILES列表
        booster: XGBoost模型
        selected: 选择的特征索引
        feat_start: 特征起始列
        feat_count: 特征数量
        train_csv_path: 训练CSV路径
        scalers: 标准化器
        meta: 元数据（包含z-score和calibration参数）

    Returns:
        logMIC预测值数组（已应用反变换和校准）
    """
    try:
        X_full, feat_ids, _ = get_full_features(smiles_batch)
        X_full, _ = align_features(X_full, feat_ids, train_csv_path, feat_start, feat_count)
        X_sel = select_and_scale(X_full, selected, scalers)
        dmx = xgb.DMatrix(X_sel.astype(np.float32))

        # 原始预测（z-score空间）
        predictions_z = booster.predict(dmx)

        # 应用反z变换和线性校准（与mol_prediction.py和reward模块完全对齐）
        if meta is not None:
            # 步骤1：反z变换 (z-score -> 原始logMIC尺度)
            target_norm = meta.get('target_normalization', {})
            if target_norm.get('enabled', False):
                y_mean = target_norm.get('y_mean', 0.0)
                y_std = target_norm.get('y_std', 1.0)
                predictions_logMIC = predictions_z * y_std + y_mean
            else:
                predictions_logMIC = predictions_z

            # 步骤2：线性校准 (使用验证集的校准参数)
            calibration = meta.get('calibration', {})
            if calibration.get('enabled', False):
                a_val = calibration.get('a_val', 1.0)
                b_val = calibration.get('b_val', 0.0)
                predictions_calibrated = a_val * predictions_logMIC + b_val
                return predictions_calibrated
            else:
                return predictions_logMIC
        else:
            # 如果没有meta.json，返回原始预测（兼容旧模型）
            print("WARNING: No meta.json found, returning raw predictions without transforms")
            return predictions_z

    except Exception as e:
        print(f"Warning: logMIC prediction failed for batch: {e}")
        return np.full(len(smiles_batch), np.nan)


def predict_toxicity_label(smiles_batch, tox_model, tox_feat_names, label_mapping):
    """
    预测毒性标签

    Args:
        smiles_batch: SMILES列表
        tox_model: 毒性分类模型
        tox_feat_names: 训练期特征名
        label_mapping: 标签映射字典

    Returns:
        毒性标签列表
    """
    try:
        # 直接用 get_full_features + 列名对齐到 tox_feat_names
        X_full, feat_ids, _ = get_full_features(smiles_batch)
        feat_idx = {nm: i for i, nm in enumerate(feat_ids)}
        reorder = [feat_idx[nm] for nm in tox_feat_names]
        X = X_full[:, reorder]

        proba = tox_model.predict_proba(X)
        idx = np.argmax(proba, axis=1)

        # 用 label_mapping 还原中文标签
        return [label_mapping[str(i)] for i in idx]
    except Exception as e:
        print(f"Warning: toxicity prediction failed for batch: {e}")
        return ["未知"] * len(smiles_batch)


def predict_toxicity_probs(smiles_batch, tox_model, tox_feat_names, label_mapping):
    """
    返回规范列序（中毒, 低毒, 微毒, 高毒）的概率矩阵 shape=(N,4)

    Args:
        smiles_batch: SMILES列表
        tox_model: 毒性分类模型
        tox_feat_names: 训练期特征名
        label_mapping: 标签映射字典

    Returns:
        规范顺序的概率矩阵 (N, 4)
    """
    try:
        # 1) 同 genmol 中一致：特征生成 + 按训练列名重排
        X_full, feat_ids, _ = get_full_features(smiles_batch)
        feat_idx = {nm: i for i, nm in enumerate(feat_ids)}
        reorder = [feat_idx[nm] for nm in tox_feat_names]
        X = X_full[:, reorder]

        # 2) 先得到与 model.classes_ 对齐的概率
        proba_model = tox_model.predict_proba(X)  # 列序 = model.classes_

        # 3) 建立 model.classes_ 到中文标签的映射
        classes = list(tox_model.classes_)  # e.g. [0,1,2,3] 或任意顺序
        label_order = [label_mapping[str(cls)] for cls in classes]  # 与列一一对应

        # 4) 重排到规范顺序（中毒, 低毒, 微毒, 高毒）
        canonical = ["中毒", "低毒", "微毒", "高毒"]
        idx_map = [label_order.index(name) for name in canonical]
        proba_canon = proba_model[:, idx_map]
        return proba_canon  # N x 4
    except Exception as e:
        print(f"Warning: toxicity prob prediction failed for batch: {e}")
        return None


def main():
    """主函数"""
    ap = argparse.ArgumentParser(description="分子生成与预测脚本（修复版）")
    ap.add_argument("--model-dir", required=True,
                    help="S4 模型目录（reinforced_s4_v2）")
    ap.add_argument("--n", type=int, default=5000,
                    help="生成分子数量")
    ap.add_argument("--batch", type=int, default=1000,
                    help="生成批次大小")
    ap.add_argument("--out", type=str, default="molecules.csv",
                    help="输出文件名")
    ap.add_argument("--required-prefix", type=str, default="c1n[nH]nc1",
                    help="骨架/前缀约束SMILES（传给 S4 的 required_prefix_smiles）")
    args = ap.parse_args()

    print("=" * 70)
    print("分子生成与预测脚本（修复版 - 包含z-score反变换和线性校准）")
    print("=" * 70)
    print(f"S4模型目录: {args.model_dir}")
    print(f"生成数量: {args.n}")
    print(f"输出文件: {args.out}")
    print(f"前缀约束: {args.required_prefix}")
    print()

    # 1) 加载 S4 生成器
    print(f"[1/4] 载入S4生成器...")
    try:
        gen = S4forDenovoDesign.from_file(args.model_dir)
        print("  S4生成器载入成功")
    except Exception as e:
        print(f"  ERROR: S4生成器载入失败: {e}")
        return

    # 2) 载入 Aureus / Ecoli 回归器（修复版：同时加载meta.json）
    print("\n[2/4] 载入预测器模型...")
    try:
        aureus_dir = "models_predicter/aureus_regresser"
        ecoli_dir = "models_predicter/ecoli_regresser"

        print(f"  Loading Aureus regressor from {aureus_dir}...")
        aur_booster, aur_sel, aur_fs, aur_fc, aur_scalers, aur_meta = load_xgb_regressor(aureus_dir)

        print(f"  Loading E.coli regressor from {ecoli_dir}...")
        eco_booster, eco_sel, eco_fs, eco_fc, eco_scalers, eco_meta = load_xgb_regressor(ecoli_dir)

        aur_train_csv = "datasets/standard_datasets/aureus_random_mic_datasets/aureus_train.csv"
        eco_train_csv = "datasets/standard_datasets/ecoli_mic_datasets/ecoli_train.csv"

        print("  Aureus和E.coli回归器载入成功")

        # 打印z-score和校准参数（验证）
        if aur_meta:
            aur_norm = aur_meta.get('target_normalization', {})
            aur_calib = aur_meta.get('calibration', {})
            print(f"  Aureus: z-score enabled={aur_norm.get('enabled')}, calibration enabled={aur_calib.get('enabled')}")

        if eco_meta:
            eco_norm = eco_meta.get('target_normalization', {})
            eco_calib = eco_meta.get('calibration', {})
            print(f"  E.coli: z-score enabled={eco_norm.get('enabled')}, calibration enabled={eco_calib.get('enabled')}")

    except Exception as e:
        print(f"  ERROR: 回归器载入失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3) 载入毒性分类器与其训练列名顺序
    print("\n[3/4] 载入毒性分类器...")
    try:
        tox_model = joblib.load("models_predicter/toxicity_classifier_4274d/toxicity_classifier.pkl")

        with open("models_predicter/toxicity_classifier_4274d/meta.json", "r", encoding="utf-8") as f:
            tox_meta = json.load(f)

        tox_feat_names = tox_meta["features"]["feature_names"]

        # 尝试从label_mapping.json或meta.json获取标签映射
        try:
            with open("models_predicter/toxicity_classifier_4274d/label_mapping.json", "r", encoding="utf-8") as f:
                label_mapping = json.load(f)
        except:
            label_mapping = tox_meta.get("label_mapping", {"0": "低毒", "1": "微毒", "2": "中毒", "3": "高毒"})

        print("  毒性分类器载入成功")
    except Exception as e:
        print(f"  ERROR: 毒性分类器载入失败: {e}")
        return

    # 4) 批量生成分子和预测（新版：严格化学合法过滤 + 规范去重 + 三唑门）
    print(f"\n[4/4] 开始生成 {args.n} 个分子（单批 {args.batch}，严格化学合法过滤 + 三唑门）...")
    rows = []
    remaining = args.n
    seen = set()   # 规范 SMILES 去重
    batch_idx = 0

    pbar = tqdm(total=args.n, desc="收集进度（化学合法 ∧ 去重 ∧ 三唑）")

    while remaining > 0:
        batch_idx += 1
        # target = min(args.batch, remaining)
        target=args.batch

        # A) 仅生成 target 条（不超采）
        try:
            smiles_batch, _ = gen.design_molecules(
                n_designs=target,
                batch_size=target,
                temperature=2.0,
                required_prefix_smiles=args.required_prefix  # ★ 关键：把前缀传入生成器
            )
        except Exception as e:
            print(f"[warn] 第 {batch_idx} 批生成失败：{e}")
            continue

        # B) 严格化学合法 + 规范去重 + 三唑门 + soft preference（13ap-like）
        preferred_smiles = []  # 优先桶：13ap-like
        other_smiles = []      # 普通桶：其它合格三唑

        for smi in smiles_batch:
            m, csmi = try_parse_and_canonical(smi)
            if m is None:
                continue              # 解析或消毒失败
            if csmi in seen:
                continue              # 规范去重
            if not is_triazole_not_tetrazole(m):
                continue              # 不符合三唑/排除门
            # REQUIRE：必须包含实验成功分子的关键子结构
            if REQUIRE_SMARTS:
                if not all(
                    s is None or m.HasSubstructMatch(s)
                    for s in REQUIRE_SMARTS
                ):
                    continue

            seen.add(csmi)

            # soft preference：13ap-like 结构优先进入候选
            if FAVOR_13AP_SMARTS is not None and m.HasSubstructMatch(FAVOR_13AP_SMARTS):
                preferred_smiles.append(csmi)
            else:
                other_smiles.append(csmi)

        # 合并：优先收集 13ap-like，其余顺序不变
        valid_smiles = preferred_smiles + other_smiles

        accepted = len(valid_smiles)
        if accepted == 0:
            # 本批无有效分子，继续下一批
            continue

        # C) 只对"化学合法 ∧ 去重"的分子调用三预测器
        aur_logmic = predict_logmic(valid_smiles, aur_booster, aur_sel, aur_fs, aur_fc, aur_train_csv, aur_scalers, aur_meta)
        eco_logmic = predict_logmic(valid_smiles, eco_booster, eco_sel, eco_fs, eco_fc, eco_train_csv, eco_scalers, eco_meta)

        # 计算概率（规范列序：中毒, 低毒, 微毒, 高毒）
        tox_probs = predict_toxicity_probs(valid_smiles, tox_model, tox_feat_names, label_mapping)
        if tox_probs is None:
            tox_probs = np.full((len(valid_smiles), 4), np.nan)

        # p_safe = P(低毒)+P(微毒) = [:,1]+[:,2]
        p_safe = tox_probs[:, 1] + tox_probs[:, 2]

        # 映射到 MIC（口径：logMIC → 10**）
        aur_mic = np.power(10.0, aur_logmic)
        eco_mic = np.power(10.0, eco_logmic)

        for smi, ps, mic_a, mic_e in zip(valid_smiles, p_safe, aur_mic, eco_mic):
            rows.append({
                "smiles": smi,
                "toxicity": None if np.isnan(ps) else round(float(ps), 4),  # 改为写入安全概率
                "aureus_MIC": None if np.isnan(mic_a) else round(float(mic_a), 4),
                "ecoli_MIC":  None if np.isnan(mic_e) else round(float(mic_e), 4),
                "chem_valid": True  # 标记这批都是严格化学合法
            })

        remaining -= accepted
        pbar.update(accepted)

    pbar.close()

    # 5) 写出 CSV（全量）
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 5.1) 新增：按三条件过滤得到命中集合
    mask = (
        df["aureus_MIC"].notna() & (df["aureus_MIC"] < 12) &
        df["ecoli_MIC"].notna() & (df["ecoli_MIC"] < 12) &
        df["toxicity"].notna() & (df["toxicity"] >= 0.70)  # toxicity列现在是p_safe
    )
    df_hit = df[mask].copy()

    # 建议额外输出"过滤后"的文件（不覆盖全量）
    out_hit = os.path.splitext(args.out)[0] + "_filtered.csv"
    df_hit.to_csv(out_hit, index=False, encoding="utf-8-sig")

    # 统计信息
    valid_aureus = df["aureus_MIC"].notna().sum()
    valid_ecoli = df["ecoli_MIC"].notna().sum()

    print("\n" + "=" * 70)
    print("生成完成！")
    print("=" * 70)
    print(f"总分子数: {len(df)}")
    print(f"有效Aureus预测: {valid_aureus} ({valid_aureus/len(df)*100:.1f}%)")
    print(f"有效E.coli预测: {valid_ecoli} ({valid_ecoli/len(df)*100:.1f}%)")
    print(f"\n满足三条件的分子数: {len(df_hit)} / {len(df)}  ({mask.mean()*100:.1f}%)")
    print(f"过滤后结果已保存到: {out_hit}")
    print(f"\n毒性安全概率分布（p_safe = P(低毒)+P(微毒)）:")
    # 分段统计p_safe分布
    p_safe_valid = df["toxicity"].dropna()
    if len(p_safe_valid) > 0:
        print(f"  Mean: {p_safe_valid.mean():.3f}")
        print(f"  Median: {p_safe_valid.median():.3f}")
        print(f"  p_safe >= 0.70: {(p_safe_valid >= 0.70).sum()} ({(p_safe_valid >= 0.70).mean()*100:.1f}%)")
        print(f"  p_safe >= 0.80: {(p_safe_valid >= 0.80).sum()} ({(p_safe_valid >= 0.80).mean()*100:.1f}%)")
        print(f"  p_safe >= 0.90: {(p_safe_valid >= 0.90).sum()} ({(p_safe_valid >= 0.90).mean()*100:.1f}%)")
    print(f"\n结果已保存到: {args.out}")

    # 简单质量检查
    if valid_aureus > 0:
        aureus_stats = df["aureus_MIC"].describe()
        print(f"\nAureus MIC统计:")
        print(f"  Min: {aureus_stats['min']:.2f}")
        print(f"  Mean: {aureus_stats['mean']:.2f}")
        print(f"  Median: {aureus_stats['50%']:.2f}")
        print(f"  Max: {aureus_stats['max']:.2f}")

        # 统计阈值
        low_mic_count = (df["aureus_MIC"] < 12).sum()
        very_low_mic_count = (df["aureus_MIC"] < 3).sum()
        print(f"\n活性分析（Aureus）:")
        print(f"  MIC < 12的分子数: {low_mic_count} ({low_mic_count/valid_aureus*100:.1f}%)")
        print(f"  MIC < 3的分子数: {very_low_mic_count} ({very_low_mic_count/valid_aureus*100:.1f}%)")

    if valid_ecoli > 0:
        ecoli_stats = df["ecoli_MIC"].describe()
        print(f"\nE.coli MIC统计:")
        print(f"  Min: {ecoli_stats['min']:.2f}")
        print(f"  Mean: {ecoli_stats['mean']:.2f}")
        print(f"  Median: {ecoli_stats['50%']:.2f}")
        print(f"  Max: {ecoli_stats['max']:.2f}")

        # 统计阈值
        low_mic_count_eco = (df["ecoli_MIC"] < 12).sum()
        very_low_mic_count_eco = (df["ecoli_MIC"] < 3).sum()
        print(f"\n活性分析（E.coli）:")
        print(f"  MIC < 12的分子数: {low_mic_count_eco} ({low_mic_count_eco/valid_ecoli*100:.1f}%)")
        print(f"  MIC < 3的分子数: {very_low_mic_count_eco} ({very_low_mic_count_eco/valid_ecoli*100:.1f}%)")

    # 新增：化学合法率统计
    chem_valid_rate = (pd.Series([r.get("chem_valid", False) for r in rows]).mean()
                       if rows else 0.0)
    print(f"\n严格化学合法率（RDKit Sanitize）: {chem_valid_rate*100:.1f}% （本脚本已强制筛选，应为 100%）")

    # 新增：三条件命中统计（毒性用 p_safe >= 0.70）
    HIT = (df["toxicity"].notna()) & (df["toxicity"] >= 0.70) \
          & (df["aureus_MIC"].notna()) & (df["aureus_MIC"] < 12) \
          & (df["ecoli_MIC"].notna()) & (df["ecoli_MIC"] < 12)
    print(f"三条件命中数（p_safe≥0.70 & 双菌MIC<12）: {HIT.sum()} / {len(df)}  ({HIT.mean()*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
