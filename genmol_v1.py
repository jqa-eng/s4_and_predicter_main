#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genmol_v1.py - 终极生成器分子批量生成与筛选脚本

目标：从 reinforced_s4/final_mol_gen_v1 载入已强化的 S4 生成器，批量生成 N=5000 条 SMILES，
用三通道预测器得到 aureus_logMIC / ecoli_logMIC / toxicity_label，转换并输出 MIC 口径到 molecules.csv

表头固定：smiles, toxicity, aureus_MIC, ecoli_MIC（注意：MIC 为 10^logMIC）
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

# S4 生成器载入
from s4dd.s4_for_denovo_design import S4forDenovoDesign
from enhanced_datasets import get_full_features


def load_xgb_regressor(model_dir):
    """
    载入XGBoost回归器模型

    Args:
        model_dir: 模型目录路径

    Returns:
        tuple: (booster, selected_indices, feat_start, feat_count, scalers)
    """
    # 读取 feature_spec.json
    with open(os.path.join(model_dir, "feature_spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)

    selected = list(map(int, spec["selected_indices"]))
    feat_start = spec["feat_start"]
    feat_count = spec["feat_count"]
    scalers = spec.get("scalers")

    # 加载 xgb 模型
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.json"))

    return booster, selected, feat_start, feat_count, scalers


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


def predict_logmic(smiles_batch, booster, selected, feat_start, feat_count, train_csv_path, scalers=None):
    """
    预测logMIC值

    Args:
        smiles_batch: SMILES列表
        booster: XGBoost模型
        selected: 选择的特征索引
        feat_start: 特征起始列
        feat_count: 特征数量
        train_csv_path: 训练CSV路径
        scalers: 标准化器

    Returns:
        logMIC预测值数组
    """
    try:
        X_full, feat_ids, _ = get_full_features(smiles_batch)
        X_full, _ = align_features(X_full, feat_ids, train_csv_path, feat_start, feat_count)
        X_sel = select_and_scale(X_full, selected, scalers)
        dmx = xgb.DMatrix(X_sel.astype(np.float32))
        return booster.predict(dmx)  # logMIC
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


def main():
    """主函数"""
    ap = argparse.ArgumentParser(description="分子生成与预测脚本")
    ap.add_argument("--model-dir", required=True,
                    help="S4 模型目录（reinforced_s4_v2）")
    ap.add_argument("--n", type=int, default=5000,
                    help="生成分子数量")
    ap.add_argument("--batch", type=int, default=1000,
                    help="生成批次大小")
    ap.add_argument("--out", type=str, default="molecules.csv",
                    help="输出文件名")
    args = ap.parse_args()

    print(f"开始从 {args.model_dir} 载入S4生成器...")

    # 1) 加载 S4 生成器
    try:
        gen = S4forDenovoDesign.from_file(args.model_dir)
        print("S4生成器载入成功")
    except Exception as e:
        print(f"S4生成器载入失败: {e}")
        return

    # 2) 载入 Aureus / Ecoli 回归器
    print("载入预测器模型...")
    try:
        aureus_dir = "models_predicter/aureus_regresser"
        ecoli_dir = "models_predicter/ecoli_regresser"

        aur_booster, aur_sel, aur_fs, aur_fc, aur_scalers = load_xgb_regressor(aureus_dir)
        eco_booster, eco_sel, eco_fs, eco_fc, eco_scalers = load_xgb_regressor(ecoli_dir)

        aur_train_csv = "datasets/standard_datasets/aureus_random_mic_datasets/aureus_train.csv"
        eco_train_csv = "datasets/standard_datasets/ecoli_mic_datasets/ecoli_train.csv"

        print("Aureus和E.coli回归器载入成功")
    except Exception as e:
        print(f"回归器载入失败: {e}")
        return

    # 3) 载入毒性分类器与其训练列名顺序
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

        print("毒性分类器载入成功")
    except Exception as e:
        print(f"毒性分类器载入失败: {e}")
        return

    # 4) 批量生成分子和预测
    print(f"\n开始生成 {args.n} 个分子...")
    rows = []
    remaining = args.n
    batch_count = 0

    progress_bar = tqdm(total=args.n, desc="生成进度")

    while remaining > 0:
        batch_count += 1
        m = min(args.batch, remaining)

        # 生成SMILES
        try:
            smiles_batch, _ = gen.design_molecules(
                n_designs=m,
                batch_size=m,
                temperature=1.0
            )
            progress_bar.update(m)
        except Exception as e:
            print(f"Warning: 第{batch_count}批次生成失败: {e}")
            remaining -= m
            continue

        # 预测logMIC
        aur_logmic = predict_logmic(smiles_batch, aur_booster, aur_sel, aur_fs, aur_fc, aur_train_csv, aur_scalers)
        eco_logmic = predict_logmic(smiles_batch, eco_booster, eco_sel, eco_fs, eco_fc, eco_train_csv, eco_scalers)

        # 预测毒性
        tox_labels = predict_toxicity_label(smiles_batch, tox_model, tox_feat_names, label_mapping)

        # 转换到 MIC（10^logMIC）
        aur_mic = np.power(10.0, aur_logmic)
        eco_mic = np.power(10.0, eco_logmic)

        # 收集结果
        for i, (smi, tox, mic_a, mic_e) in enumerate(zip(smiles_batch, tox_labels, aur_mic, eco_mic)):
            # 处理NaN值
            if np.isnan(mic_a):
                mic_a = None
            else:
                mic_a = round(float(mic_a), 4)

            if np.isnan(mic_e):
                mic_e = None
            else:
                mic_e = round(float(mic_e), 4)

            rows.append({
                "smiles": smi,
                "toxicity": tox,
                "aureus_MIC": mic_a,
                "ecoli_MIC": mic_e
            })

        remaining -= m

    progress_bar.close()

    # 5) 写出 CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 统计信息
    valid_aureus = df["aureus_MIC"].notna().sum()
    valid_ecoli = df["ecoli_MIC"].notna().sum()

    print(f"\n生成完成！")
    print(f"总分子数: {len(df)}")
    print(f"有效Aureus预测: {valid_aureus} ({valid_aureus/len(df)*100:.1f}%)")
    print(f"有效E.coli预测: {valid_ecoli} ({valid_ecoli/len(df)*100:.1f}%)")
    print(f"毒性分布: {df['toxicity'].value_counts().to_dict()}")
    print(f"结果已保存到: {args.out}")

    # 简单质量检查
    if valid_aureus > 0:
        aureus_stats = df["aureus_MIC"].describe()
        print(f"\nAureus MIC统计: min={aureus_stats['min']:.2f}, mean={aureus_stats['mean']:.2f}, max={aureus_stats['max']:.2f}")

        # 统计阈值
        low_mic_count = (df["aureus_MIC"] < 12).sum()
        print(f"Aureus MIC < 12的分子数: {low_mic_count} ({low_mic_count/valid_aureus*100:.1f}%)")


if __name__ == "__main__":
    main()