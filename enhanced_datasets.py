#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_datasets.py - 就地追加特征脚本（基于 guidance.md）

功能：
1. 对任意 CSV 就地追加固定 426 维特征（11 描述符 + ECFP4 2048 + MACCS 167 + RDK 2048）
2. 支持三种输入形态：完整、缺 logMIC、仅 smiles
3. 幂等：默认阻止重复追加，可选覆盖重写
4. 保持固定维度：不删除全 0 列
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger, __version__ as rdkit_version
from rdkit.Chem import Descriptors, rdMolDescriptors as rdm, MACCSkeys
RDLogger.DisableLog('rdApp.*')

# ============ 特征前缀定义 ============
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotBonds", "RingCount", "Chi0v", "Chi1v", "Kappa1", "BalabanJ"
]

ECFP4_PREFIX = "ECFP4_"
MACCS_PREFIX = "MACCS_"
RDK_PREFIX = "RDK_"

ECFP4_BITS = 2048
MACCS_BITS = 167
RDK_BITS = 2048

# ============ 特征构建函数 ============

def build_descriptors(smiles_list):
    """构建 11 个连续描述符（SMILES 解析失败 → NaN）"""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            rows.append({name: np.nan for name in DESCRIPTOR_NAMES})
            continue

        row = {}
        try:
            row["MolWt"] = Descriptors.MolWt(mol)
        except:
            row["MolWt"] = np.nan

        try:
            row["MolLogP"] = Descriptors.MolLogP(mol)
        except:
            row["MolLogP"] = np.nan

        try:
            row["TPSA"] = rdm.CalcTPSA(mol)
        except:
            row["TPSA"] = np.nan

        try:
            row["NumHDonors"] = rdm.CalcNumHBD(mol)
        except:
            row["NumHDonors"] = np.nan

        try:
            row["NumHAcceptors"] = rdm.CalcNumHBA(mol)
        except:
            row["NumHAcceptors"] = np.nan

        try:
            row["NumRotBonds"] = rdm.CalcNumRotatableBonds(mol)
        except:
            row["NumRotBonds"] = np.nan

        try:
            row["RingCount"] = rdm.CalcNumRings(mol)
        except:
            row["RingCount"] = np.nan

        try:
            row["Chi0v"] = Descriptors.Chi0v(mol)
        except:
            row["Chi0v"] = np.nan

        try:
            row["Chi1v"] = Descriptors.Chi1v(mol)
        except:
            row["Chi1v"] = np.nan

        try:
            row["Kappa1"] = Descriptors.Kappa1(mol)
        except:
            row["Kappa1"] = np.nan

        try:
            row["BalabanJ"] = Descriptors.BalabanJ(mol)
        except:
            row["BalabanJ"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
    return df.replace([np.inf, -np.inf], np.nan)


def build_ecfp4_counts_hashed(smiles_list, radius=2, hash_mod=2048):
    """构建 ECFP4 计数向量（手动哈希到 2048 维）"""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            # 全 0
            rows.append({f"{ECFP4_PREFIX}{i}": 0 for i in range(hash_mod)})
            continue

        try:
            # 获取计数向量（不是 bit 向量）
            ecfp_sparse = rdm.GetMorganFingerprint(mol, radius=radius)
            ecfp_counts_raw = ecfp_sparse.GetNonzeroElements()  # {feature_id: count}

            # 手动哈希到 hash_mod 维
            ecfp_counts_hashed = {}
            for k, c in ecfp_counts_raw.items():
                idx = int(k) % hash_mod
                ecfp_counts_hashed[idx] = ecfp_counts_hashed.get(idx, 0) + int(c)

            row = {f"{ECFP4_PREFIX}{i}": ecfp_counts_hashed.get(i, 0) for i in range(hash_mod)}
            rows.append(row)
        except:
            rows.append({f"{ECFP4_PREFIX}{i}": 0 for i in range(hash_mod)})

    df = pd.DataFrame(rows).fillna(0).astype(int)
    return df


def build_maccs_bits(smiles_list, bits=167, start_index=0):
    """构建 MACCS Keys（167 位）"""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            rows.append({f"{MACCS_PREFIX}{i}": 0 for i in range(bits)})
            continue

        try:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            row = {f"{MACCS_PREFIX}{i}": int(maccs.GetBit(i + start_index)) for i in range(bits)}
            rows.append(row)
        except:
            rows.append({f"{MACCS_PREFIX}{i}": 0 for i in range(bits)})

    df = pd.DataFrame(rows).fillna(0).astype(int)
    return df


def build_rdk_bits(smiles_list, fp_size=2048, min_path=1, max_path=7):
    """构建 RDKFingerprint（2048 位）"""
    from rdkit.Chem import RDKFingerprint

    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            rows.append({f"{RDK_PREFIX}{i}": 0 for i in range(fp_size)})
            continue

        try:
            fp = RDKFingerprint(mol, fpSize=fp_size, minPath=min_path, maxPath=max_path)
            row = {f"{RDK_PREFIX}{i}": int(fp.GetBit(i)) for i in range(fp_size)}
            rows.append(row)
        except:
            rows.append({f"{RDK_PREFIX}{i}": 0 for i in range(fp_size)})

    df = pd.DataFrame(rows).fillna(0).astype(int)
    return df


# ============ 幂等检查与特征追加 ============

def check_already_enhanced(df):
    """检查是否已增强（是否存在任一目标特征前缀）"""
    cols = df.columns.tolist()
    for col in cols:
        if col in DESCRIPTOR_NAMES:
            return True
        if col.startswith(ECFP4_PREFIX) or col.startswith(MACCS_PREFIX) or col.startswith(RDK_PREFIX):
            return True
    return False


def remove_existing_features(df):
    """删除已有的目标特征列（用于 --overwrite-features）"""
    cols_to_remove = []
    for col in df.columns:
        if col in DESCRIPTOR_NAMES:
            cols_to_remove.append(col)
        elif col.startswith(ECFP4_PREFIX) or col.startswith(MACCS_PREFIX) or col.startswith(RDK_PREFIX):
            cols_to_remove.append(col)

    if cols_to_remove:
        print(f"  删除已有特征列：{len(cols_to_remove)} 列")
        df = df.drop(columns=cols_to_remove)
    return df


def add_logmic_if_missing(df, task='aureus'):
    """若存在 {task}_MIC 但缺 {task}_logMIC，则自动补全"""
    mic_col = f'{task}_MIC'
    logmic_col = f'{task}_logMIC'

    if mic_col in df.columns and logmic_col not in df.columns:
        print(f"  检测到 {mic_col} 列，自动补全 {logmic_col}")
        df[logmic_col] = df[mic_col].apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)

    return df


def enhance_single_file(file_path, overwrite=False, task='aureus'):
    """就地增强单个 CSV 文件"""
    print(f"\n处理文件: {file_path}")

    # 读取
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    print(f"  原始维度: {df.shape}")

    # 幂等检查
    if check_already_enhanced(df):
        if not overwrite:
            print(f"   文件已增强，跳过（使用 --overwrite-features 强制重写）")
            return
        else:
            print(f"  检测到已有特征，执行覆盖重写...")
            df = remove_existing_features(df)

    # 补全 logMIC（若需要）
    df = add_logmic_if_missing(df, task=task)

    # 校验 smiles 列
    if 'smiles' not in df.columns:
        print(f"  错误：缺少 'smiles' 列")
        return

    smiles_list = df['smiles'].tolist()
    print(f"  样本数: {len(smiles_list)}")

    # 构建特征块
    print(f"  构建特征块...")
    print(f"    - 描述符（11 维）...", end='', flush=True)
    df_desc = build_descriptors(smiles_list)
    print(f" 完成")

    print(f"    - ECFP4 计数哈希（{ECFP4_BITS} 维）...", end='', flush=True)
    df_ecfp4 = build_ecfp4_counts_hashed(smiles_list, radius=2, hash_mod=ECFP4_BITS)
    print(f" 完成")

    print(f"    - MACCS Keys（{MACCS_BITS} 维）...", end='', flush=True)
    df_maccs = build_maccs_bits(smiles_list, bits=MACCS_BITS, start_index=0)
    print(f" 完成")

    print(f"    - RDK Fingerprint（{RDK_BITS} 维）...", end='', flush=True)
    df_rdk = build_rdk_bits(smiles_list, fp_size=RDK_BITS, min_path=1, max_path=7)
    print(f" 完成")

    # 横向拼接
    total_features = 11 + ECFP4_BITS + MACCS_BITS + RDK_BITS
    print(f"  拼接特征块（总计 {total_features} 维）...")
    df_enhanced = pd.concat([df, df_desc, df_ecfp4, df_maccs, df_rdk], axis=1)

    # 验证
    added_cols = df_enhanced.shape[1] - df.shape[1]
    print(f"  增强后维度: {df_enhanced.shape}")
    print(f"  新增列数: {added_cols}（预期 {total_features}）")

    if added_cols != total_features:
        print(f"   警告：新增列数不匹配！")

    # 覆盖写回
    print(f"  覆盖写回原文件...")
    df_enhanced.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"  完成")


# ============ CLI 入口 ============

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="就地追加特征（基于 guidance.md）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单文件增强
  python enhanced_datasets.py --in train.csv

  # 批量增强
  python enhanced_datasets.py --in train.csv --in val.csv --in test.csv

  # 覆盖重写
  python enhanced_datasets.py --in train.csv --overwrite-features
        """
    )
    parser.add_argument('--in', dest='input_files', action='append', required=True,
                        help='输入 CSV 文件路径（可多次指定）')
    parser.add_argument('--overwrite-features', action='store_true',
                        help='允许覆盖已有特征（删除旧特征后重写）')
    parser.add_argument('--task', default='aureus',
                        help='任务名（用于 logMIC 补全，默认 aureus）')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("就地追加特征脚本（enhanced_datasets.py）")
    print("=" * 60)
    print(f"RDKit 版本: {rdkit_version}")
    print(f"任务名: {args.task}")
    print(f"覆盖模式: {'是' if args.overwrite_features else '否'}")
    print(f"文件数量: {len(args.input_files)}")

    # 处理每个文件
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"\n 文件不存在，跳过: {file_path}")
            continue

        enhance_single_file(file_path, overwrite=args.overwrite_features, task=args.task)

    print("\n" + "=" * 60)
    print("全部文件处理完成！")
    print("=" * 60)
    return 0


# ============ 奖励模块稳定API ============

def get_full_features(smiles_list):
    """
    为奖励模块提供的稳定特征生成API

    Args:
        smiles_list: List[str] - SMILES分子列表

    Returns:
        X: np.ndarray (N, 4274) - 特征矩阵
        feature_ids: List[str] - 特征ID列表
        meta: dict - 管线元信息
    """
    if not isinstance(smiles_list, list):
        smiles_list = [smiles_list]

    # 构建各特征块（与训练时完全相同的顺序和参数）
    df_desc = build_descriptors(smiles_list)
    df_ecfp4 = build_ecfp4_counts_hashed(smiles_list, radius=2, hash_mod=ECFP4_BITS)
    df_maccs = build_maccs_bits(smiles_list, bits=MACCS_BITS, start_index=0)
    df_rdk = build_rdk_bits(smiles_list, fp_size=RDK_BITS, min_path=1, max_path=7)

    # 按训练时的顺序拼接特征
    df_full = pd.concat([df_desc, df_ecfp4, df_maccs, df_rdk], axis=1)
    X = df_full.values

    # 构建特征ID列表（按顺序）
    feature_ids = (
        DESCRIPTOR_NAMES +
        [f"{ECFP4_PREFIX}{i}" for i in range(ECFP4_BITS)] +
        [f"{MACCS_PREFIX}{i}" for i in range(MACCS_BITS)] +
        [f"{RDK_PREFIX}{i}" for i in range(RDK_BITS)]
    )

    # 元信息
    meta = {
        "rdkit_version": rdkit_version,
        "total_features": len(feature_ids),
        "descriptors": {"count": 11, "names": DESCRIPTOR_NAMES},
        "ecfp4": {"radius": 2, "bits": ECFP4_BITS, "type": "count_hashed"},
        "maccs": {"bits": MACCS_BITS, "start_index": 0},
        "rdk": {"fp_size": RDK_BITS, "min_path": 1, "max_path": 7}
    }

    return X, feature_ids, meta


def get_pipeline_signature():
    """
    返回特征管线的稳定签名，用于版本对齐检查

    Returns:
        str - 管线签名字符串
    """
    import hashlib

    # 构建签名字符串（包含所有关键参数）
    sig_data = {
        "rdkit_version": rdkit_version,
        "descriptors": DESCRIPTOR_NAMES,
        "ecfp4": {"prefix": ECFP4_PREFIX, "radius": 2, "bits": ECFP4_BITS, "type": "count_hashed"},
        "maccs": {"prefix": MACCS_PREFIX, "bits": MACCS_BITS, "start_index": 0},
        "rdk": {"prefix": RDK_PREFIX, "fp_size": RDK_BITS, "min_path": 1, "max_path": 7},
        "feature_order": ["desc", "ecfp4", "maccs", "rdk"],
        "total_features": 11 + ECFP4_BITS + MACCS_BITS + RDK_BITS
    }

    # 生成SHA256签名
    sig_str = str(sorted(sig_data.items()))
    signature = hashlib.sha256(sig_str.encode('utf-8')).hexdigest()[:16]

    return f"enhanced_v1_{signature}"


if __name__ == "__main__":
    sys.exit(main())
