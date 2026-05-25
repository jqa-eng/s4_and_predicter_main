#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mol_filter.py - 分子筛选脚本

筛选条件：
1. 是有效分子，SMILES 可以被 RDKit 解析；
2. 是三唑分子，使用 SMARTS 匹配三唑环，并排除四唑环；
3. aureus_MIC < 20。

说明：
- toxicity 不参与筛选，仅作为预测结果保留；
- ecoli_MIC 不参与筛选，仅作为预测结果保留；
- aureus_MIC 使用原始 MIC 数值，不进行 log 转换。

从 molecules.csv 读取数据，筛选后写入 molecules_final.csv
"""

import argparse
from typing import Any

import pandas as pd

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore
except ImportError:
    Chem = None
    rdMolStandardize = None


# 三唑和四唑的 SMARTS 模式（来自 run_rl_v1.py）
TRIAZOLE_PATTERNS = (
    "c1n[nH]nc1",  # 1,2,4-triazole (aromatic)
    "c1nc[nH]n1",  # 1,2,4-triazole (tautomer)
    "c1c[nH]nn1",  # 1,2,3-triazole (aromatic)
    "c1[nH]nnc1",  # 1,2,3-triazole (tautomer)
    "n1ncnc1",     # 1,2,4-triazole (deprotonated)
    "n1nccn1",     # 1,2,3-triazole (deprotonated)
)

TETRAZOLE_PATTERNS = (
    "c1nnn[nH]1",  # tetrazole aromatic
    "c1nn[nH]n1",  # tetrazole tautomer
    "c1[nH]nnn1",  # tetrazole tautomer 2
    "n1nnnn1",     # deprotonated tetrazole
    # 以下部分为额外排除的结构
    "[*]-N1N=NC2N=NCN2N1",
    "C1=NN2C([*])N=NC2S1",
    "[*]-N1N=CN=N1",
    "[*]-C1N=NN=N1",
    "[*]-C1N=NC2N=NCN12",
)

if Chem is not None:
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
else:
    _TRIAZOLE_SMARTS = ()
    _TETRAZOLE_SMARTS = ()


def is_valid_smiles(smiles: str) -> bool:
    """检查 SMILES 是否可被 RDKit 解析。"""
    if Chem is None:
        raise RuntimeError("RDKit is required but is not available.")
    return Chem.MolFromSmiles(smiles) is not None


def validate_triazole(smiles: str) -> bool:
    """检查分子是否包含三唑并排除四唑。"""
    if Chem is None:
        raise RuntimeError("RDKit is required for triazole validation but is not available.")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return False

    try:
        tau = rdMolStandardize.TautomerEnumerator()
        mol = tau.Canonicalize(mol)
    except Exception:
        # 归一化失败时继续使用原分子匹配，避免中断筛选流程。
        pass

    has_triazole = any(mol.HasSubstructMatch(p) for p in _TRIAZOLE_SMARTS)
    if not has_triazole:
        return False

    has_tetrazole = any(mol.HasSubstructMatch(p) for p in _TETRAZOLE_SMARTS)
    return not has_tetrazole


def is_low_toxicity(toxicity: str) -> bool:
    """
    保留旧函数以兼容历史调用；当前主筛选流程不再使用 toxicity 作为条件。
    """
    if toxicity is None:
        return False
    t = str(toxicity).strip().lower()
    allow = {
        "微毒", "低毒", "低毒性", "轻度毒性", "轻度", "微", "低",
        "low", "low toxicity", "slight", "slightly toxic", "minor",
    }
    return t in allow


def meets_mic_criteria(aureus_mic: float) -> bool:
    """仅按 Aureus MIC 原始数值筛选。"""
    return pd.notna(aureus_mic) and aureus_mic < 20.0


def safe_format_value(value: Any, numeric: bool = False) -> str:
    """用于样例打印，兼容 None/NaN。"""
    if pd.isna(value):
        return "NA"
    if numeric:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def filter_molecules(input_file: str, output_file: str) -> None:
    """筛选分子并保存结果。"""
    print(f"读取分子数据: {input_file}")

    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
        print(f"总分子数: {len(df)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    required_columns = ["smiles", "aureus_MIC"]
    optional_columns = ["toxicity", "ecoli_MIC"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必要列: {missing_columns}")
        return

    for col in optional_columns:
        if col not in df.columns:
            df[col] = None

    # 注意：这里仅做数值转换，使用原始 MIC，不进行 log10 或 logMIC 转换。
    df["aureus_MIC"] = pd.to_numeric(df["aureus_MIC"], errors="coerce")
    df["ecoli_MIC"] = pd.to_numeric(df["ecoli_MIC"], errors="coerce")

    print("MIC 列数值化完成（原始值）")

    results = []
    output_columns = ["smiles", "toxicity", "aureus_MIC", "ecoli_MIC"]
    stats = {
        "valid_smiles": 0,
        "triazole": 0,
        "aureus_mic_criteria": 0,
        "all_criteria": 0,
    }

    print("开始筛选...")

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        toxicity = row["toxicity"]
        aureus_mic = row["aureus_MIC"]
        ecoli_mic = row["ecoli_MIC"]

        if pd.isna(smiles) or not isinstance(smiles, str):
            continue

        if not is_valid_smiles(smiles):
            continue
        stats["valid_smiles"] += 1

        if not validate_triazole(smiles):
            continue
        stats["triazole"] += 1

        if not meets_mic_criteria(aureus_mic):
            continue
        stats["aureus_mic_criteria"] += 1

        stats["all_criteria"] += 1
        results.append(
            {
                "smiles": smiles,
                "toxicity": toxicity,
                "aureus_MIC": aureus_mic,
                "ecoli_MIC": ecoli_mic,
            }
        )

        if (idx + 1) % 1000 == 0:
            print(f"已处理: {idx + 1}/{len(df)}")

    if results:
        result_df = pd.DataFrame(results, columns=output_columns)
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print("\n筛选完成！")
        print(f"符合条件的分子数: {len(results)}")
        print(f"结果已保存到: {output_file}")
    else:
        print("\n警告: 没有分子符合所有筛选条件！")
        result_df = pd.DataFrame(columns=output_columns)
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    total = len(df)
    denom = total if total > 0 else 1
    print("\n筛选统计:")
    print(f"原始分子数:           {total:6d}")
    print(
        f"有效SMILES:         {stats['valid_smiles']:6d} "
        f"({stats['valid_smiles'] / denom * 100:.1f}%)"
    )
    print(
        f"三唑分子:             {stats['triazole']:6d} "
        f"({stats['triazole'] / denom * 100:.1f}%)"
    )
    print(
        f"Aureus MIC条件满足: {stats['aureus_mic_criteria']:6d} "
        f"({stats['aureus_mic_criteria'] / denom * 100:.1f}%)"
    )
    print(
        f"全部条件满足:         {stats['all_criteria']:6d} "
        f"({stats['all_criteria'] / denom * 100:.1f}%)"
    )

    if not result_df.empty:
        print("\n前5个符合条件的分子:")
        sample_df = result_df.head(5)
        for _, row in sample_df.iterrows():
            print(f"SMILES: {row['smiles']}")
            print(
                "  毒性: "
                f"{safe_format_value(row['toxicity'])}, "
                f"Aureus MIC: {safe_format_value(row['aureus_MIC'], numeric=True)}, "
                f"E.coli MIC: {safe_format_value(row['ecoli_MIC'], numeric=True)}"
            )


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="分子筛选脚本")
    parser.add_argument(
        "--input",
        type=str,
        default="molecules.csv",
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="molecules_final.csv",
        help="输出 CSV 文件路径",
    )

    args = parser.parse_args()

    if Chem is None:
        print("错误: 需要安装 RDKit 才能运行此脚本。")
        return

    filter_molecules(args.input, args.output)


if __name__ == "__main__":
    main()
