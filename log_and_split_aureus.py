#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_and_split_aureus.py

用途：
1. 读取 aureus 回归训练数据 CSV（utf-8-sig）。
2. 仅在 aureus_logMIC 缺失时，用 aureus_MIC 计算并填充 log10(MIC)。
3. 清洗无效 MIC（非数值或 <= 0），并删除最终仍缺失 aureus_logMIC 的样本。
4. 按给定比例随机划分为 train / val / test。
5. 输出可直接用于 train_aureus_regresser_fixed.py 的数据文件。
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)

    if np.any(ratios < 0):
        raise ValueError("train/val/test ratio 不能为负数。")

    total = float(ratios.sum())
    if total <= 0:
        raise ValueError("train/val/test ratio 之和必须大于 0。")

    if not np.isclose(total, 1.0, atol=1e-6):
        print(f"WARNING: ratio 之和为 {total:.12f}，将自动归一化为 1。")
        ratios = ratios / total

    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def _to_numeric_with_strip(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "none": np.nan,
            "None": np.nan,
            "null": np.nan,
            "NULL": np.nan,
        }
    )
    return pd.to_numeric(text, errors="coerce")


def _split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) == 0:
        raise ValueError("清洗后样本数为 0，无法划分数据集。")

    if np.isclose(train_ratio, 1.0):
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    if np.isclose(train_ratio, 0.0):
        train_df = df.iloc[0:0].copy()
        temp_df = df.copy()
    else:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )

    remain_ratio = val_ratio + test_ratio
    if len(temp_df) == 0 or np.isclose(remain_ratio, 0.0):
        return train_df, temp_df.iloc[0:0].copy(), temp_df.iloc[0:0].copy()

    if np.isclose(val_ratio, 0.0):
        return train_df, temp_df.iloc[0:0].copy(), temp_df
    if np.isclose(test_ratio, 0.0):
        return train_df, temp_df, temp_df.iloc[0:0].copy()

    val_relative = val_ratio / remain_ratio
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        random_state=seed,
        shuffle=True,
    )
    return train_df, val_df, test_df


def process_and_split(args: argparse.Namespace) -> None:
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    train_ratio, val_ratio, test_ratio = _normalize_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "aureus_split_datasets")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Aureus logMIC 填充与数据划分脚本")
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"seed: {args.seed}")
    print(f"ratio: train={train_ratio:.6f}, val={val_ratio:.6f}, test={test_ratio:.6f}")

    df = pd.read_csv(input_path, sep=None, engine="python", encoding="utf-8-sig")
    print("检测到列名：", df.columns.tolist())
    original_columns = list(df.columns)
    original_n = len(df)

    if "aureus_MIC" not in df.columns:
        raise ValueError("输入数据缺少必需列: aureus_MIC")

    if "aureus_logMIC" not in df.columns:
        print("WARNING: 输入数据缺少 aureus_logMIC 列，已创建空列用于填充。")
        df["aureus_logMIC"] = np.nan
        original_columns = original_columns + ["aureus_logMIC"]

    mic_numeric = _to_numeric_with_strip(df["aureus_MIC"])
    invalid_mic_mask = mic_numeric.isna() | (mic_numeric <= 0)
    invalid_mic_n = int(invalid_mic_mask.sum())
    non_positive_mic_n = int((mic_numeric <= 0).fillna(False).sum())
    non_numeric_mic_n = int(mic_numeric.isna().sum())
    if invalid_mic_n > 0:
        print(
            f"WARNING: 发现无效 aureus_MIC {invalid_mic_n} 条 "
            f"(非数值/缺失: {non_numeric_mic_n}, <=0: {non_positive_mic_n})。"
        )

    df["aureus_MIC"] = mic_numeric

    log_text = df["aureus_logMIC"].astype(str).str.strip()
    missing_tokens = {"", "nan", "NaN", "none", "None", "null", "NULL"}
    missing_log_mask = df["aureus_logMIC"].isna() | log_text.isin(missing_tokens)
    missing_log_n = int(missing_log_mask.sum())

    fillable_mask = missing_log_mask & (~invalid_mic_mask)
    fill_n = int(fillable_mask.sum())
    df.loc[fillable_mask, "aureus_logMIC"] = np.log10(df.loc[fillable_mask, "aureus_MIC"])

    df["aureus_logMIC"] = _to_numeric_with_strip(df["aureus_logMIC"])

    delete_mask = df["aureus_logMIC"].isna()
    delete_n = int(delete_mask.sum())
    df_clean = df.loc[~delete_mask].copy()
    final_n = len(df_clean)

    if final_n == 0:
        raise ValueError("清洗后没有可用样本（aureus_logMIC 全部缺失）。")

    df_clean = df_clean.loc[:, original_columns]

    train_df, val_df, test_df = _split_dataframe(
        df_clean, train_ratio, val_ratio, test_ratio, args.seed
    )

    train_path = os.path.join(output_dir, "aureus_train.csv")
    val_path = os.path.join(output_dir, "aureus_val.csv")
    test_path = os.path.join(output_dir, "aureus_test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"1) 原始数据量: {original_n}")
    print("2) aureus_logMIC:")
    print(f"   - 原始缺失数量: {missing_log_n}")
    print(f"   - 成功填充数量: {fill_n}")
    print(f"   - 删除数量: {delete_n}")
    print(f"3) 最终样本数: {final_n}")
    print("4) train / val / test 样本数量:")
    print(f"   - train: {len(train_df)}")
    print(f"   - val:   {len(val_df)}")
    print(f"   - test:  {len(test_df)}")
    print("5) 输出路径:")
    print(f"   - {train_path}")
    print(f"   - {val_path}")
    print(f"   - {test_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="填充 aureus_logMIC 并划分 Aureus 回归训练数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径。")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例。")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例。")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例。")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录。默认与输入文件同级的 aureus_split_datasets。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_and_split(args)


if __name__ == "__main__":
    main()
