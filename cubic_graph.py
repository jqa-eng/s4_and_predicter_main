# -*- coding: utf-8 -*-
"""
cubic_graph.py - 三维分子性质分布图绘制脚本

功能：
- 读取完整分子预测结果文件 mol-processed；
- 以 S.aureus_MIC 为 x 轴；
- 以 E.coli_MIC 为 y 轴；
- 将毒性标签 toxicity 映射为 IC50 代表值作为 z 轴；
- 使用 toxicity 控制散点颜色；
- 读取筛选后候选分子文件 mol-filtered；
- 根据 mol-filtered 中的 SMILES 在完整分子列表中匹配候选分子；
- 将筛选通过的候选分子以星号标记。

输入：
1. --mol-processed：完整分子列表，必须包含 smiles、toxicity、aureus_MIC、ecoli_MIC；
2. --mol-filtered：筛选通过的分子列表，至少包含 smiles。

输出：
- molecules_3d_scatter.png
- molecules_3d_scatter.svg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# =============== 参数区（按需修改）===============
DATA_DIR = Path(".")

TOX_TO_Z = {
    "高毒": 5.0,
    "中毒": (10 + 75) / 2,
    "微毒": (75 + 200) / 2,
    "低毒": 250.0,
}

TOX_TO_COLOR = {
    "低毒": "#2ca02c",
    "微毒": "#ffdd00",
    "中毒": "#ff7f0e",
    "高毒": "#d62728",
}

SIZE_CIRCLE = 12
SIZE_STAR = 60
ALPHA_CIRCLE = 0.6
ALPHA_STAR = 0.95

ELEV = 22
AZIM = -65

Z_LIM = (0, 300)
Z_TICKS = [0, 10, 75, 200, 300]
X_LIM = (0, 100)
Y_LIM = (0, 100)

OUT_PNG = "molecules_3d_scatter.png"
OUT_SVG = "molecules_3d_scatter.svg"
DPI = 180
FIGSIZE = (10, 8)
# ==============================================


def read_clean(path: Path) -> pd.DataFrame:
    """读取 CSV，并清理列名和 smiles 字段。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if "smiles" in df.columns:
        df["smiles"] = df["smiles"].astype(str).str.strip().str.strip('"').str.strip("'")
    return df


def main():
    parser = argparse.ArgumentParser(description="绘制分子三维散点图")
    parser.add_argument(
        "--mol-processed",
        type=str,
        default="molecules.csv",
        help="完整分子列表 CSV 文件路径，用于绘制所有散点",
    )
    parser.add_argument(
        "--mol-filtered",
        type=str,
        default="molecules_final.csv",
        help="筛选通过的分子 CSV 文件路径，用于标星",
    )
    args = parser.parse_args()

    base = DATA_DIR
    processed_path = Path(args.mol_processed)
    filtered_path = Path(args.mol_filtered)

    mol_processed = read_clean(processed_path)
    mol_filtered = read_clean(filtered_path)

    required_processed_columns = ["smiles", "toxicity", "aureus_MIC", "ecoli_MIC"]
    missing_processed = [c for c in required_processed_columns if c not in mol_processed.columns]
    if missing_processed:
        raise ValueError(f"mol-processed 文件缺少必要列: {missing_processed}")

    if "smiles" not in mol_filtered.columns:
        raise ValueError("mol-filtered 文件缺少必要列: ['smiles']")

    candidate_set = set(mol_filtered["smiles"].astype(str).str.strip())
    mol_processed["is_candidate"] = (
        mol_processed["smiles"].astype(str).str.strip().isin(candidate_set)
    )

    for col in ["aureus_MIC", "ecoli_MIC"]:
        mol_processed[col] = pd.to_numeric(mol_processed[col], errors="coerce")

    mol_processed["toxicity"] = mol_processed["toxicity"].astype(str).str.strip()
    mol_processed["IC50_z"] = mol_processed["toxicity"].map(TOX_TO_Z)

    must_cols = ["aureus_MIC", "ecoli_MIC", "IC50_z"]
    plot_df = mol_processed.dropna(subset=must_cols).copy()

    x0, x1 = X_LIM
    y0, y1 = Y_LIM
    z0, z1 = Z_LIM
    in_range = (
        plot_df["aureus_MIC"].between(x0, x1)
        & plot_df["ecoli_MIC"].between(y0, y1)
        & plot_df["IC50_z"].between(z0, z1)
    )
    plot_df = plot_df.loc[in_range].copy()

    plt.close("all")
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r"MIC$_{S.\,aureus}$ ($\mu$g/mL)")
    ax.set_ylabel(r"MIC$_{E.\,coli}$ ($\mu$g/mL)")
    ax.set_zlabel(r"IC$_{50}$ ($\mu$g/mL)")

    ax.set_zlim(*Z_LIM)
    ax.set_zticks(Z_TICKS)

    for tox, color in TOX_TO_COLOR.items():
        sub = plot_df[plot_df["toxicity"] == tox]
        if sub.empty:
            continue

        sub_nc = sub[~sub["is_candidate"]]
        if not sub_nc.empty:
            ax.scatter(
                sub_nc["aureus_MIC"].values,
                sub_nc["ecoli_MIC"].values,
                sub_nc["IC50_z"].values,
                marker="o",
                s=SIZE_CIRCLE,
                edgecolors="none",
                c=color,
                alpha=ALPHA_CIRCLE,
            )

        sub_c = sub[sub["is_candidate"]]
        if not sub_c.empty:
            ax.scatter(
                sub_c["aureus_MIC"].values,
                sub_c["ecoli_MIC"].values,
                sub_c["IC50_z"].values,
                marker="*",
                s=SIZE_STAR,
                edgecolors="k",
                linewidths=0.3,
                c=color,
                alpha=ALPHA_STAR,
            )

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TOX_TO_COLOR[t],
            markeredgecolor="none",
            markersize=8,
            label=t,
        )
        for t in ["低毒", "微毒", "中毒", "高毒"]
        if (plot_df["toxicity"] == t).any()
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=6,
            label="非候选",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=10,
            label="候选（星标）",
        ),
    ]
    handles = color_handles + shape_handles

    ax.legend(
        handles=handles,
        title="图例：颜色=毒性；形状=是否候选",
        loc="upper left",
        bbox_to_anchor=(1.00, 1.02),
        frameon=True,
        borderaxespad=0.6,
        borderpad=0.6,
        handletextpad=0.6,
        labelspacing=0.5,
        columnspacing=1.0,
        ncol=1,
    )

    ax.grid(True)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    fig.savefig(base / OUT_PNG, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(base / OUT_SVG, bbox_inches="tight", pad_inches=0.15)

    matched_before_range = int(mol_processed["is_candidate"].sum())
    print(
        {
            "mol_processed": str(processed_path),
            "mol_filtered": str(filtered_path),
            "total_processed_rows": int(len(mol_processed)),
            "filtered_smiles": int(len(candidate_set)),
            "matched_candidates_before_range_filter": matched_before_range,
            "total_points_plotted": int(len(plot_df)),
            "candidates_plotted": int(plot_df["is_candidate"].sum()),
            "non_candidates_plotted": int((~plot_df["is_candidate"]).sum()),
            "dropped_missing_or_out_of_range": int(len(mol_processed) - len(plot_df)),
            "out_png": str(Path(OUT_PNG)),
            "out_svg": str(Path(OUT_SVG)),
        }
    )


if __name__ == "__main__":
    main()
