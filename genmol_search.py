#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genmol_search.py  (strict-mode include/exclude SMARTS; QUIET output)

目标：
- 与 genmol_v1_fixed.py 一样 **安静/简洁** 的输出：
  * 仅使用 tqdm 展示“已收集数量”进度；
  * 过滤阶段 **不打印逐条 warning**；
  * 结束时给出 **精炼的统计汇总**（各类剔除原因计数）。

规则（与你的需求一致）：
1) INCLUDE_SMARTS（全部必须命中）：三唑 + (C(=O) 或 C(=N)) + 6元含氮芳环(azine)
2) EXCLUDE_SMARTS（任意命中即剔）：异吲哚啉酮核心 + 四唑
3) 长度约束：canonical SMILES 长度 ≤ 60
4) 生成口径：批量生成→在线拒采→累计到 N（默认 N=100, batch=400）
"""

import argparse, sys
from collections import defaultdict
from typing import Tuple

# Suppress RDKit warnings and errors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs

# -------------------- INCLUDE_SMARTS （必须全部满足） --------------------
INCLUDE_SMARTS = {
    "TRIAZOLE_ANY": [
        "c1n[nH]nc1",  # 1,2,4
        "c1nc[nH]n1",  # 1,2,4 tautomer
        "c1c[nH]nn1",  # 1,2,3
        "c1[nH]nnc1",  # 1,2,3 tautomer
        "n1ncnc1",     # deprotonated
        "n1nccn1",     # deprotonated
    ],
    "CARBONYL_OR_IMID": [
        "[CX3](=O)",   # any carbonyl
        "[CX3](=N)",   # imidoyl (covers C(=NH))
    ],
    "AZINE_RING": [
        "n1ccccc1",
        "c1ncccc1",
        "c1cnccc1",
        "c1ccncc1",
        "c1cccnc1",
    ]
}

# -------------------- EXCLUDE_SMARTS （任意命中即剔除） --------------------
EXCLUDE_SMARTS = {
    "ISOINDOLINONE_CORE": [
        "C1(=O)Nc2ccccc2C1",
        "c1ccc2c(c1)CN(C2=O)",
        "c1ccc2c(c1)C(*)N(*)C2=O",
    ],
    "TETRAZOLE": [
        "n1nnnc1",
        "[nH]1nnnc1",
        "[n-]1nnnc1",
        "c1nnnn1",
        "c1nnn[nH]1",
        "c1nn[nH]n1",
        "c1[nH]nnn1",
    ],
}

def _compile_list(smarts_list):
    pats = []
    for s in smarts_list:
        p = Chem.MolFromSmarts(s)
        if p is not None:
            pats.append(p)
    return tuple(pats)

_COMPILED_INCLUDE = {k: _compile_list(v) for k, v in INCLUDE_SMARTS.items()}
_COMPILED_EXCLUDE = {k: _compile_list(v) for k, v in EXCLUDE_SMARTS.items()}

def parse_canonical(smi: str):
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
        if m is None: return None, None
        return m, Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None, None

def has_any(mol, pats) -> bool:
    return any(mol.HasSubstructMatch(p) for p in pats)

def meet_include(mol: Chem.Mol) -> bool:
    return (
        has_any(mol, _COMPILED_INCLUDE["TRIAZOLE_ANY"]) and
        has_any(mol, _COMPILED_INCLUDE["CARBONYL_OR_IMID"]) and
        has_any(mol, _COMPILED_INCLUDE["AZINE_RING"])
    )

def violate_exclude(mol: Chem.Mol) -> bool:
    for pats in _COMPILED_EXCLUDE.values():
        if has_any(mol, pats):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="S4 模型目录（与 genmol_v1_fixed.py 一致）")
    ap.add_argument("--required-prefix", default='Cc1n[nH]nc1', help="可选：三唑/关键片段前缀")
    ap.add_argument("--n", type=int, default=100, help="目标收集数量")
    ap.add_argument("--batch", type=int, default=1000, help="每批生成数")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=60, help="canonical SMILES 硬上限（默认60）")
    ap.add_argument("--out", default="gen_search.csv")
    args = ap.parse_args()

    # 延迟导入生成器（与 genmol_v1_fixed.py 风格一致）
    try:
        from s4dd.s4_for_denovo_design import S4forDenovoDesign
    except Exception as e:
        print("[ERROR] 未找到 S4 生成器模块（s4dd.s4_for_denovo_design）:", e)
        sys.exit(2)

    try:
        gen = S4forDenovoDesign.from_file(args.model_dir)
    except Exception as e:
        print(f"[ERROR] S4 生成器载入失败: {e}")
        sys.exit(2)

    print("="*72)
    print("genmol_search.py — QUIET 在线拒采：INCLUDE/EXCLUDE + 长度≤60")
    print("="*72)
    print(f"Model dir     : {args.model_dir}")
    print(f"Prefix        : {args.required_prefix}")
    print(f"Target N      : {args.n}   | batch={args.batch} | T={args.temperature}")
    print(f"Max len       : {args.max_len}")
    print("="*72)

    collected = []
    seen = set()
    drop_stats = defaultdict(int)   # 统计剔除原因

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.n, desc="收集进度（严格筛选）")
    except Exception:
        pbar = None

    def accept(smi: str) -> bool:
        m, csmi = parse_canonical(smi)
        if m is None:            drop_stats["parse_fail"] += 1;        return False
        if len(csmi) > args.max_len: drop_stats["too_long"] += 1;      return False
        if csmi in seen:         drop_stats["duplicate"] += 1;         return False
        if violate_exclude(m):   drop_stats["hit_exclude"] += 1;       return False
        if not meet_include(m):  drop_stats["miss_include"] += 1;      return False
        # 通过
        seen.add(csmi)
        collected.append(csmi)
        if pbar: pbar.update(1)
        return True

    # 主循环：批量生成 → 在线拒采 → 收集到 N
    while len(collected) < args.n:
        try:
            smiles_batch, _ = gen.design_molecules(
                n_designs=args.batch,
                batch_size=args.batch,
                temperature=args.temperature,
                required_prefix_smiles=args.required_prefix
            )
        except Exception as e:
            # 批次级别的失败提示一行即可（不逐条打印）
            if pbar: pbar.write(f"[warn] 生成批次失败：{e}")
            else:    print(f"[warn] 生成批次失败：{e}")
            continue

        for smi in smiles_batch:
            if len(collected) >= args.n: break
            accept(smi)

    if pbar: pbar.close()

    # 输出 CSV
    import pandas as pd
    df = pd.DataFrame([{"smiles": s, "len": len(s)} for s in collected])
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 汇总统计（单次打印，安静）
    total_seen = sum(drop_stats.values()) + len(df)
    print("\n" + "="*72)
    print(f"收集完成：{len(df)} 条  ->  {args.out}")
    print("="*72)
    print("剔除原因统计（quiet 模式，仅汇总）：")
    for k in ("parse_fail","too_long","duplicate","hit_exclude","miss_include"):
        print(f"  - {k:12s}: {drop_stats.get(k,0)}")
    print(f"  - accepted   : {len(df)}")
    print(f"总处理条目数（估算）: {total_seen}")
    print("="*72)
    print("提示：输出风格已对齐 genmol_v1_fixed.py —— 仅进度条 + 汇总统计，无逐条 warning。")

if __name__ == "__main__":
    main()
