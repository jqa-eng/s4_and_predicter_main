# -*- coding: utf-8 -*-
"""
boost_white_moonlight.py
Purpose:
  Around a seed (your "白月光"), locally enumerate SAR variants (halogen swaps, linker isosteres,
  chain length ±1, tetrazole tweaks), filter by constraints (non-isoindolinone, tetrazole present),
  compute QED/SA & basic props, and RANK by your project predictors (MIC_s.aureus / MIC_e.coli / tox).

Requires:
  - rdkit
Optional:
  - rdkit.Chem.SA_Score.sascorer  (for SA score)

Usage:
  python boost_white_moonlight.py --seed "SMILES_HERE" --out candidates_wm_neighborhood.csv \
      --n_each 8 --max_total 512

You MUST implement your scoring in `score_with_project_models(mol, smi)`.
"""
import argparse, random
from typing import List, Tuple, Optional, Dict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors, Crippen, rdMolDescriptors, Lipinski

# ---------- SMARTS filters ----------
ISO_CORES = [
    "C1(=O)Nc2ccccc2C1",
    "c1ccc2c(c1)CN(C2=O)",
    "c1ccc2c(c1)C(*)N(*)C2=O",
]
ISO_PATS = [Chem.MolFromSmarts(s) for s in ISO_CORES]

# Triazole patterns (from project requirements)
TRIAZOLE_SMARTS = [
    "c1n[nH]nc1",  # 1,2,4-triazole (aromatic)
    "c1nc[nH]n1",  # 1,2,4-triazole (tautomer)
    "c1c[nH]nn1",  # 1,2,3-triazole (aromatic)
    "c1[nH]nnc1",  # 1,2,3-triazole (tautomer)
    "n1ncnc1",     # 1,2,4-triazole (deprotonated)
    "n1nccn1",     # 1,2,3-triazole (deprotonated)
]
TRIAZOLE_PATS = [Chem.MolFromSmarts(s) for s in TRIAZOLE_SMARTS]

def has_iso_core(m: Chem.Mol) -> bool:
    for p in ISO_PATS:
        if p is not None and m.HasSubstructMatch(p):
            return True
    return False

def has_triazole(m: Chem.Mol) -> bool:
    for p in TRIAZOLE_PATS:
        if p is not None and m.HasSubstructMatch(p):
            return True
    return False

# ---------- mutation operators (minimal, chemistry-safe-ish) ----------
HALOGENS = ["F","Cl","Br"]
AZINE_RING_SMARTS = Chem.MolFromSmarts("c1ncccc1")   # loose pyridine-like finder
PHENYL_SMARTS     = Chem.MolFromSmarts("c1ccccc1")

def mutate_halogen(smi: str) -> List[str]:
    """Swap halogens in aryl positions: Cl/F/Br interchanges."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    res = set()
    for atom in m.GetAtoms():
        sym = atom.GetSymbol()
        if sym in HALOGENS:
            for new in HALOGENS:
                if new != sym:
                    nm = Chem.RWMol(m)
                    nm.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new))
                    try:
                        Chem.SanitizeMol(nm)
                        res.add(Chem.MolToSmiles(nm, isomericSmiles=True))
                    except: pass
    return list(res)

def mutate_add_methyl_scan(smi: str, n_each=4) -> List[str]:
    """Naive methyl scan: attach a CH3 to eligible aromatic carbons by simple replace of H->CH3 (rough)."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    res = set()
    patt = PHENYL_SMARTS
    matches = m.GetSubstructMatches(patt)
    for match in matches:
        for idx in match:
            atom = m.GetAtomWithIdx(idx)
            if atom.GetIsAromatic() and atom.GetSymbol()=="C" and atom.GetTotalDegree()<=3:
                nm = Chem.RWMol(m)
                # add carbon and a single bond
                new_idx = nm.AddAtom(Chem.Atom("C"))
                nm.AddBond(idx, new_idx, Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(nm)
                    res.add(Chem.MolToSmiles(nm, isomericSmiles=True))
                except: pass
                if len(res)>=n_each: break
        if len(res)>=n_each: break
    return list(res)

def mutate_linker_isostere(smi: str) -> List[str]:
    """Replace C(=O) linkers by isosteres: amide/urea/carbamate/reverse amide (string hacks, keep simple)."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    res=set()
    if "C(=O)" in smi:
        for rep in ["NC(=O)", "OC(=O)", "N(C)=O", "NC(=O)N"]:
            s = smi.replace("C(=O)", rep, 1)
            mm = Chem.MolFromSmiles(s)
            if mm:
                try:
                    Chem.SanitizeMol(mm)
                    res.add(Chem.MolToSmiles(mm, isomericSmiles=True))
                except: pass
    return list(res)

def mutate_chain_length(smi: str) -> List[str]:
    """Very light chain edits: insert/remove CH2 next to a carbonyl (pattern-based)."""
    res=set()
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    if "C(=O)C" in smi:
        s = smi.replace("C(=O)C", "C(=O)CC", 1)  # +CH2
        if Chem.MolFromSmiles(s): res.add(Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True))
    if "C(=O)CC" in smi:
        t = smi.replace("C(=O)CC", "C(=O)C", 1)  # -CH2
        if Chem.MolFromSmiles(t): res.add(Chem.MolToSmiles(Chem.MolFromSmiles(t), isomericSmiles=True))
    return list(res)

def uniq_valid(cands: List[str]) -> List[str]:
    out=set()
    for s in cands:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        out.add(Chem.MolToSmiles(m, isomericSmiles=True))
    return list(out)

# ---------- property calculation ----------
def calc_props(m: Chem.Mol) -> Dict[str, float]:
    try:
        qed = float(QED.qed(m))
    except:
        qed = None

    try:
        # Try multiple import paths for SA Score
        try:
            from rdkit.Chem import RDConfig
            import os, sys
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            sa = float(sascorer.calculateScore(m))
        except:
            from rdkit.Chem.SA_Score import sascorer
            sa = float(sascorer.calculateScore(m))
    except Exception:
        sa = None

    d = {
        "MW": float(Descriptors.MolWt(m)),
        "cLogP": float(Crippen.MolLogP(m)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(m)),
        "HBD": int(Lipinski.NumHDonors(m)),
        "HBA": int(Lipinski.NumHAcceptors(m)),
        "RotB": int(Lipinski.NumRotatableBonds(m)),
        "AromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "QED": qed,
        "SA": sa
    }
    return d

def is_ok_basic(m: Chem.Mol) -> bool:
    if not has_triazole(m): return False
    if has_iso_core(m): return False
    return True

def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="seed smiles (白月光)")
    ap.add_argument("--out", default=None, help="output csv (default: molecules_extended.csv in script dir)")
    ap.add_argument("--n_each", type=int, default=8, help="each operator proposals (cap)")
    ap.add_argument("--max_total", type=int, default=512, help="max candidates to keep before scoring")
    args = ap.parse_args()

    seed = args.seed.strip()
    smi_set = set([seed])

    # Default output path
    if args.out is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, "molecules_extended.csv")
    else:
        output_file = args.out

    print(f"[INFO] Seed molecule: {seed}")
    print(f"[INFO] Generating neighbors...")

    # propose mutations
    ops = [
        lambda s: mutate_halogen(s),
        lambda s: mutate_add_methyl_scan(s, n_each=args.n_each),
        lambda s: mutate_linker_isostere(s),
        lambda s: mutate_chain_length(s),
    ]
    proposals = set()
    for op in ops:
        for smi in list(smi_set):
            try:
                out = op(smi)
                for x in out:
                    proposals.add(x)
            except Exception:
                continue

    print(f"[INFO] Generated {len(proposals)} raw proposals")
    cands = uniq_valid(list(proposals))[:args.max_total]
    print(f"[INFO] After dedup/validation: {len(cands)} candidates")

    # Calculate properties and filter
    rows = []
    for smi in cands:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if not is_ok_basic(m):
            continue

        props = calc_props(m)
        rows.append({
            "smiles": smi,
            **props
        })

    if not rows:
        print("[WARN] No candidates passed basic filters (triazole present, no isoindolinone).")
        print("[WARN] Try increasing --n_each or --max_total.")
        return

    # Sort by composite score: higher QED (better), lower SA (easier synthesis)
    df = pd.DataFrame(rows)
    df['composite_score'] = df['QED'].fillna(0.0) * 2.0 - df['SA'].fillna(10.0) * 0.1
    df = df.sort_values("composite_score", ascending=False)

    df.to_csv(output_file, index=False)
    print(f"[DONE] Wrote {len(df)} molecules to: {output_file}")
    print(f"\n[INFO] Top 5 candidates by composite score:")
    print(df[['smiles', 'QED', 'SA', 'MW', 'cLogP']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
