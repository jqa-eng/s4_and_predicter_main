# top10_sim_in_train.py
# 用法：python top10_sim_in_train.py train.txt
import sys
import heapq
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchiKey

TARGET_SMILES = "O=C(c1ccc(Br)cc1)c1n[nH]nc1N(Cc1ccccc1)Cc1ccccc1"

def morgan_fp(mol, radius=2, nbits=2048):
    # 建议用 bit vector，算 Tanimoto 很快
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def main():
    if len(sys.argv) < 2:
        print("用法：python top10_sim_in_train.py train.txt")
        sys.exit(1)

    train_path = sys.argv[1]

    target_mol = Chem.MolFromSmiles(TARGET_SMILES)
    if target_mol is None:
        raise ValueError("TARGET_SMILES 解析失败，请检查：\n" + TARGET_SMILES)

    target_fp = morgan_fp(target_mol)
    target_key = MolToInchiKey(target_mol)

    # 小顶堆：存 (sim, smiles) ，堆大小保持为 10
    topk = []
    invalid_cnt = 0
    total_cnt = 0
    exact_inchikey_hits = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            total_cnt += 1

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_cnt += 1
                continue

            # 结构级“完全相同”检查（更硬的证据）
            try:
                if MolToInchiKey(mol) == target_key:
                    exact_inchikey_hits += 1
            except Exception:
                # 极少数分子可能 InChI 失败，不影响相似度计算
                pass

            fp = morgan_fp(mol)
            sim = DataStructs.TanimotoSimilarity(target_fp, fp)

            if len(topk) < 10:
                heapq.heappush(topk, (sim, smi))
            else:
                # 只保留前10
                if sim > topk[0][0]:
                    heapq.heapreplace(topk, (sim, smi))

    topk_sorted = sorted(topk, key=lambda x: x[0], reverse=True)

    print("=== 目标分子 ===")
    print("SMILES:", TARGET_SMILES)
    print("InChIKey:", target_key)
    print()
    print("=== 训练集扫描统计 ===")
    print("总行数(非空):", total_cnt)
    print("无效SMILES数:", invalid_cnt)
    print("InChIKey完全相同命中数:", exact_inchikey_hits)
    print()
    print("=== Top10 最相似分子（Morgan r=2, nBits=2048）===")
    for i, (sim, smi) in enumerate(topk_sorted, 1):
        print(f"{i:02d}\t{sim:.4f}\t{smi}")

if __name__ == "__main__":
    main()
