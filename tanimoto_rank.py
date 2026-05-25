# tanimoto_rank.py

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# ========= 配置区 =========

INPUT_FILE = "molecules.csv"

REFERENCED = "O=C(C1=CC=C(Br)C=C1)C2=C(N(CC3=CC=CC=C3)CC4=CC=CC=C4)NN=N2"

TOP_K = 10

# Morgan 参数
RADIUS = 2
NBITS = 2048


# ========= 主程序 =========

def main():

    # 1. 读取 CSV
    df = pd.read_csv(INPUT_FILE)

    if "smiles" not in df.columns:
        raise ValueError("CSV 中找不到 smiles 列")

    smiles_list = df["smiles"].dropna().tolist()

    print(f"读取到 {len(smiles_list)} 条分子")


    # 2. 参考分子
    ref_mol = Chem.MolFromSmiles(REFERENCED)

    if ref_mol is None:
        raise ValueError("参考分子 SMILES 无效")

    generator = GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
    ref_fp = generator.GetFingerprint(ref_mol)


    # 3. 计算相似度
    results = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        fp = generator.GetFingerprint(mol)

        sim = DataStructs.TanimotoSimilarity(ref_fp, fp)

        results.append((smi, sim))


    print(f"成功计算 {len(results)} 条有效分子")


    # 4. 排序
    results.sort(key=lambda x: x[1], reverse=True)


    # 5. 输出 Top-K
    print("\n===== Top {} Similar Molecules =====".format(TOP_K))

    for i, (smi, sim) in enumerate(results[:TOP_K], 1):

        print(f"{i:02d}. {smi}, {sim:.4f}")


if __name__ == "__main__":
    main()
