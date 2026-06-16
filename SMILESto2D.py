import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 创建保存目录
SAVE_DIR = "./data/structure_2dmol"
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取正负样本数据
df_pos = pd.read_csv("AMP.csv")      # 有列 'ID', 'SMILES'
df_neg = pd.read_csv("non_AMP.csv")
df_all = pd.concat([df_pos, df_neg], ignore_index=True)

success, fail = 0, 0

for idx, row in df_all.iterrows():
    mol_id = row["ID"]
    smiles = row["SMILES"]
    save_path = os.path.join(SAVE_DIR, f"{mol_id}.mol")

    if os.path.exists(save_path):
        continue  # 已存在文件，跳过

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        # 生成 2D 坐标并保存为 mol 文件
        AllChem.Compute2DCoords(mol)
        Chem.MolToMolFile(mol, save_path)

        print(f"成功生成: {mol_id}.mol", flush=True)
        success += 1

    except Exception as e:
        print(f"[失败] ID={mol_id}, SMILES={smiles}, 错误: {str(e)}", flush=True)
        fail += 1

print(f"\n成功保存 mol 文件: {success}")
print(f"失败或跳过样本数: {fail}")
