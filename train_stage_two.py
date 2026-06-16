import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from model_stage_two import AMPFunctionClassifier
from torch_geometric.data import Data as GeoData, Batch
from rdkit import Chem
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from GGCLoss import GGCLoss
from GSCLoss import GSCLoss
from SSCLoss import SSCLoss
from Loss import FocalDiceLoss, AsymmetricLoss, FocalLoss
from evaluate_stage_two import compute_multilabel_metrics
from tqdm.auto import tqdm
from rdkit.Chem import MolToSmiles
from MLSC import MLSCLoss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================== 固定随机种子 ==================
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ================== 参数配置 ==================
MODEL_PATH = "./models/ChemBERTa"
DATA_PATH = "new_cycpeptides.csv"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
N_SPLITS = 5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATALOADER_NUM_WORKERS = 4

# ================== 对比学习：统一开关 ==================
CONTRASTIVE_MODE = "ALL"
SSC_WEIGHT = 0.1
GSC_WEIGHT = 0.1
GGC_WEIGHT = 0.1

GGC_STRATEGY = "mask_backbone_atoms"
SSC_TEMPERATURE = 0.1
GSC_TEMPERATURE = 0.1
GGC_TEMPERATURE = 0.5

# ====== MLSC 单独配置 ======
MLSC_ENABLE = True
MLSC_WEIGHT = 0.6
MLSC_TEMPERATURE = 0.3
MLSC_USE_FUSED_Z = False

# ====== 模型保存配置 ======
SAVE_DIR = "./save_models"
SAVE_BASENAME = "model_stage_two"

# ====== 结构特征缓存 ======
FEATURE_CACHE_DIR = os.path.join(".", "features", "graph")
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

# ================== 数据、图构建与 Dataset ==================
def mol_to_graph(mol):
    atoms = []
    atom_types = []
    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        atoms.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic())
        ])
    x = torch.tensor(atoms, dtype=torch.float)
    atom_type = torch.tensor(atom_types, dtype=torch.long)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return GeoData(x=x, edge_index=edge_index, atom_type=atom_type)

def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == "graph_data":
            batch_dict[key] = Batch.from_data_list([item[key] for item in batch])
        elif key == "smiles":
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = torch.utils.data.default_collate([item[key] for item in batch])
    return batch_dict

class AMPFunctionDataset(Dataset):
    def __init__(self, df, tokenizer, mol_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.mol_dir = mol_dir
        self.cache_dir = FEATURE_CACHE_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        encoded = self.tokenizer(smiles, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        mol_path = os.path.join(self.mol_dir, f"{row['ID']}.mol")

        cache_path = os.path.join(self.cache_dir, f"{row['ID']}.pt")
        if os.path.exists(cache_path):
            try:
                graph_data = torch.load(cache_path)
            except Exception as e:
                print(f"Failed to load cached graph for {row['ID']}, recalculating...")
                mol = Chem.MolFromMolFile(mol_path, sanitize=True)
                graph_data = mol_to_graph(mol)
                torch.save(graph_data, cache_path)
        else:
            mol = Chem.MolFromMolFile(mol_path, sanitize=True)
            graph_data = mol_to_graph(mol)
            torch.save(graph_data, cache_path)

        labels_np = row[["Anti-Bacterial", "Anti-Gram+", "Anti-Gram-", "Anti-Fungal", "Anti-Biotics", "Anti-Viral"]].values
        labels_np = np.array(labels_np, dtype=np.float32)
        labels = torch.tensor(labels_np, dtype=torch.float)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "graph_data": graph_data,
            "labels": labels,
            "idx": idx,
            "smiles": smiles,
        }

# ================== SMILES 随机增强（SSC 用） ==================
def augment_smiles_batch(smiles_list):
    smiles_aug1, smiles_aug2 = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            smiles_aug1.append(MolToSmiles(mol, doRandom=True))
            smiles_aug2.append(MolToSmiles(mol, doRandom=True))
        else:
            smiles_aug1.append(s)
            smiles_aug2.append(s)
    return smiles_aug1, smiles_aug2

# ====== GGC 输入适配器 ======
class _GGCInputWrapper:
    def __init__(self, batch_obj):
        self._batch = batch_obj

    def to_data_list(self):
        lst = self._batch.to_data_list()
        for d in lst:
            if getattr(d, "batch", None) is None:
                device = d.x.device if hasattr(d, "x") and d.x is not None else None
                num_nodes = d.num_nodes if hasattr(d, "num_nodes") else (d.x.size(0) if hasattr(d, "x") else 0)
                d.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        return lst

    def __getattr__(self, name):
        return getattr(self._batch, name)

def train_single_model(fold, model, train_loader, val_loader, tokenizer):
    print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    use_ssc = ("SSC" in CONTRASTIVE_MODE) or (CONTRASTIVE_MODE == "ALL")
    use_gsc = ("GSC" in CONTRASTIVE_MODE) or (CONTRASTIVE_MODE == "ALL")
    use_ggc = ("GGC" in CONTRASTIVE_MODE) or (CONTRASTIVE_MODE == "ALL")
    use_mlsc = MLSC_ENABLE

    if use_ggc:
        ggc_contrastive_loss_fn = GGCLoss(strategy=GGC_STRATEGY, temperature=GGC_TEMPERATURE).to(DEVICE)
    if use_gsc:
        gsc_contrastive_loss_fn = GSCLoss(hidden_dim=384, proj_dim=384, temperature=GSC_TEMPERATURE, lam=0.5, normalize=True).to(DEVICE)
    if use_ssc:
        ssc_contrastive_loss_fn = SSCLoss(temperature=SSC_TEMPERATURE).to(DEVICE)
    if use_mlsc:
        mlsc_loss_fn = MLSCLoss(temperature=MLSC_TEMPERATURE, normalize=True).to(DEVICE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    fold_path = os.path.join(SAVE_DIR, f"{SAVE_BASENAME}_fold{fold+1}.pt")
    best_loss = float('inf')

    with tqdm(total=EPOCHS,
              desc=f"Fold {fold+1}",
              ncols=100,
              bar_format='{l_bar}{n_fmt}/{total_fmt} |{bar}|') as pbar:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            # ====== 常规训练 ======
            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                graph_data = batch["graph_data"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                if use_gsc or use_ssc or use_mlsc:
                    logits, seq_embed, graph_embed = model(input_ids, attention_mask, graph_data, return_features=True)
                else:
                    logits = model(input_ids, attention_mask, graph_data, return_features=False)

                loss = criterion(logits, labels)

                if use_ggc:
                    loss_ggc = ggc_contrastive_loss_fn(model.gat, graph_data)
                    loss = loss + GGC_WEIGHT * loss_ggc

                if use_gsc:
                    loss_gsc = gsc_contrastive_loss_fn(graph_embed, seq_embed)
                    loss = loss + GSC_WEIGHT * loss_gsc

                if use_ssc:
                    smiles_list = batch["smiles"]
                    smiles_aug1, smiles_aug2 = augment_smiles_batch(smiles_list)
                    enc1 = tokenizer(smiles_aug1, truncation=True, padding="max_length", max_length=512,
                                     return_tensors="pt").to(DEVICE)
                    enc2 = tokenizer(smiles_aug2, truncation=True, padding="max_length", max_length=512,
                                     return_tensors="pt").to(DEVICE)

                    z1 = model.encode_sequence(enc1["input_ids"], enc1["attention_mask"])
                    z2 = model.encode_sequence(enc2["input_ids"], enc2["attention_mask"])
                    loss_ssc = ssc_contrastive_loss_fn(z1, z2)
                    loss = loss + SSC_WEIGHT * loss_ssc

                if use_mlsc and logits.size(0) > 1:
                    if MLSC_USE_FUSED_Z:
                        z_seq = F.normalize(seq_embed, dim=1)
                        z_graph = F.normalize(graph_embed, dim=1)
                        z = F.normalize((z_seq + z_graph) / 2.0, dim=1)
                    else:
                        z = F.normalize(seq_embed, dim=1)

                    loss_mlsc = mlsc_loss_fn(z, labels)
                    loss = loss + MLSC_WEIGHT * loss_mlsc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), fold_path)
                print(f"Fold {fold + 1}: model saved (avg_loss={best_loss:.4f}) -> {fold_path}")

            # 验证
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    graph_data = batch["graph_data"].to(DEVICE)
                    labels_tensor = batch["labels"].to(DEVICE)

                    logits = model(input_ids, attention_mask, graph_data, return_features=False)

                    probs = logits.sigmoid().cpu().numpy()
                    labels_np = labels_tensor.cpu().numpy()

                    all_logits.append(probs)
                    all_labels.append(labels_np)

            metrics = compute_multilabel_metrics(np.vstack(all_labels), np.vstack(all_logits))
            pbar.update(1)

    print(f"Final Metrics for Fold {fold + 1}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics

# ================== 主程序 ==================
def main():
    df = pd.read_csv(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = AMPFunctionDataset(df, tokenizer, mol_dir="./data/structure_2dmol")

    label_cols = ["Anti-Bacterial", "Anti-Gram+", "Anti-Gram-", "Anti-Fungal", "Anti-Biotics", "Anti-Viral"]
    y = df[label_cols].values

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(df, y)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=custom_collate_fn, num_workers=DATALOADER_NUM_WORKERS)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE,
                                collate_fn=custom_collate_fn, num_workers=DATALOADER_NUM_WORKERS)

        model = AMPFunctionClassifier().to(DEVICE)
        metrics = train_single_model(fold, model, train_loader, val_loader, tokenizer)
        all_metrics.append(metrics)

    print("\n========== Average Metrics Across Folds ==========")
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
