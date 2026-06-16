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
from sampling import select_hard_negatives, select_hard_positives
from MLSC import MLSCLoss   # ⬅️ 多标签监督对比学习（保留可选）
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行，消除警告

# ================== 固定随机种子 ==================
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ================== 参数配置（保持原有） ==================
MODEL_PATH = "DeepChem/ChemBERTa-77M-MLM"
DATA_PATH = "new_cycpeptides.csv"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
N_SPLITS = 5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATALOADER_NUM_WORKERS = 4  # ← DataLoader 的 num_workers（视 CPU 而定，可设 4~8）

# ================== 对比学习：统一开关 ==================
# 选项: "none", "SSC", "GSC", "GGC", "SSC+GSC", "SSC+GGC", "GSC+GGC", "ALL"
CONTRASTIVE_MODE = "ALL"
SSC_WEIGHT = 0.1
GSC_WEIGHT = 0.1
GGC_WEIGHT = 0.1

# 具体超参
GGC_STRATEGY = "mask_backbone_atoms"  # "mask_backbone_atoms" / "random_mask_nodes" / "drop_non_backbone_nodes" / "perturb_edges_and_nodes"
SSC_TEMPERATURE = 0.1
GSC_TEMPERATURE = 0.1
GGC_TEMPERATURE = 0.5

# ====== MLSC 单独配置（仅新增配置，不改原有对比学习代码）======
MLSC_ENABLE = True
MLSC_WEIGHT = 0.6
MLSC_TEMPERATURE = 0.3
MLSC_USE_FUSED_Z = False

# ====== 采样控制参数 ======
USE_HARD_NEGATIVE = False
USE_HARD_POSITIVE = False
SELECTED_EPOCH = 10
N_SAMPLES = 5
CONFIDENCE_THRESHOLD_NEGATIVE = 0.5
CONFIDENCE_THRESHOLD_POSITIVE = 0.5
SAMPLING_NUM_WORKERS = 5  # 采样 DataLoader 的 num_workers

# ====== 模型保存配置（新增）======
SAVE_DIR = "./save_models"
SAVE_BASENAME = "new_model_stage_two"

# ====== 结构特征缓存（仅缓存图结构 Data，不缓存可学习嵌入）======
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

    # 关键修复：若无边，则构造 [2, 0] 空图，避免 GAT 报错
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
            batch_dict[key] = [item[key] for item in batch]  # ← 与一阶段一致：保持为 list[str]
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

        # 先检查是否存在缓存图结构特征文件
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

            # 交给 Batch.from_data_list 在 collate_fn 中自动生成 batch 向量
            # graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

            torch.save(graph_data, cache_path)

        # 提取标签
        labels_np = row[["Anti-Bacterial", "Anti-Gram+", "Anti-Gram-", "Anti-Fungal", "Anti-Biotics", "Anti-Viral"]].values
        labels_np = np.array(labels_np, dtype=np.float32)
        labels = torch.tensor(labels_np, dtype=torch.float)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "graph_data": graph_data,
            "labels": labels,
            "idx": idx,  # 供 SSC 使用
            "smiles": smiles,  # ← 新增：与一阶段保持一致
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

# ====== GGC 输入适配器（不改 GGCLoss，本地包一层）======
class _GGCInputWrapper:
    def __init__(self, batch_obj):
        self._batch = batch_obj  # 原始 Batch

    def to_data_list(self):
        lst = self._batch.to_data_list()  # List[Data]
        for d in lst:
            # 单图默认没有 batch，这里补一个全 0（与 Data.x 同设备）
            if getattr(d, "batch", None) is None:
                device = d.x.device if hasattr(d, "x") and d.x is not None else None
                num_nodes = d.num_nodes if hasattr(d, "num_nodes") else (d.x.size(0) if hasattr(d, "x") else 0)
                d.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        return lst

    # 关键：把其他属性（x/edge_index/…）都代理给原始 Batch
    def __getattr__(self, name):
        return getattr(self._batch, name)


def train_single_model(fold, model, train_loader, val_loader, df=None, tokenizer=None):
    print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 初始化对比学习损失函数（按需）
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

    # ====== 最优权重保存（按验证损失）======
    os.makedirs(SAVE_DIR, exist_ok=True)
    fold_path = os.path.join(SAVE_DIR, f"{SAVE_BASENAME}_fold{fold+1}.pt")
    best_loss = float('inf')

    # 获取原始训练索引
    train_indices = torch.tensor(train_loader.dataset.indices, dtype=torch.long)

    with tqdm(total=EPOCHS,
              desc=f"Fold {fold+1}",
              ncols=100,
              bar_format='{l_bar}{n_fmt}/{total_fmt} |{bar}|') as pbar:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            # ===== 采样周期：用 sampling.py 选择难样本并重建 train_loader =====
            if (epoch > 0) and (epoch % SELECTED_EPOCH == 0) and (USE_HARD_NEGATIVE or USE_HARD_POSITIVE):
                sampling_loader = DataLoader(
                    train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=custom_collate_fn, num_workers=SAMPLING_NUM_WORKERS
                )

                selected_indices = [train_indices]

                if USE_HARD_NEGATIVE:
                    hard_neg_indices = select_hard_negatives(
                        model, sampling_loader, k=N_SAMPLES,
                        confidence_threshold=CONFIDENCE_THRESHOLD_NEGATIVE, device=DEVICE
                    )
                    if hard_neg_indices.numel() > 0:
                        selected_indices.append(hard_neg_indices)

                if USE_HARD_POSITIVE:
                    hard_pos_indices = select_hard_positives(
                        model, sampling_loader, k=N_SAMPLES,
                        confidence_threshold=CONFIDENCE_THRESHOLD_POSITIVE, device=DEVICE
                    )
                    if hard_pos_indices.numel() > 0:
                        selected_indices.append(hard_pos_indices)

                # ====== 重建数据集（允许困难样本重复采样，不做去重）======
                combined_indices = torch.cat(selected_indices)
                combined_indices = combined_indices.cpu().numpy()

                train_subset = Subset(train_loader.dataset.dataset, combined_indices)
                train_loader = DataLoader(
                    train_subset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=custom_collate_fn
                )
                print(f"Epoch {epoch + 1}: Updated train_loader with {len(combined_indices)} samples.")

            # ===== 常规训练 =====
            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                graph_data = batch["graph_data"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # 前向：若需要对比学习，则取出序列/图表征
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

                # ===== MLSC：多标签监督对比学习（新增，但不改原有风格）=====
                if use_mlsc and logits.size(0) > 1:
                    # 选择用于监督对比的表征：支持“序列/图/融合”
                    if MLSC_USE_FUSED_Z:
                        # 先分别归一化再做平均，最后再归一化（稳健，维度与单路一致）
                        z_seq = F.normalize(seq_embed, dim=1)
                        z_graph = F.normalize(graph_embed, dim=1)
                        z = F.normalize((z_seq + z_graph) / 2.0, dim=1)
                    else:
                        # 若不融合，默认采用序列表征（与你一阶段保持一致）
                        z = F.normalize(seq_embed, dim=1)

                    # 关键：MLSC 用多标签监督（labels 为多热向量）；内部按“至少共享一个标签”为正样本
                    loss_mlsc = mlsc_loss_fn(z, labels)
                    loss = loss + MLSC_WEIGHT * loss_mlsc
                # ===== MLSC 结束 =====

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # ===== 按平均训练损失保存最优权重（并覆盖最新）=====
            avg_loss = total_loss / max(1, len(train_loader))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), fold_path)
                print(f"Fold {fold + 1}: model saved (avg_loss={best_loss:.4f}) -> {fold_path}")

            # 验证损失计算
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

    # 使用标签矩阵作为分层依据
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
        metrics = train_single_model(fold, model, train_loader, val_loader, df, tokenizer)
        all_metrics.append(metrics)

    print("\n========== Average Metrics Across Folds ==========")
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
