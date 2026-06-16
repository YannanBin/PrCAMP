import torch
import torch.nn as nn
from transformers import AutoModel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')  # 忽略这个手性信息的警告
from model_enconder_graph import GATEncoder
from feature_fusion import CrossAttentionFusion


# ========================== 一阶段二分类 ==========================
class AMPBinaryClassifier(nn.Module):
    def __init__(self, pretrained_path="DeepChem/ChemBERTa-77M-MLM"):
        super().__init__()

        self.bert_dim = 384  # ChemBERTa hidden size
        self.gcn_dim = 128  # GCN 输出向量维度
        self.encoder = AutoModel.from_pretrained(pretrained_path)
        self.gat = GATEncoder()
        self.project = nn.Linear(self.gcn_dim, self.bert_dim)
        self.fusion = CrossAttentionFusion(dim=self.bert_dim, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),  # 使用拼接后的特征，维度是两者的和
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    def forward(self, input_ids,attention_mask,graph_data, return_features=False):
        # 编码SMILES序列
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)  # [B, 384]
        if pooled is None:  # 兜底：掩码平均池化
            last_hidden = outputs.last_hidden_state  # [B, L, 384]
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # 编码2D图结构
        gat_vec = self.gat(graph_data)  # [B, 128]
        gat_384 = self.project(gat_vec)  # 投影到384维 [B, 384]

        # 交叉注意力融合
        fused = self.fusion(pooled, gat_384)  # [B, 384]

        # 分类
        logits = self.classifier(fused)

        if return_features:
            return logits, pooled, gat_384  # 返回序列/图嵌入用于对比学习
        return logits

    def encode_sequence(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            last_hidden = outputs.last_hidden_state  # [B,  L, 384]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled  # [B,384]
