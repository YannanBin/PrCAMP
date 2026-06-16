import torch
import torch.nn as nn
from transformers import AutoModel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')  # 忽略这个手性信息的警告
from model_enconder_graph import GCNEncoder, GATEncoder, GINEncoder, GraphSAGEEncoder, DiffPoolEncoder
from feature_fusion import GatedFusion, SelfAttentionFusion, CrossAttentionFusion


# ========================== 一阶段二分类 ==========================
class AMPBinaryClassifier(nn.Module):
    def __init__(self, pretrained_path="./models/ChemBERTa"):
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
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooled = outputs.pooler_output  # [batch, dim]
        # gat_vec = self.gat(graph_data)  # [batch, 128]
        # gat_384 = self.project(gat_vec)  # → [B, 384]
        # fused = self.fusion(pooled, gat_384)
        # # 拼接操作
        # # combined = torch.cat([pooled, gat_vec], dim=1)  # [B, 768] 拼接后维度为 384 + 384
        # logits = self.classifier(fused)  # 通过分类器输出结果
        # return logits
        # # if return_features:
        # #     return logits, pooled, gat_vec
        # # else:
        # #     return logits
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
        # with torch.no_grad():
        #     outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #     pooled = getattr(outputs, "pooler_output", None)
        #     if pooled is None:
        #         last_hidden = outputs.last_hidden_state  # [B,  L, 384]
        #         mask = attention_mask.unsqueeze(-1).float()
        #         pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        # return pooled  # [B,384]

        # 去掉 no_grad，让 SSC 有梯度
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            last_hidden = outputs.last_hidden_state  # [B,  L, 384]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled  # [B,384]
    
# import torch
# import torch.nn as nn
# from transformers import AutoModel
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.warning')  # 忽略这个手性信息的警告
# from model_enconder_graph import GCNEncoder, GATEncoder, GINEncoder, GraphSAGEEncoder, DiffPoolEncoder
# from feature_fusion import GatedFusion, SelfAttentionFusion, CrossAttentionFusion
#
#
# # ========================== 一阶段二分类 ==========================
# class AMPBinaryClassifier(nn.Module):
#     def __init__(self, pretrained_path="./models/ChemBERTa"):
#         super().__init__()
#
#         self.bert_dim = 384  # ChemBERTa hidden size
#         self.gcn_dim = 128  # GCN 输出向量维度（你设置的）
#         self.encoder = AutoModel.from_pretrained(pretrained_path)
#         self.gat = GATEncoder()
#         self.project = nn.Linear(self.gcn_dim, self.bert_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(2 * self.bert_dim, 256),  # 使用拼接后的特征，维度是两者的和
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, input_ids, attention_mask, graph_data, return_features=False):
#         # 编码SMILES序列
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = getattr(outputs, "pooler_output", None)  # [B, 384]
#         if pooled is None:  # 兜底：掩码平均池化
#             last_hidden = outputs.last_hidden_state  # [B, L, 384]
#             mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
#             pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#
#         # 编码2D图结构
#         gat_vec = self.gat(graph_data)  # [B, 128]
#         gat_384 = self.project(gat_vec)  # 投影到384维 [B, 384]
#
#         # 拼接操作：合并序列特征与图特征
#         combined = torch.cat([pooled, gat_384], dim=1)  # [B, 768]
#
#         # 分类
#         logits = self.classifier(combined)
#
#         if return_features:
#             return logits, pooled, gat_384  # 返回序列/图嵌入用于对比学习
#         return logits
#
#     def encode_sequence(self, input_ids, attention_mask):
#         # 去掉 no_grad，让 SSC 有梯度
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         pooled = getattr(outputs, "pooler_output", None)
#         if pooled is None:
#             last_hidden = outputs.last_hidden_state  # [B,  L, 384]
#             mask = attention_mask.unsqueeze(-1).float()
#             pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#         return pooled  # [B,384]



# 单序列
# import torch
# import torch.nn as nn
# from transformers import AutoModel
# from rdkit import RDLogger
#
# RDLogger.DisableLog('rdApp.warning')  # 忽略这个手性信息的警告
#
#
# # ========================== 一阶段二分类（仅序列版本） ==========================
# class AMPBinaryClassifier(nn.Module):
#     def __init__(self, pretrained_path="./models/ChemBERTa"):
#         super().__init__()
#
#         self.bert_dim = 384  # ChemBERTa hidden size
#         self.encoder = AutoModel.from_pretrained(pretrained_path)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.bert_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, input_ids, attention_mask, graph_data=None, return_features=False):
#         """
#         前向传播（仅使用序列数据）
#
#         参数:
#             input_ids: [B, L] 输入ID
#             attention_mask: [B, L] 注意力掩码
#             graph_data: 保留参数但不再使用
#             return_features: 是否返回特征用于对比学习
#         """
#         # 编码SMILES序列
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = getattr(outputs, "pooler_output", None)  # [B, 384]
#
#         if pooled is None:  # 兜底：掩码平均池化
#             last_hidden = outputs.last_hidden_state  # [B, L, 384]
#             mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
#             pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#
#         # 分类
#         logits = self.classifier(pooled)
#
#         if return_features:
#             return logits, pooled, None  # 返回序列嵌入用于对比学习，图嵌入为None
#         return logits
#
#     def encode_sequence(self, input_ids, attention_mask):
#         """
#         编码SMILES序列，用于对比学习
#
#         返回:
#             pooled: [B, 384] 序列表示
#         """
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         pooled = getattr(outputs, "pooler_output", None)
#
#         if pooled is None:
#             last_hidden = outputs.last_hidden_state  # [B, L, 384]
#             mask = attention_mask.unsqueeze(-1).float()
#             pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#
#         return pooled  # [B, 384]

# # 单结构
# import torch
# import torch.nn as nn
# from rdkit import RDLogger
#
# RDLogger.DisableLog('rdApp.warning')
# from model_enconder_graph import GATEncoder  # 只导入图编码器
#
#
# # ========================== 一阶段二分类（仅使用图数据） ==========================
# class AMPBinaryClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.gat_dim = 128  # GAT 输出向量维度
#         self.hidden_dim = 256  # 分类器隐藏层维度
#
#         # 仅使用图编码器
#         self.gat = GATEncoder()
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(self.gat_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(self.hidden_dim, 1)
#         )
#
#     def forward(self, graph_data, return_features=False):
#         """
#         前向传播（仅使用图数据）
#         Args:
#             graph_data: 图数据
#             return_features: 是否返回特征用于对比学习
#         Returns:
#             logits: 分类logits
#         """
#         # 编码2D图结构
#         gat_vec = self.gat(graph_data)  # [B, 128]
#
#         # 分类
#         logits = self.classifier(gat_vec)  # [B, 1]
#
#         if return_features:
#             return logits, gat_vec  # 返回logits和图嵌入用于对比学习
#         return logits
#
#     def encode_graph(self, graph_data):
#         """
#         编码图数据（用于对比学习）
#         Args:
#             graph_data: 图数据
#         Returns:
#             graph_embed: 图嵌入向量
#         """
#         return self.gat(graph_data)  # [B, 128]