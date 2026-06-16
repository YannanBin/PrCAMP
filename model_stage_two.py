from transformers import AutoModel
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')   # 忽略这个手性信息的警告
from model_enconder_graph import GATEncoder
from feature_fusion import CrossAttentionFusion
import torch
import torch.nn as nn

class AMPFunctionClassifier(nn.Module):
    def __init__(self, pretrained_path="DeepChem/ChemBERTa-77M-MLM", num_labels=6):
        super().__init__()
        self.bert_dim = 384
        self.gat_dim = 128

        self.encoder = AutoModel.from_pretrained(pretrained_path)

        self.gat = GATEncoder()
        # 将GAT输出对齐到BERT维度

        self.project = nn.Linear(self.gat_dim, self.bert_dim)
        # 交叉注意力融合层   fusion和classifier顺序换结果会不同
        self.fusion = CrossAttentionFusion(dim=self.bert_dim, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        # 最好的性能这个顺序
        # self.encoder = AutoModel.from_pretrained(pretrained_path)
        #
        # self.gat = GATEncoder()
        # # 将GAT输出对齐到BERT维度
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, num_labels)
        # )
        # self.project = nn.Linear(self.gat_dim, self.bert_dim)
        # # 交叉注意力融合层
        #
        # self.fusion = CrossAttentionFusion(dim=self.bert_dim, num_heads=8, dropout=0.1)


    def forward(self,input_ids ,attention_mask,graph_data, return_features=True):
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooled = outputs.pooler_output  # [B, 384]
        # gat_vec = self.gat(graph_data)  # [B,128] or 节点 depending on fusion
        # gat_384 = self.project(gat_vec)
        # # combined = torch.cat([pooled, gat_vec], dim=1)
        # fused = self.fusion(pooled, gat_384)  # [B,384]  # [B,384]
        # logits = self.classifier(fused)  # 只用 fused
        # return logits

        # 编码SMILES序列
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooled = outputs.pooler_output  # [B, 384]

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
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                last_hidden = outputs.last_hidden_state  # [B,  L, 384]
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled  # [B,384]

        ## 去掉 no_grad，让 SSC 有梯度
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # pooled = getattr(outputs, "pooler_output", None)
        # if pooled is None:
        #     last_hidden = outputs.last_hidden_state  # [B,  L, 384]
        #     mask = attention_mask.unsqueeze(-1).float()
        #     pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        # return pooled  # [B,384]


# from transformers import AutoModel
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.warning')   # 忽略这个手性信息的警告
# from model_enconder_graph import GCNEncoder,GATEncoder,GINEncoder,GraphSAGEEncoder,DiffPoolEncoder
# from feature_fusion import GatedFusion,SelfAttentionFusion,CrossAttentionFusion
# import torch
# import torch.nn as nn
#
# class AMPFunctionClassifier(nn.Module):
#     def __init__(self, pretrained_path="./models/ChemBERTa", num_labels=6):
#         super().__init__()
#         self.bert_dim = 384
#         self.gat_dim = 128
#
#         self.encoder = AutoModel.from_pretrained(pretrained_path)
#
#         self.gat = GATEncoder()
#         # 将GAT输出对齐到BERT维度
#
#         self.project = nn.Linear(self.gat_dim, self.bert_dim)
#         # 简单拼接融合   fusion和classifier顺序换结果会不同
#         # self.fusion = CrossAttentionFusion(dim=self.bert_dim, num_heads=8, dropout=0.1)
#         self.classifier = nn.Sequential(
#             nn.Linear(384 * 2, 256),   # 拼接后 384 + 384 = 768
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, num_labels)
#         )
#         # 最好的性能这个顺序
#         # self.encoder = AutoModel.from_pretrained(pretrained_path)
#         #
#         # self.gat = GATEncoder()
#         # # 将GAT输出对齐到BERT维度
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(384, 256),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(256, num_labels)
#         # )
#         # self.project = nn.Linear(self.gat_dim, self.bert_dim)
#         # # 交叉注意力融合层
#         #
#         # self.fusion = CrossAttentionFusion(dim=self.bert_dim, num_heads=8, dropout=0.1)
#
#
#     def forward(self,input_ids ,attention_mask,graph_data, return_features=True):
#         # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         # pooled = outputs.pooler_output  # [B, 384]
#         # gat_vec = self.gat(graph_data)  # [B,128] or 节点 depending on fusion
#         # gat_384 = self.project(gat_vec)
#         # # combined = torch.cat([pooled, gat_vec], dim=1)
#         # fused = self.fusion(pooled, gat_384)  # [B,384]  # [B,384]
#         # logits = self.classifier(fused)  # 只用 fused
#         # return logits
#
#         # 编码SMILES序列
#         # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         # pooled = outputs.pooler_output  # [B, 384]
#
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
#         # 简单拼接融合
#         # fused = self.fusion(pooled, gat_384)  # [B, 384]
#         fused = torch.cat([pooled, gat_384], dim=1)  # [B, 768]
#
#         # 分类
#         logits = self.classifier(fused)
#
#         if return_features:
#             return logits, pooled, gat_384  # 返回序列/图嵌入用于对比学习
#         return logits
#
#     def encode_sequence(self, input_ids, attention_mask):
#         with torch.no_grad():
#             outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#             pooled = getattr(outputs, "pooler_output", None)
#             if pooled is None:
#                 last_hidden = outputs.last_hidden_state  # [B,  L, 384]
#                 mask = attention_mask.unsqueeze(-1).float()
#                 pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#         return pooled  # [B,384]
#         # # 去掉 no_grad，让 SSC 有梯度
#         # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         # pooled = getattr(outputs, "pooler_output", None)
#         # if pooled is None:
#         #     last_hidden = outputs.last_hidden_state  # [B,  L, 384]
#         #     mask = attention_mask.unsqueeze(-1).float()
#         #     pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
#         # return pooled  # [B,384]
