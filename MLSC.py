# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MLSCLoss(nn.Module):
#     """
#     Multi-Label Supervised Contrastive Learning (MulSupCon)
#     论文提出的多标签监督对比学习损失（AAAI 2024）。
#     - 批内：对每个 anchor 样本 i，对其每个正标签 j 分别形成正样本集合 P_{i,j}（同标签 j 的其它样本）
#     - 对每个 (i, j)：取该标签下所有正样本的 log-softmax 概率的平均（按 1/|P_{i,j}|），再对 j 累加
#     - 对整个 batch：用 “有效 (i,j) 对”的计数进行归一（∑_i |y_i| 的有效项），不使用 1/|y(i)| 的样本级权重
#       （对应论文中更优的设定）
#
#     记号对应论文：
#       s_{i,a} = z_i · z_a / τ
#       L_i = Σ_{j∈y(i)} [ - (1/|P_{i,j}|) Σ_{p∈P_{i,j}} log( exp(s_{i,p}) / Σ_{a≠i} exp(s_{i,a}) ) ]
#       L = (1 / Σ_i |y(i)|_valid) Σ_i L_i
#
#     参数:
#         temperature: 温度 τ
#         normalize: 是否对输入特征做 L2 归一化
#         eps: 数值稳定项
#         use_per_sample_weight: 是否额外乘 1/|y(i)|（论文消融显示不加更好，默认 False）
#     """
#     def __init__(self, temperature: float = 0.1, normalize: bool = True,
#                  eps: float = 1e-12, use_per_sample_weight: bool = False):
#         super().__init__()
#         self.tau = temperature
#         self.normalize = normalize
#         self.eps = eps
#         self.use_per_sample_weight = use_per_sample_weight  # 不建议打开
#
#     def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """
#         输入:
#             z: [B, D]  批内样本表征（任意模态/融合的向量都可以）
#             labels: [B, C]  多标签 one-hot/多热（二值 {0,1} 或 float 0/1）
#         输出:
#             loss: 标量
#         """
#         assert z.dim() == 2 and labels.dim() == 2, "z:[B,D], labels:[B,C]"
#         B, D = z.shape
#         device = z.device
#
#         # 归一化以稳定相似度
#         if self.normalize:
#             z = F.normalize(z, p=2, dim=1)
#
#         # 相似度矩阵 + 温度
#         sim = torch.matmul(z, z.t()) / max(self.tau, self.eps)  # [B,B]
#
#         # 排除自对比：对角置为 -inf，使其不进入 softmax 分母与后续求和
#         diag_mask = torch.eye(B, dtype=torch.bool, device=device)
#         sim_masked = sim.masked_fill(diag_mask, float("-inf"))
#
#         # 对每个 anchor 的分母 log Σ_{a≠i} exp(sim_{i,a})
#         denom_log = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # [B,1]
#         # pair-wise 的对数概率：log( exp(sim_{i,a}) / Σ_{a≠i} exp(sim_{i,a}) )
#         log_probs = sim_masked - denom_log  # [B,B]，对角为 -inf
#
#         # 标签张量转 bool
#         Y = labels > 0.5  # [B,C]，支持 float/bool/int
#         C = Y.shape[1]
#         if C == 0:
#             return torch.zeros([], device=device, dtype=z.dtype)
#
#         # 构造每个标签的 (B,B) 正样本掩码：M_j[i,p] = 1 ↔ 样本 i 与 p 都含标签 j，且 p≠i
#         # 先得到 [C,B]，再广播到 [C,B,B]
#         Yt = Y.t().contiguous()                     # [C,B]
#         M = (Yt.unsqueeze(2) & Yt.unsqueeze(1))     # [C,B,B]
#         # 去掉对角
#         M[:, torch.arange(B), torch.arange(B)] = False
#
#         # 每个 (i,j) 的正样本个数 |P_{i,j}|，形状 [C,B]
#         P_counts = M.sum(dim=2)  # 沿列聚合，得到每个 anchor i 在标签 j 下的正样本计数
#
#         # 将 log_probs 扩到 [C,B,B]，只对正样本位置求和
#         lp = log_probs.unsqueeze(0)                 # [1,B,B]
#         sum_logp_pos = (lp * M).sum(dim=2)         # [C,B]，Σ_{p∈P_{i,j}} log_prob(i,p)
#
#         # 有效 (i,j)：至少有 1 个正样本
#         valid = P_counts > 0                        # [C,B]
#         # (1/|P_{i,j}|) * Σ log_prob
#         avg_logp = torch.zeros_like(sum_logp_pos)
#         avg_logp[valid] = sum_logp_pos[valid] / (P_counts[valid].to(sum_logp_pos.dtype) + self.eps)
#
#         # - 平均对数概率（各标签求和）
#         per_i = -avg_logp.sum(dim=0)               # [B]，Σ_j ...
#         # 是否乘 1/|y(i)|（论文消融显示不乘更优，默认不乘）
#         if self.use_per_sample_weight:
#             yi_counts = Y.sum(dim=1).clamp_min(1)  # [B]
#             per_i = per_i / yi_counts.to(per_i.dtype)
#
#         # 归一化：除以有效 (i,j) 对的数量（Σ_i |y(i)|_valid）
#         total_pos_sets = valid.sum().clamp_min(1)  # 标量
#         loss = per_i.sum() / total_pos_sets.to(per_i.dtype)
#
#         # 若整个 batch 没有任何正对（极端小 batch 或标签全互斥），返回 0 以避免 NaN
#         if torch.isinf(loss) or torch.isnan(loss):
#             return torch.zeros([], device=device, dtype=z.dtype)
#         return loss
# MLSC.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSCLoss(nn.Module):
    """
    Multi-Label Supervised Contrastive Learning Loss (MulSupCon)
    基于论文: Multi-Label Supervised Contrastive Learning

    核心思想:
    - 对于多标签场景，根据标签重叠程度动态调整正样本权重
    - 对每个标签单独构建正样本集合，避免"ALL"和"ANY"策略的缺点
    """

    def __init__(self, temperature=0.1, normalize=True):
        """
        参数:
            temperature: 温度系数，控制分布的尖锐程度
            normalize: 是否对特征进行L2归一化
        """
        super(MLSCLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features, labels):
        """
        前向传播计算多标签监督对比损失

        参数:
            features: 特征向量 [batch_size, feature_dim]
            labels: 多热标签向量 [batch_size, num_classes]

        返回:
            loss: 多标签监督对比损失
        """
        batch_size, num_classes = labels.shape

        if self.normalize:
            features = F.normalize(features, dim=1)

        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 计算每个样本的损失
        total_loss = 0.0
        total_labels = 0

        for i in range(batch_size):
            # 当前锚点样本的标签
            anchor_labels = labels[i]  # [num_classes]

            # 计算每个标签对应的正样本集合
            label_loss = 0.0
            valid_labels = 0

            for j in range(num_classes):
                if anchor_labels[j] == 1:  # 锚点样本有这个标签
                    # 找到所有包含这个标签的样本
                    positive_mask = labels[:, j] == 1  # [batch_size]

                    if positive_mask.sum() > 0:
                        # 计算这个标签的对比损失
                        pos_count = positive_mask.sum().float()

                        # 计算logits
                        logits = similarity_matrix[i]  # [batch_size]

                        # 计算log_prob
                        log_prob = logits - torch.logsumexp(logits, dim=0)

                        # 计算这个标签的损失
                        label_loss += -torch.sum(log_prob[positive_mask]) / pos_count
                        valid_labels += 1

            if valid_labels > 0:
                total_loss += label_loss / valid_labels
                total_labels += 1

        if total_labels == 0:
            return torch.tensor(0.0, device=features.device)

        return total_loss / total_labels


