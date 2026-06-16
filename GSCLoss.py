import torch
import torch.nn as nn
import torch.nn.functional as F

class GSCLoss(nn.Module):
    """
    图-序列跨模态对比（Contrast 版）:
      - 正样本: 同一样本的 (graph_embed[i], seq_embed[i])
      - 负样本: 同一 batch 内的 (graph_embed[i], seq_embed[j]), j != i
    超参:
      hidden_dim: 输入表征维度（如 384）
      proj_dim:   投影到对比学习空间的维度
      temperature: 温度系数 tau
      lam:         双向权重（g->s 与 s->g）
    """
    def __init__(self, hidden_dim=384, proj_dim=384, temperature=0.1, lam=0.5, normalize=True):
        super(GSCLoss, self).__init__()
        self.temperature = temperature
        self.lam = lam
        self.normalize = normalize

        # 投影头（与 Contrast 一致：Linear-ELU-Linear）
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        # 初始化（与 Contrast 一致）
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, graph_embed, seq_embed):
        """
        graph_embed: [B, D]
        seq_embed:   [B, D]
        return: 标量 loss
        """
        # 投影到对比空间
        z_g = self.proj(graph_embed)  # [B, P]
        z_s = self.proj(seq_embed)    # [B, P]

        # 归一化后余弦相似度更稳
        if self.normalize:
            z_g = F.normalize(z_g, dim=1)
            z_s = F.normalize(z_s, dim=1)

        # 相似度矩阵（带温度），用 exp 转成正值权重
        # sim[i, j] = exp( cos(z_g[i], z_s[j]) / tau )
        sim = torch.exp(torch.matmul(z_g, z_s.t()) / (self.temperature + 1e-8))  # [B, B]
        sim_t = sim.t()

        # 正样本（对角）与负样本（行/列和减去对角）
        pos_g2s = torch.diag(sim)            # graph->seq
        neg_g2s = sim.sum(dim=1) - pos_g2s

        pos_s2g = torch.diag(sim_t)          # seq->graph
        neg_s2g = sim_t.sum(dim=1) - pos_s2g

        eps = 1e-8
        # 双向损失
        loss_g2s = -torch.log(pos_g2s / (neg_g2s + eps)).mean()
        loss_s2g = -torch.log(pos_s2g / (neg_s2g + eps)).mean()

        loss = self.lam * loss_g2s + (1.0 - self.lam) * loss_s2g
        return loss
