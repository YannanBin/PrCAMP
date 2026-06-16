import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    双向 Cross-Attention 融合两个模态的全局向量：
      - s: [B, D] (SMILES/文本)
      - g: [B, D] (Graph/结构，已投影到同维度)
    步骤：
      1) s 作为 Query，g 作为 K/V -> s2g
      2) g 作为 Query，s 作为 K/V -> g2s
      3) 拼接 [s, g, s2g, g2s] -> MLP -> [B, D] 作为 fused
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.h = num_heads

        self.ln_s = nn.LayerNorm(dim)
        self.ln_g = nn.LayerNorm(dim)

        self.attn_s2g = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_g2s = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)

        # 将 [s, g, s2g, g2s] 共 4*D 维压回 D 维
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        s: [B, D]  (SMILES向量)
        g: [B, D]  (Graph向量，和 s 同维)
        return: [B, D]
        """
        # 归一化
        s_n = self.ln_s(s)
        g_n = self.ln_g(g)

        # 变成长度为1的序列，适配 MultiheadAttention(batch_first=True)
        s_seq = s_n.unsqueeze(1)  # [B, 1, D]
        g_seq = g_n.unsqueeze(1)  # [B, 1, D]

        # s->g：s 作为 Q，g 作为 K/V
        s2g, _ = self.attn_s2g(query=s_seq, key=g_seq, value=g_seq, need_weights=False)  # [B,1,D]
        s2g = s2g.squeeze(1)  # [B, D]

        # g->s：g 作为 Q，s 作为 K/V
        g2s, _ = self.attn_g2s(query=g_seq, key=s_seq, value=s_seq, need_weights=False)  # [B,1,D]
        g2s = g2s.squeeze(1)  # [B, D]

        # 融合并压回 D 维
        fused = self.mlp(torch.cat([s, g, s2g, g2s], dim=-1))  # [B, D]
        return fused
