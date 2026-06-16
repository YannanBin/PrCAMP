import torch
import torch.nn as nn

# ========================== 门控融合 ==========================
class GatedFusion(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.dim = dim
        # 预归一化让两路分布可比
        self.ln_a = nn.LayerNorm(dim)
        self.ln_b = nn.LayerNorm(dim)

        # 门控信号生成器（逐维），保持你的风格
        self.gate = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

        # 轻量 FFN + 残差，提升表示力且更稳
        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # 残差上的小 dropout，防过拟合
        self.resid_drop = nn.Dropout(0.1)

        # 可学注入强度（BERT 为主，初始略小）
        self.inject_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, vec_a, vec_b):
        # 预归一化
        a = self.ln_a(vec_a)             # BERT 向量
        b = self.ln_b(vec_b)             # 图向量(已投到同维)

        concated = torch.cat([a, b], dim=-1)         # [B, 2*D]
        z = self.sigmoid(self.gate(concated))        # [B, D] 门控

        # 门控注入（保持你的融合形式），然后把“变化量”注入到原始 BERT 上（BERT 为主）
        base = vec_a                                    # 用未归一化的 BERT 做残差锚点
        injected = (1 - z) * a + z * b                  # [B,D]
        x = base + self.resid_drop(torch.sigmoid(self.inject_scale) * (injected - a))

        # 轻量 FFN 残差
        x = x + self.resid_drop(self.ffn(self.ffn_ln(x)))
        return x


# ========================== 自注意力融合（2 token） ==========================
class SelfAttentionFusion(nn.Module):
    """
    两个 token（BERT, GAT）做缩放点积注意力：
    - 仅取第 0 行输出 out[:,0,:]，即“BERT 这一行”的上下文（BERT 为主）
    - 预归一化 + 注意力/残差 dropout + 轻量 FFN
    """
    def __init__(self, dim=384):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.ln_a = nn.LayerNorm(dim)
        self.ln_b = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)

        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

        # 可学注入强度（稳）
        self.inject_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_vec, gat_vec):
        # 预归一化
        a = self.ln_a(bert_vec)      # [B,D]
        b = self.ln_b(gat_vec)       # [B,D]

        stack = torch.stack([a, b], dim=1)              # [B,2,D]   token0=BERT, token1=GAT
        Q = self.query(stack)                           # [B,2,D]
        K = self.key(stack)                             # [B,2,D]
        V = self.value(stack)                           # [B,2,D]

        attn = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # [B,2,2]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = attn @ V                                        # [B,2,D]
        ctx = out[:, 0, :]                                    # 只取 BERT 这一行（BERT 为主）

        # 把上下文注入原始 BERT
        x = bert_vec + self.resid_drop(torch.sigmoid(self.inject_scale) * (ctx - a))
        # 轻量 FFN 残差
        x = x + self.resid_drop(self.ffn(self.ffn_ln(x)))
        return x

# ========================== 交叉注意力融合 ==========================
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


# class CrossAttentionFusion(nn.Module):
#     def __init__(self, dim=384, num_heads=8):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#
#         # 线性变换层，用于生成Q、K、V
#         self.q_linear = nn.Linear(dim, dim)
#         self.k_linear = nn.Linear(dim, dim)
#         self.v_linear = nn.Linear(dim, dim)
#
#         # 输出线性层和LayerNorm
#         self.out_linear = nn.Linear(dim, dim)
#         self.layer_norm = nn.LayerNorm(dim)
#
#     def forward(self, smiles_feat, graph_feat):
#         """
#         smiles_feat: [B, 384] 来自BERT的SMILES特征
#         graph_feat: [B, 384] 来自GAT的图特征（已投影到384维）
#         """
#         batch_size = smiles_feat.size(0)
#
#         # 生成Q、K、V
#         Q = self.q_linear(smiles_feat)  # [B, 384]
#         K = self.k_linear(graph_feat)  # [B, 384]
#         V = self.v_linear(graph_feat)  # [B, 384]
#
#         # 多头切分
#         Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # [H, B, D_head]
#         K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
#         V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
#
#         # 计算注意力分数
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [H, B, B]
#         attn_weights = F.softmax(scores, dim=-1)
#
#         # 应用注意力
#         context = torch.matmul(attn_weights, V)  # [H, B, D_head]
#         context = context.transpose(0, 1).contiguous().view(batch_size, self.dim)  # [B, 384]
#
#         # 输出变换和残差连接
#         output = self.out_linear(context)
#         output = self.layer_norm(output + smiles_feat)  # 残差连接 + LayerNorm
#
#         return output



# class CrossAttentionFusion(nn.Module):
#     def __init__(self, dim=384,bert_weight=1, gat_weight=1):
#         super().__init__()
#         self.query = nn.Linear(dim, dim)
#         self.key = nn.Linear(dim, dim)
#         self.value = nn.Linear(dim, dim)
#         self.scale = dim ** -0.5  # 缩放因子
#
#         self.bert_weight = bert_weight  # BERT特征权重系数
#         self.gat_weight = gat_weight  # GCN特征权重系数
#
#     def forward(self, bert_vec, gcn_vec):
#         Q = self.query(bert_vec)  # [B, D] -> [B, D]
#         K = self.key(gcn_vec)  # [B, D] -> [B, D]
#         V = self.value(gcn_vec)  # [B, D] -> [B, D]
#
#         # 计算注意力分数
#         attn_scores = torch.einsum('bd,bd->b', Q, K) * self.scale  # [B]
#         attn_weights = torch.softmax(attn_scores, dim=0)  # [B]
#
#         # 加权融合 (按注意力权重融合GCN信息到BERT)
#         fused = self.gat_weight * attn_weights[:, None] * V + self.bert_weight * bert_vec
#         return fused

