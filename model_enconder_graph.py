import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool
)
try:
    from torch_geometric.nn import dense_diff_pool
except Exception:
    from torch_geometric.nn.dense.diff_pool import dense_diff_pool

from torch_geometric.utils import to_dense_adj, to_dense_batch

from torch_geometric.nn import TransformerConv, global_mean_pool

# ========================== GCN ==========================
class GCNEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)
# ========================== GAT ==========================
class GATEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128,num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)  # 输出 64×4=256
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return global_mean_pool(x, batch)




# ========================== GIN ==========================
class GINEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128):
        super().__init__()
        self.nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin1 = GINConv(self.nn1)

        self.nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.gin2 = GINConv(self.nn2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gin1(x, edge_index))
        x = self.gin2(x, edge_index)
        return global_mean_pool(x, batch)

# ========================== GraphSAGE ==========================
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)
# ========================== DiffPool ==========================
class DiffPoolEncoder(nn.Module):
    """
    单层 DiffPool：SAGE 得到节点嵌入 Z（out_channels），
    另一套 SAGE + Linear 得到软分配 S（K），调用 diff_pool 后做 mean-pool。
    """
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=128, K=20, aggr="mean"):
        super().__init__()
        # 节点特征分支 Z
        self.embed_gnn1  = SAGEConv(in_channels,  hidden_channels, aggr=aggr)
        self.embed_gnn2  = SAGEConv(hidden_channels, out_channels,  aggr=aggr)
        # 分配矩阵分支 S
        self.assign_gnn1 = SAGEConv(in_channels,  hidden_channels, aggr=aggr)
        self.assign_gnn2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.assign_lin  = nn.Linear(hidden_channels, K)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:[N,F], batch:[N]

        # 1) 节点嵌入 Z（稀疏图上）
        z = F.relu(self.embed_gnn1(x, edge_index))
        z = self.embed_gnn2(z, edge_index)  # [N, out_channels]

        # 2) 软分配 S（稀疏图上），记得 softmax
        s = F.relu(self.assign_gnn1(x, edge_index))
        s = F.relu(self.assign_gnn2(s, edge_index))
        s = self.assign_lin(s)  # [N, K] logits
        s = F.softmax(s, dim=-1)  # [N, K]

        # 3) 转稠密批（pad 到同一 N_max）
        z_dense, z_mask = to_dense_batch(z, batch)  # [B, N_max, F_out], [B, N_max]
        s_dense, _ = to_dense_batch(s, batch)  # [B, N_max, K]
        adj_dense = to_dense_adj(edge_index, batch=batch)  # [B, N_max, N_max]

        # 4) 稠密 DiffPool
        x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(z_dense, adj_dense, s_dense, mask=z_mask)
        # x_pool: [B, K, F_out]

        # 5) 图级向量（对簇维求均值；你也可以 global_mean_pool 一下 batch 但这里 batch 已经对齐）
        graph_vec = x_pool.mean(dim=1)  # [B, F_out]
        return graph_vec
        # 若想把正则项加入损失： return graph_vec, (link_loss, ent_loss)
