import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv, global_mean_pool


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




