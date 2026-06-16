import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
import copy
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph, to_undirected
class GGCLoss(nn.Module):
    def __init__(self, temperature=0.5, strategy="mask+edge"):
        super().__init__()
        self.temperature = temperature
        self.strategy = strategy

    def forward(self, gcn_encoder, graph_data):
        # graph_data 是一个 Batch，其中包含多个小图

        # 1. 对每个图做两次不同增强，得到两个 Batch
        graphs_1 = [self._augment(g) for g in graph_data.to_data_list()]
        graphs_2 = [self._augment(g) for g in graph_data.to_data_list()]
        batch_1 = Batch.from_data_list(graphs_1).to(graph_data.x.device)
        batch_2 = Batch.from_data_list(graphs_2).to(graph_data.x.device)

        # 2. 使用共享 GCN 编码器编码两批数据
        z1 = gcn_encoder(batch_1)  # shape: (B, d)
        z2 = gcn_encoder(batch_2)  # shape: (B, d)

        # 3. 计算对比损失
        return self.nt_xent_loss(z1, z2)

    def nt_xent_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        sim_exp = torch.exp(sim)

        # 避免自对比
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_exp = sim_exp.masked_fill(mask, 0)

        # 正样本相似度（对称位置）
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # 分母：每个样本对其余样本的相似度求和
        denom = sim_exp.sum(dim=1)
        loss = -torch.log(pos / denom).mean()
        return loss

    def _augment(self, data):                         # 可以组合使用
        data = copy.deepcopy(data)
        if self.strategy == "mask_backbone_atoms":
            data = self._mask_backbone_atoms(data, ratio=0.2)
        elif self.strategy == "random_mask_nodes":
            data = self._random_mask_nodes(data, ratio=0.2)
        elif self.strategy == "drop_non_backbone_nodes":
            data = self._drop_non_backbone_nodes(data, ratio=0.2)
        elif self.strategy == "perturb_edges_and_nodes":
            data = self._perturb_edges_and_nodes(data, edge_ratio=0.1, node_ratio=0.1)
        return data

    def _mask_backbone_atoms(self, data, ratio=0.2):   # 掩蔽碳骨架原子特征（如C-N主链）
        device = data.x.device
        backbone_atoms = torch.tensor([6, 7, 8], device=device)
        backbone_mask = torch.isin(data.atom_type, backbone_atoms)
        backbone_idx = backbone_mask.nonzero(as_tuple=True)[0]
        if len(backbone_idx) == 0:
            return data
        num_mask = max(1, int(ratio * len(backbone_idx)))
        mask_idx = backbone_idx[torch.randperm(len(backbone_idx), device=device)[:num_mask]]
        data.x[mask_idx] = data.x[mask_idx] + torch.randn_like(data.x[mask_idx]) * 0.1  #使用随机噪声模拟不确定性
        return data

    def _random_mask_nodes(self, data, ratio=0.2):    # 随机掩蔽节点特征
        device = data.x.device
        num_nodes = data.num_nodes
        num_mask = int(num_nodes * ratio)
        mask_idx = torch.randperm(num_nodes, device=device)[:num_mask]
        prob = torch.rand(num_mask, device=device)
        noise = torch.randn_like(data.x[mask_idx]) * 0.1
        data.x[mask_idx] = prob.view(-1, 1) * noise

        return data

    def _drop_non_backbone_nodes(self, data, ratio=0.2, min_nodes=5):
        device = data.x.device
        backbone_atoms = torch.tensor([6, 7, 8], device=device)  # C, N, O
        is_backbone = torch.isin(data.atom_type, backbone_atoms)
        is_non_backbone = ~is_backbone
        non_backbone_idx = is_non_backbone.nonzero(as_tuple=True)[0]

        if len(non_backbone_idx) == 0:
            return data  # 没有非骨架原子，跳过增强

        num_drop = int(len(non_backbone_idx) * ratio)
        if num_drop == 0:
            return data

        drop_idx = non_backbone_idx[torch.randperm(len(non_backbone_idx), device=device)[:num_drop]]
        keep_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        keep_mask[drop_idx] = False

        # 至少保留 min_nodes
        if keep_mask.sum() < min_nodes:
            return data  # 不做增强，保结构

        # 使用 subgraph 保持连贯性
        edge_index, _ = subgraph(keep_mask, data.edge_index, relabel_nodes=True)
        edge_index = to_undirected(edge_index)

        data.edge_index = edge_index
        data.x = data.x[keep_mask]
        data.atom_type = data.atom_type[keep_mask]
        data.batch = data.batch[keep_mask]

        return data

    def _perturb_edges_and_nodes(self, data, edge_ratio=0.1, node_ratio=0.1, min_nodes=5, min_edges=1):
        device = data.x.device
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)

        # ========== 边扰动 ==========
        if num_edges == 0:
            # 无边图直接返回原图，或你可以跳过该样本
            return data

        edge_keep_mask = torch.rand(num_edges, device=device) > edge_ratio
        if edge_keep_mask.sum() < min_edges:
            if num_edges >= min_edges:
                edge_keep_mask[torch.randint(0, num_edges, (min_edges,), device=device)] = True
            else:
                # 只有少于min_edges的边，就全保留
                edge_keep_mask[:] = True

        edge_index = data.edge_index[:, edge_keep_mask]
        edge_index = to_undirected(edge_index)

        # ========== 节点扰动 ==========
        if num_nodes == 0:
            return data  # 无节点，直接返回

        node_keep_mask = torch.rand(num_nodes, device=device) > node_ratio
        if node_keep_mask.sum() < min_nodes:
            if num_nodes >= min_nodes:
                node_keep_mask[torch.randint(0, num_nodes, (min_nodes,), device=device)] = True
            else:
                node_keep_mask[:] = True  # 全保留

        # 子图生成（保留节点 & 边）
        edge_index, _ = subgraph(node_keep_mask, edge_index, relabel_nodes=True)

        # 若没有边，增强失效，返回原图
        if edge_index.size(1) == 0:
            return data

        # ========== 更新图结构 ==========
        data.edge_index = edge_index
        data.x = data.x[node_keep_mask]
        data.atom_type = data.atom_type[node_keep_mask]
        data.batch = data.batch[node_keep_mask]

        return data




