# SSC.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSCLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, seq_embed_1, seq_embed_2):
        """
        seq_embed_1: (B, D) - augmented SMILES view 1
        seq_embed_2: (B, D) - augmented SMILES view 2
        """
        z1 = F.normalize(seq_embed_1, dim=1)
        z2 = F.normalize(seq_embed_2, dim=1)

        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_i + loss_j) / 2
        return loss
