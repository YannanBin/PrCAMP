import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSCLoss(nn.Module):
    """
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


