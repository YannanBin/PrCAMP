import numpy as np


def _binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """将概率/得分按阈值二值化"""
    return (probs > threshold).astype(int)


def Aiming(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Aiming（样本级 Precision）：平均每个样本“命中的预测标签 / 预测为正的标签”
    预测全为负时，按 0 贡献。
    """
    n, _ = y_hat.shape
    score = 0.0
    for v in range(n):
        inter = np.sum((y_hat[v] == 1) & (y[v] == 1))
        pred_pos = np.sum(y_hat[v])
        score += 0.0 if pred_pos == 0 else inter / pred_pos
    return score / n if n > 0 else 0.0


def Coverage(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Coverage（样本级 Recall）：平均每个样本“命中的预测标签 / 真实为正的标签”
    真实全为负时，按 0 贡献。
    """
    n, _ = y_hat.shape
    score = 0.0
    for v in range(n):
        inter = np.sum((y_hat[v] == 1) & (y[v] == 1))
        true_pos = np.sum(y[v])
        score += 0.0 if true_pos == 0 else inter / true_pos
    return score / n if n > 0 else 0.0


def Accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Accuracy（样本级 Jaccard）：平均每个样本“交集 / 并集”
    若并集为 0（预测与真实均全负），按照你原训练代码的定义，记为 1.0。
    """
    n, _ = y_hat.shape
    score = 0.0
    for v in range(n):
        inter = np.sum((y_hat[v] == 1) & (y[v] == 1))
        union = np.sum((y_hat[v] == 1) | (y[v] == 1))
        score += 1.0 if union == 0 else inter / union
    return score / n if n > 0 else 0.0


def AbsoluteTrue(y_hat: np.ndarray, y: np.ndarray) -> float:
    """样本级完全匹配比例（预测集合与真实集合完全一致）"""
    n, _ = y_hat.shape
    exact = sum(int(np.array_equal(y_hat[v], y[v])) for v in range(n))
    return exact / n if n > 0 else 0.0


def AbsoluteFalse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    AbsoluteFalse（等价于样本级 Hamming Loss）：
    每个样本“错误标签数 / 标签总数”的平均值。
    错误标签数 = XOR =（并集 - 交集）。
    """
    n, m = y_hat.shape
    score = 0.0
    for v in range(n):
        union = np.sum((y_hat[v] == 1) | (y[v] == 1))
        inter = np.sum((y_hat[v] == 1) & (y[v] == 1))
        score += (union - inter) / m
    return score / n if n > 0 else 0.0


def compute_multilabel_metrics(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               threshold: float = 0.5) -> dict:
    """
    计算并返回你需要的 5 个多标签指标。
    传入：
      - y_true: (N, M) 的 0/1 标签
      - y_prob: (N, M) 的概率/分数（会按阈值二值化）
      - threshold: 二值化阈值（默认 0.5）
    返回：
      dict 包含 precision/coverage/accuracy/absolute_true/absolute_false
    """
    y_true = y_true.astype(int)
    y_hat = _binarize(y_prob, threshold=threshold)

    return {
        'precision': Aiming(y_hat, y_true),
        'coverage': Coverage(y_hat, y_true),
        'accuracy': Accuracy(y_hat, y_true),
        'absolute_true': AbsoluteTrue(y_hat, y_true),
        'absolute_false': AbsoluteFalse(y_hat, y_true),
    }
