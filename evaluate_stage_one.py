import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    return {"ACC": acc, "F1": f1, "AUC": auc, "SN": sn, "SP": sp}
