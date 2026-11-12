"""评估指标：AUC 为主，ACC / F1 / 混淆矩阵为辅（完整教程第 5 节）。"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_metrics(y_true, y_prob):
    """二分类：y_prob 为正类概率，shape (N,)。"""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).tolist()
    return out


@torch.no_grad()
def predict(model, loader, device):
    """返回 (正类概率, 标签)，供 evaluate / demo 复用。"""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(y.view(-1).numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)
