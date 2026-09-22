"""metrics.py — 分类评估指标。

提供:
  - compute_all_metrics(): 计算全面分类指标
  - per_class_confusion_pairs(): 识别关键混淆对
"""

import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, classification_report)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算所有评估指标。

    Returns:
        dict with: accuracy, balanced_accuracy, macro_precision, macro_recall,
                   macro_f1, weighted_f1, per_class_*, top2_accuracy, confusion_matrix
    """
    n = len(np.unique(y_true))
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class
    p_cls, r_cls, f1_cls, supp = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class: Dict[str, Dict] = {}
    for i, cn in enumerate(class_names):
        if i < len(p_cls):
            per_class[cn] = {
                "precision": round(float(p_cls[i]), 4),
                "recall": round(float(r_cls[i]), 4),
                "f1": round(float(f1_cls[i]), 4),
                "support": int(supp[i]),
            }

    # Top-2 accuracy
    top2_acc = 0.0
    if y_proba is not None and y_proba.shape[1] >= 2:
        top2_preds = np.argsort(y_proba, axis=1)[:, -2:]
        top2_acc = float(np.mean([y_true[i] in top2_preds[i] for i in range(len(y_true))]))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4), "balanced_accuracy": round(bacc, 4),
        "macro_precision": round(float(p_macro), 4), "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4), "weighted_f1": round(float(f1_weighted), 4),
        "top2_accuracy": round(top2_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def key_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str]) -> Dict[str, int]:
    """识别关键混淆对。"""
    pairs = {}
    label_to_name = {i: cn for i, cn in enumerate(class_names)}
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            pair = f"{label_to_name[y_true[i]]}->{label_to_name[y_pred[i]]}"
            pairs[pair] = pairs.get(pair, 0) + 1
    return dict(sorted(pairs.items(), key=lambda x: -x[1]))
