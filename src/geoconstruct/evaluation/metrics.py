"""Classification metrics: accuracy, macro/weighted F1, per-class P/R/F1, CM."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int],
                    num_classes: int) -> dict:
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    labels = list(range(num_classes))

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels,
                           average="weighted", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": s.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }


def format_report(y_true: Iterable[int], y_pred: Iterable[int],
                  target_names: list[str]) -> str:
    return classification_report(
        list(y_true), list(y_pred), target_names=target_names, zero_division=0
    )
