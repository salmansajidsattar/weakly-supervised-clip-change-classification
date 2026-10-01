"""Classification loss with optional class-balanced weighting.

For weakly-supervised LEVIR-CC labels we expect heavy class imbalance
(no_change and completed will dominate, active_construction will be rare).
We default to class-frequency-inverse weighting computed on the training
labels file.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def compute_class_weights(labels_json: str | Path, num_classes: int,
                          smoothing: float = 1.0) -> torch.Tensor:
    """Inverse-frequency weights with additive smoothing."""
    labels_json = Path(labels_json)
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    counts = Counter(int(v) for v in labels.values() if int(v) >= 0)
    freqs = torch.tensor(
        [counts.get(c, 0) + smoothing for c in range(num_classes)],
        dtype=torch.float32,
    )
    weights = 1.0 / freqs
    weights = weights * (num_classes / weights.sum())
    log.info("Class weights: %s", weights.tolist())
    return weights


def build_loss(cfg, train_labels_json: str | Path | None = None) -> nn.Module:
    if cfg.loss.get("class_weighted", True) and train_labels_json is not None:
        w = compute_class_weights(train_labels_json, cfg.head.num_classes)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.loss.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=cfg.loss.get("label_smoothing", 0.0))
