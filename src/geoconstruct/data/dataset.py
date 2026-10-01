"""PyTorch Dataset that loads pre-extracted CLIP features from a .npz cache.

We never re-run CLIP during training. Feature extraction is a one-time
operation (see scripts/extract_clip_features.py). This keeps CPU training
trivial — each sample is just a memory read of two 512-d vectors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class CachedFeatureDataset(Dataset):
    """Yields (feat_a, feat_b, label, image_id) for a single split.

    Drops items with label == -1 (caption-rule ignored).
    """

    def __init__(
        self,
        feature_cache: str | Path,
        labels_json: str | Path,
        drop_ignore: bool = True,
    ):
        self.feature_cache = Path(feature_cache)
        self.labels_json = Path(labels_json)

        log.info("Loading feature cache: %s", self.feature_cache)
        cache = np.load(self.feature_cache, allow_pickle=True)
        # Expected keys: "feat_a" [N, D], "feat_b" [N, D], "ids" [N] (uint string)
        self._feat_a = cache["feat_a"]
        self._feat_b = cache["feat_b"]
        ids = cache["ids"]
        # Ensure ids are python strings
        self._ids = [str(x) for x in ids.tolist()]
        self._id_to_idx = {i: k for k, i in enumerate(self._ids)}

        log.info("Loading weak labels: %s", self.labels_json)
        with open(self.labels_json, "r", encoding="utf-8") as f:
            self._labels = json.load(f)

        # Build a list of (cache_idx, label) for items present in both
        self._index: list[tuple[int, int]] = []
        missing = 0
        for img_id, label in self._labels.items():
            if drop_ignore and label < 0:
                continue
            idx = self._id_to_idx.get(img_id)
            if idx is None:
                missing += 1
                continue
            self._index.append((idx, int(label)))
        if missing:
            log.warning(
                "%d labeled items had no cached features (mismatch).", missing
            )
        log.info("Dataset size after filtering: %d", len(self._index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        cache_idx, label = self._index[i]
        feat_a = torch.from_numpy(self._feat_a[cache_idx]).float()
        feat_b = torch.from_numpy(self._feat_b[cache_idx]).float()
        return feat_a, feat_b, label, self._ids[cache_idx]


def load_cached_split(features_dir: str | Path, labels_dir: str | Path, split: str):
    """Convenience constructor: features_dir/{split}_features.npz +
    labels_dir/{split}_labels.json."""
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    return CachedFeatureDataset(
        feature_cache=features_dir / f"{split}_features.npz",
        labels_json=labels_dir / f"{split}_labels.json",
    )
