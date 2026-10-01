# Lightweight package init. dataset.py is lazy-imported to keep label-building
# scripts runnable without a torch install.

from .captions import CLASS_NAMES, CLASS_TEMPLATES, build_weak_labels, vote_label

__all__ = [
    "CLASS_NAMES",
    "CLASS_TEMPLATES",
    "build_weak_labels",
    "vote_label",
    "CachedFeatureDataset",
    "load_cached_split",
]


def __getattr__(name):
    if name in ("CachedFeatureDataset", "load_cached_split"):
        from . import dataset as _dataset
        return getattr(_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
