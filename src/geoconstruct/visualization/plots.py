"""Matplotlib-only plotting helpers. No seaborn (kept simple for paper figs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive, safe in headless / scripts
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history_json: str | Path, out_path: str | Path) -> None:
    history_json = Path(history_json)
    out_path = Path(out_path)
    with open(history_json, "r", encoding="utf-8") as f:
        hist = json.load(f)
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_loss = [h["val_loss"] for h in hist]
    val_acc = [h["val_acc"] for h in hist]
    val_f1 = [h["val_macro_f1"] for h in hist]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Loss")

    ax[1].plot(epochs, val_acc, label="val accuracy")
    ax[1].plot(epochs, val_f1, label="val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metric"); ax[1].legend()
    ax[1].set_title("Validation metrics")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray,
                          class_names: list[str],
                          out_path: str | Path,
                          normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=np.float32)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sums, 1), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                    ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tsne(features: np.ndarray, labels: Iterable[int],
              class_names: list[str], out_path: str | Path,
              perplexity: int = 30, seed: int = 42) -> None:
    """t-SNE scatter of fused features colored by predicted/true class."""
    from sklearn.manifold import TSNE
    labels = np.asarray(list(labels))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=seed, max_iter=1000)
    emb = tsne.fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(class_names):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, label=name)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("t-SNE of fused features (test)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
