"""t-SNE visualisation of frozen CLIP features on the test split.

Addresses Reviewer 1 comment #6 (part 2): "A t-SNE visualization of frozen
CLIP features colored by weak label would make linear separability visually
apparent."

For each fusion mode (t2, diff, concat), compute a 2D t-SNE embedding of
the 1849 test features, colour points by weak label, and save a PNG.

Output
------
outputs/figures/tsne_test_{mode}.png    for mode in {t2, diff, concat}
outputs/figures/tsne_combined.png       three-panel side-by-side

Runtime
-------
* CPU: ~5-15 minutes total (t-SNE is O(n^2))
* GPU: not required
* Disk: three ~500 KB PNGs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")

log = logging.getLogger("make_tsne_figure")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def fuse(fa, fb, mode):
    if mode == "t2": return fb
    if mode == "diff": return fb - fa
    if mode == "concat": return np.concatenate([fa, fb], axis=-1)
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "outputs/features/test_features.npz"))
    ap.add_argument("--labels",   default=str(ROOT / "outputs/labels/test_labels.json"))
    ap.add_argument("--out-dir",  default=str(ROOT / "outputs/figures"))
    ap.add_argument("--modes",    nargs="+", default=["t2", "diff", "concat"])
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-iter",   type=int, default=1000)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading features from {args.features}")
    z = np.load(args.features, allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    fa = z["feat_a"]; fb = z["feat_b"]

    log.info(f"Loading labels from {args.labels}")
    with open(args.labels) as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
    idx_ok = [i for i, pid in enumerate(ids) if pid in labels]
    fa = fa[idx_ok]; fb = fb[idx_ok]
    y = np.array([labels[ids[i]] for i in idx_ok], dtype=np.int64)
    log.info(f"n={len(y)}  n_no_change={int((y == 0).sum())}  n_completed={int((y == 1).sum())}")

    # Per-mode t-SNE + save
    fig_all, axes = plt.subplots(1, len(args.modes),
                                 figsize=(5 * len(args.modes), 5))
    if len(args.modes) == 1:
        axes = [axes]
    for i, mode in enumerate(args.modes):
        log.info(f"===== t-SNE fuse={mode} =====")
        X = fuse(fa, fb, mode)
        # Normalise each row for stable t-SNE
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
        tsne = TSNE(n_components=2, perplexity=args.perplexity,
                    max_iter=args.n_iter, random_state=args.seed,
                    init="pca")
        emb = tsne.fit_transform(Xn)
        # Per-mode figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=6, alpha=0.5,
                   label="no_change", color="tab:blue")
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=6, alpha=0.5,
                   label="completed", color="tab:red")
        ax.set_title(f"t-SNE of frozen CLIP features ({mode})")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = out_dir / f"tsne_test_{mode}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Wrote {out}")
        # Combined panel
        axes[i].scatter(emb[y == 0, 0], emb[y == 0, 1], s=6, alpha=0.5,
                        color="tab:blue")
        axes[i].scatter(emb[y == 1, 0], emb[y == 1, 1], s=6, alpha=0.5,
                        color="tab:red")
        axes[i].set_title(mode)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    axes[0].legend(["no_change", "completed"], loc="upper left", frameon=True)
    fig_all.tight_layout()
    combo_out = out_dir / "tsne_combined.png"
    fig_all.savefig(combo_out, dpi=200, bbox_inches="tight")
    plt.close(fig_all)
    log.info(f"Wrote {combo_out}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
