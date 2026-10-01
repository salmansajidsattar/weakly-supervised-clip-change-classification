"""Classical ML baselines on cached CLIP features.

Trains LogisticRegression, LinearSVC, and k-NN on the same train/val/test
splits and the same weak labels as the headline MLP head. Reports macro-F1
for four fusion modes: f1-only, f2-only, f2-f1, concat[f1;f2].

Output:
    outputs/classical_baselines.json     machine-readable
    outputs/classical_baselines.log      human-readable progress log
    A LaTeX-ready table printed to stdout.

Usage:
    python scripts/run_classical_baselines.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from geoconstruct.utils.logging import setup_logger

log = setup_logger("classical_baselines",
                   log_file=PROJECT / "outputs" / "classical_baselines.log")


def _load_split(features_dir: Path, labels_dir: Path, split: str):
    cache = np.load(features_dir / f"{split}_features.npz", allow_pickle=True)
    fa, fb, ids = cache["feat_a"], cache["feat_b"], cache["ids"]
    ids = [str(x) for x in ids.tolist()]
    with open(labels_dir / f"{split}_labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    id_to_idx = {i: k for k, i in enumerate(ids)}
    keep_idx, y = [], []
    for fname, lab in labels.items():
        lab = int(lab)
        if lab < 0:
            continue
        if fname not in id_to_idx:
            continue
        keep_idx.append(id_to_idx[fname])
        y.append(lab)
    keep_idx = np.asarray(keep_idx)
    return fa[keep_idx], fb[keep_idx], np.asarray(y)


def _fuse(fa, fb, mode):
    if mode == "f1":   return fa
    if mode == "f2":   return fb
    if mode == "diff": return fb - fa
    if mode == "concat": return np.concatenate([fa, fb], axis=-1)
    raise ValueError(mode)


def _eval_model(model, Xtr, ytr, Xte, yte):
    from sklearn.metrics import f1_score
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    return float(f1_score(yte, yp, average="macro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/features")
    ap.add_argument("--labels-dir", default="outputs/labels")
    ap.add_argument("--out", default="outputs/classical_baselines.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    fa_tr, fb_tr, ytr = _load_split(Path(args.features_dir),
                                     Path(args.labels_dir), "train")
    fa_te, fb_te, yte = _load_split(Path(args.features_dir),
                                     Path(args.labels_dir), "test")
    log.info("Loaded: train=%d, test=%d", len(ytr), len(yte))

    modes = ["f1", "f2", "diff", "concat"]
    results: dict = {}
    for mode in modes:
        Xtr = _fuse(fa_tr, fb_tr, mode)
        Xte = _fuse(fa_te, fb_te, mode)
        log.info("Mode=%s | shape train=%s test=%s", mode, Xtr.shape, Xte.shape)

        clfs = {
            "LogReg":     make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             C=1.0, random_state=args.seed)),
            "LinearSVC":  make_pipeline(StandardScaler(),
                          LinearSVC(C=1.0, class_weight="balanced",
                                    random_state=args.seed, max_iter=5000)),
            "kNN":        make_pipeline(StandardScaler(),
                          KNeighborsClassifier(n_neighbors=15)),
        }
        results[mode] = {}
        for name, clf in clfs.items():
            f1 = _eval_model(clf, Xtr, ytr, Xte, yte)
            log.info("  %-10s | macro-F1=%.4f", name, f1)
            results[mode][name] = f1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote %s", args.out)

    # LaTeX-style table for direct paste into the paper.
    print("\n% Paste this into the Table IV stub in grsl_letter.tex")
    print(r"% \begin{tabular}{lrrr}")
    print(r"% \toprule")
    print(r"% Fusion mode & LogReg & Linear SVM & $k$-NN \\")
    print(r"% \midrule")
    nicelabel = {"f1": "$f_1$", "f2": "$f_2$",
                 "diff": "$f_2 - f_1$", "concat": "$[f_1; f_2]$"}
    for mode in modes:
        r = results[mode]
        print(f"% {nicelabel[mode]:<12} & {r['LogReg']:.3f} & "
              f"{r['LinearSVC']:.3f} & {r['kNN']:.3f} \\\\")
    print(r"% \bottomrule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
