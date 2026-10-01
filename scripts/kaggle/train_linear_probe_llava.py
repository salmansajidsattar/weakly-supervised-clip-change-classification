"""Train a logistic-regression linear probe on LLaVA vision-tower features.

Uses the features cached by extract_llava_features.py. This step is CPU-only
and fast; it separates GPU work (feature extraction, done once) from
CPU work (probe training and evaluation).

Reports macro-F1 for each fusion mode {f1, f2, diff, concat}, allowing a
direct row-by-row comparison to Table IV of the paper (classical baselines
on frozen CLIP features).

Output
------
outputs/runs/llava_linear_probe/
    metrics_test.json           macro-F1 per fusion mode
    predictions_test_concat.jsonl  per-pair predictions for concat mode
    summary.json                combined table
    log.txt

Runtime
-------
* CPU: ~2-5 minutes total for all fusion modes
* Disk: <10 MB output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")

log = logging.getLogger("train_linear_probe_llava")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CLASS_NAMES = ["no_change", "completed"]


def load_split(feats_dir, labels_dir, split):
    z = np.load(Path(feats_dir) / f"{split}_features.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    fa = z["feat_a"].astype(np.float32)
    fb = z["feat_b"].astype(np.float32)
    with open(Path(labels_dir) / f"{split}_labels.json") as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
    y = np.array([labels[i] for i in ids], dtype=np.int64)
    return ids, fa, fb, y


def fuse(fa, fb, mode):
    if mode == "f1":
        return fa
    if mode == "f2":
        return fb
    if mode == "diff":
        return fb - fa
    if mode == "concat":
        return np.concatenate([fa, fb], axis=-1)
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats-dir",  default=str(ROOT / "outputs/features_llava"))
    ap.add_argument("--labels-dir", default=str(ROOT / "outputs/labels"))
    ap.add_argument("--out",        default=str(ROOT / "outputs/runs/llava_linear_probe"))
    ap.add_argument("--fuse-modes", nargs="+", default=["f1", "f2", "diff", "concat"])
    ap.add_argument("--C",          type=float, default=1.0)
    ap.add_argument("--max-iter",   type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Loading LLaVA vision-tower features...")
    ids_tr, fa_tr, fb_tr, y_tr = load_split(args.feats_dir, args.labels_dir, "train")
    ids_te, fa_te, fb_te, y_te = load_split(args.feats_dir, args.labels_dir, "test")
    log.info(f"train n={len(y_tr)}  test n={len(y_te)}  dim={fa_tr.shape[1]}")

    results = {}
    for mode in args.fuse_modes:
        log.info(f"===== fuse={mode} =====")
        Xtr = fuse(fa_tr, fb_tr, mode)
        Xte = fuse(fa_te, fb_te, mode)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                                 solver="lbfgs", class_weight="balanced")
        clf.fit(Xtr_s, y_tr)
        y_pred = clf.predict(Xte_s)
        macro = f1_score(y_te, y_pred, average="macro")
        report = classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                       zero_division=0)
        log.info("\n" + report)
        log.info(f"macro-F1: {macro:.4f}")
        results[mode] = {
            "macro_f1": float(macro),
            "n_train": len(y_tr), "n_test": len(y_te),
            "feature_dim": int(Xtr.shape[1]),
            "backbone": "llava-1.5-7b vision tower (mean-pooled)",
        }
        if mode == "concat":
            with open(out_dir / "predictions_test_concat.jsonl", "w") as f:
                for pid, y, yp in zip(ids_te, y_te.tolist(), y_pred.tolist()):
                    f.write(json.dumps({"id": pid, "label": int(y),
                                        "pred": int(yp),
                                        "fuse": mode,
                                        "backbone": "llava_linear_probe"}) + "\n")

    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        summary = {mode: {"macro_f1": r["macro_f1"]} for mode, r in results.items()}
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("SUMMARY of LLaVA vision-tower linear probe")
    for mode, r in results.items():
        log.info(f"  {mode:>8s}: macro-F1 = {r['macro_f1']:.4f}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
