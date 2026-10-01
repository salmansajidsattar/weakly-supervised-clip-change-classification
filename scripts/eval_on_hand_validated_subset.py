"""Re-score all three methods against the 183 hand-validated pairs (JARS R1#3).

Reviewer 1, comment #3 (major revision, JARS):
    "The authors should directly evaluate LLaVA, zero-shot CLIP, and the
    trained CLIP head on these 183 manually labeled samples. This would show
    whether the reported macro-F1 of 0.882 also reflects performance under
    human-defined labels. Confidence intervals should also be reported. This
    experiment requires no additional annotation and could significantly
    strengthen the paper."

This script requires NO new inference and NO GPU. All three methods were
already run on the full test split; this just intersects their cached
per-pair predictions with the 183 pairs you hand-labelled in
outputs/label_validation/manual_test.jsonl, and rescores against the
`manual` column instead of the weak-label column.

Reads
-----
  outputs/label_validation/manual_test.jsonl
      {"id", "weak", "manual"} per pair; manual == -1 means skipped/ambiguous
      and is excluded here (this is the same n=183 filter used to report
      kappa=0.596 in the paper).
  outputs/runs/vlm_zero_shot/predictions_test.jsonl        (LLaVA-7B)
  outputs/runs/clip_zero_shot/predictions_test_diff.jsonl  (zero-shot CLIP, z2-z1)
  outputs/runs/concat_seed42/fused_test.npz                (trained concat head)

Writes
------
  outputs/stats/hand_validated_subset.json
  outputs/stats/hand_validated_subset_latex.tex

Usage
-----
    python scripts/eval_on_hand_validated_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "stats"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


def f1_macro(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for k in (0, 1):
        tp = int(np.sum((y_pred == k) & (y_true == k)))
        fp = int(np.sum((y_pred == k) & (y_true != k)))
        fn = int(np.sum((y_pred != k) & (y_true == k)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def bootstrap_ci(y_true, y_pred, n=1000):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n_pairs = len(y_true)
    if n_pairs == 0:
        return float("nan"), float("nan"), float("nan")
    scores = np.empty(n)
    for k in range(n):
        idx = RNG.integers(0, n_pairs, n_pairs)
        scores[k] = f1_macro(y_true[idx], y_pred[idx])
    point = f1_macro(y_true, y_pred)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def load_manual_subset(path):
    ids, manual = [], {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if int(r["manual"]) < 0:
                continue  # skipped / ambiguous during hand-validation
            manual[r["id"]] = int(r["manual"])
    return manual


def load_jsonl_preds(path):
    pred_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred_map[r["id"]] = int(r["pred"])
    return pred_map


def load_head_preds():
    z = np.load(ROOT / "outputs/runs/concat_seed42/fused_test.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    pred = np.asarray(z["pred"], dtype=np.int64)
    return dict(zip(ids, pred.tolist()))


def main() -> int:
    manual_path = ROOT / "outputs/label_validation/manual_test.jsonl"
    manual = load_manual_subset(manual_path)
    subset_ids = sorted(manual.keys())
    print(f"Hand-validated subset: {len(subset_ids)} pairs (manual label available)")
    y_manual = np.array([manual[i] for i in subset_ids], dtype=np.int64)
    print(f"  class balance (manual): {np.bincount(y_manual, minlength=2).tolist()}")

    llava_map = load_jsonl_preds(ROOT / "outputs/runs/vlm_zero_shot/predictions_test.jsonl")
    clip_map = load_jsonl_preds(ROOT / "outputs/runs/clip_zero_shot/predictions_test_diff.jsonl")
    head_map = load_head_preds()

    methods = {
        "LLaVA-7B zero-shot": llava_map,
        "Zero-shot CLIP (z2-z1)": clip_map,
        "Trained concat head": head_map,
    }

    rows = []
    for name, pred_map in methods.items():
        ids_avail = [i for i in subset_ids if i in pred_map and pred_map[i] >= 0]
        n_missing = len(subset_ids) - len(ids_avail)
        y_t = np.array([manual[i] for i in ids_avail], dtype=np.int64)
        y_p = np.array([pred_map[i] for i in ids_avail], dtype=np.int64)
        pt, lo, hi = bootstrap_ci(y_t, y_p, n=1000)
        acc = float(np.mean(y_t == y_p)) if len(y_t) else float("nan")
        print(f"  {name:<26s} n={len(ids_avail):3d} (missing/unparseable={n_missing:2d})  "
              f"macro-F1={pt:.3f} [{lo:.3f}, {hi:.3f}]  acc={acc:.3f}")
        rows.append({
            "method": name, "n_scored": len(ids_avail), "n_missing": n_missing,
            "macro_f1": pt, "ci_lo": lo, "ci_hi": hi, "accuracy": acc,
        })

    with open(OUT / "hand_validated_subset.json", "w", encoding="utf-8") as f:
        json.dump({"n_subset": len(subset_ids), "rows": rows}, f, indent=2)

    snip = [
        "% Auto-generated by scripts/eval_on_hand_validated_subset.py",
        f"% Re-scored against {len(subset_ids)} hand-validated (human-judged) labels, "
        "not weak labels.",
        "% Method & Macro-F1 [95% CI] & Acc & n \\\\",
        "% \\midrule",
    ]
    for r in rows:
        snip.append(
            f"% {r['method']:<26s} & {r['macro_f1']:.3f} "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] & {r['accuracy']:.3f} & "
            f"{r['n_scored']} \\\\"
        )
    with open(OUT / "hand_validated_subset_latex.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(snip) + "\n")
    print("\nWrote outputs/stats/hand_validated_subset.json and _latex.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
