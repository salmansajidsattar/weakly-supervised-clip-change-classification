"""Statistical hardening for the GRSL letter (pure numpy/sklearn).

Computes:
  * bootstrap 95% CIs (1000 resamples) on macro-F1 for the headline rows
    of Table I (random, LLaVA-7B, zero-shot CLIP f2-f1, trained concat head)
  * McNemar (continuity-corrected chi-square + exact binomial p):
        - trained head vs LLaVA-7B
        - trained head vs zero-shot CLIP (diff)
  * worst-case LLaVA macro-F1 if the 141 unparseable responses are counted
    as errors (predicted = opposite of truth).

Reads only cached prediction artefacts; no models are loaded.

REQUIRED INPUTS (re-run these once on the user machine if missing):

  outputs/runs/concat_seed42/fused_test.npz
      -> trained-head test predictions (already exists; produced by
         scripts/run_ablations.py)
  outputs/runs/vlm_zero_shot/predictions_test.jsonl
      -> LLaVA-7B test predictions (already exists; produced by
         scripts/eval_vlm_zero_shot.py)
  outputs/runs/clip_zero_shot/predictions_test_diff.jsonl
      -> zero-shot CLIP (f2 - f1) test predictions; produced by
         `python scripts/eval_clip_zero_shot.py --fuse diff`
         after the small patch added to that script that writes a
         per-pair jsonl alongside metrics_test.json.

Outputs:
  outputs/stats/headline_ci.json
  outputs/stats/mcnemar.json
  outputs/stats/llava_worstcase.json
  outputs/stats/latex_snippet.tex
"""

from __future__ import annotations

import json
import sys
from math import comb, erfc, sqrt
from pathlib import Path

import numpy as np


def f1_score(y_true, y_pred, average="macro", zero_division=0):
    """Pure-numpy binary macro-F1. Equivalent to sklearn for K=2 and the
    parameters we use here (average='macro', zero_division=0)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for k in (0, 1):
        tp = int(np.sum((y_pred == k) & (y_true == k)))
        fp = int(np.sum((y_pred == k) & (y_true != k)))
        fn = int(np.sum((y_pred != k) & (y_true == k)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if average == "macro" else f1s

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "stats"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Loaders.
# --------------------------------------------------------------------------- #
def load_head_predictions():
    z = np.load(ROOT / "outputs/runs/concat_seed42/fused_test.npz",
                allow_pickle=True)
    ids = [str(x) for x in z["ids"].tolist()]
    return ids, np.asarray(z["labels"], dtype=np.int64), \
                np.asarray(z["pred"],   dtype=np.int64)


def load_jsonl_preds(path, n_expected):
    pred_map, label_map = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred_map[r["id"]]  = int(r["pred"])
            label_map[r["id"]] = int(r["label"])
    return pred_map, label_map


# --------------------------------------------------------------------------- #
# Stats helpers.
# --------------------------------------------------------------------------- #
def bootstrap_macro_f1(y_true, y_pred, n=1000, mask=None):
    if mask is not None:
        y_true = y_true[mask]; y_pred = y_pred[mask]
    n_pairs = len(y_true)
    if n_pairs == 0:
        return float("nan"), float("nan"), float("nan")
    scores = np.empty(n)
    for k in range(n):
        idx = RNG.integers(0, n_pairs, n_pairs)
        scores[k] = f1_score(y_true[idx], y_pred[idx],
                             average="macro", zero_division=0)
    point = f1_score(y_true, y_pred, average="macro", zero_division=0)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar(y_true, y_a, y_b):
    a_right = (y_a == y_true)
    b_right = (y_b == y_true)
    n01 = int(np.sum(a_right & ~b_right))   # A right, B wrong
    n10 = int(np.sum(~a_right & b_right))   # A wrong, B right
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0,
                "p_chi2": 1.0, "p_exact": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n_disc
    k = min(n01, n10)
    pmf = sum(comb(n_disc, i) for i in range(k + 1))
    p_exact = min(1.0, pmf / (2 ** n_disc) * 2)
    p_chi2 = erfc(sqrt(chi2 / 2))   # 1 - CDF at chi2, df=1
    return {"n01": n01, "n10": n10, "chi2": float(chi2),
            "p_chi2": float(p_chi2), "p_exact": float(p_exact)}


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    print("Loading aligned test pairs from concat_seed42 cache...")
    ids_head, y, y_head = load_head_predictions()
    n = len(y)
    print(f"  {n} pairs, class balance: {np.bincount(y).tolist()}")

    print("Loading LLaVA-7B predictions (jsonl)...")
    llava_pred_map, _ = load_jsonl_preds(
        ROOT / "outputs/runs/vlm_zero_shot/predictions_test.jsonl",
        n_expected=n)
    y_llava = np.array([llava_pred_map.get(i, -1) for i in ids_head],
                       dtype=np.int64)
    llava_mask = y_llava >= 0
    print(f"  parseable: {int(llava_mask.sum())} of {n}")

    print("Loading zero-shot CLIP (diff) predictions (jsonl)...")
    clip_path = ROOT / "outputs/runs/clip_zero_shot/predictions_test_diff.jsonl"
    if clip_path.exists():
        clip_pred_map, _ = load_jsonl_preds(clip_path, n_expected=n)
        y_clip = np.array([clip_pred_map.get(i, -1) for i in ids_head],
                          dtype=np.int64)
        have_clip = True
    else:
        print(f"  NOTE: {clip_path} not found; CLIP-diff CI/McNemar will be skipped.")
        print("  To populate, run: python scripts/eval_clip_zero_shot.py --fuse diff")
        y_clip = None; have_clip = False

    print("Synthesising random baseline (uniform 50/50)...")
    y_rand = RNG.integers(0, 2, n)

    # ---------------------------------------------------------------- CIs
    print("\nBootstrap 95% CIs (n=1000 resamples):")
    rows = []
    triples = [
        ("Random",                  y_rand,  None),
        ("LLaVA-7B zero-shot",      y_llava, llava_mask),
    ]
    if have_clip:
        triples.append(("Zero-shot CLIP (f2-f1)", y_clip, None))
    triples.append(("Trained concat head", y_head, None))
    for name, yp, m in triples:
        pt, lo, hi = bootstrap_macro_f1(y, yp, n=1000, mask=m)
        print(f"  {name:<28s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append({"method": name, "macro_f1": pt,
                     "ci_lo": lo, "ci_hi": hi})
    with open(OUT / "headline_ci.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ---------------------------------------------------------------- McNemar
    print("\nMcNemar tests:")
    mc = {}
    res = mcnemar(y[llava_mask], y_head[llava_mask], y_llava[llava_mask])
    mc["head_vs_llava"] = res
    print(f"  head vs LLaVA   n01={res['n01']:4d} n10={res['n10']:4d}  "
          f"chi2={res['chi2']:.1f}  p_exact={res['p_exact']:.2e}")
    if have_clip:
        res = mcnemar(y, y_head, y_clip)
        mc["head_vs_clip_diff"] = res
        print(f"  head vs CLIP    n01={res['n01']:4d} n10={res['n10']:4d}  "
              f"chi2={res['chi2']:.1f}  p_exact={res['p_exact']:.2e}")
    with open(OUT / "mcnemar.json", "w") as f:
        json.dump(mc, f, indent=2)

    # ---------------------------------------------------------------- worst-case LLaVA
    print("\nWorst-case LLaVA (unparseable = opposite of truth):")
    y_llava_worst = y_llava.copy()
    y_llava_worst[~llava_mask] = 1 - y[~llava_mask]
    pt, lo, hi = bootstrap_macro_f1(y, y_llava_worst, n=1000)
    print(f"  worst-case macro-F1 = {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    with open(OUT / "llava_worstcase.json", "w") as f:
        json.dump({"macro_f1": pt, "ci_lo": lo, "ci_hi": hi}, f, indent=2)

    # ---------------------------------------------------------------- LaTeX
    snip = ["% Auto-generated by scripts/run_stats.py",
            "% Bootstrap 95% CIs (n=1000):"]
    for r in rows:
        snip.append(f"%   {r['method']:<28s} {r['macro_f1']:.3f} "
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    snip.append("% McNemar:")
    for k, v in mc.items():
        snip.append(f"%   {k}: chi2={v['chi2']:.1f}, "
                    f"p_exact={v['p_exact']:.2e}, "
                    f"n01={v['n01']}, n10={v['n10']}")
    snip.append("")
    snip.append("% Table I row text (drop into the existing table):")
    for r in rows:
        if r["method"] == "Random":
            continue
        snip.append(
            f"%   {r['method']:<28s} & "
            f"{r['macro_f1']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] \\\\"
        )
    with open(OUT / "latex_snippet.tex", "w") as f:
        f.write("\n".join(snip) + "\n")
    print(f"\nWrote outputs/stats/{{headline_ci.json,mcnemar.json,"
          f"llava_worstcase.json,latex_snippet.tex}}")


if __name__ == "__main__":
    main()
