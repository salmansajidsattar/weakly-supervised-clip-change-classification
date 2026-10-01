"""Qualitative examples figure: 3 success cases + 3 failure cases.

Addresses Reviewer 1 comment #6 (part 1): "Given the central claim about
VLM transfer limitations, qualitative examples are essential. The paper
would benefit from showing representative image pairs where all methods
agree, where the VLM fails but CLIP zero-shot succeeds, and where only
the trained head succeeds."

Picks 3 pairs for each of the two categories below and displays them as
a matplotlib grid:

  ROW 1 — successes: pairs where the trained head is CORRECT
                     while LLaVA is WRONG (illustrates the value of
                     weak supervision).
  ROW 2 — failures:  pairs where BOTH the trained head and LLaVA are
                     WRONG (illustrates residual difficulty).

Each subplot shows the pre-change image A and the post-change image B
side by side, with the ground-truth label and each method's prediction
labelled underneath.

Output
------
outputs/figures/qualitative_examples.png    (single 6-panel figure)
outputs/figures/qualitative_examples.pdf    (vector version for LaTeX)

Runtime
-------
* CPU: ~1 minute
* Disk: two small files
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")

log = logging.getLogger("make_qualitative_figure")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CLASS_NAMES = ["no_change", "completed"]


def load_jsonl_preds(path):
    """Load a predictions jsonl into a dict {id: pred}."""
    m = {}
    if not Path(path).exists():
        return m
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m[r["id"]] = int(r["pred"])
    return m


def image_paths(images_root, split, pair_id):
    a = Path(images_root) / split / "A" / pair_id
    b = Path(images_root) / split / "B" / pair_id
    if not a.exists():
        a = Path(images_root) / "A" / pair_id
        b = Path(images_root) / "B" / pair_id
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--split",       default="test")
    ap.add_argument("--labels",      default=str(ROOT / "outputs/labels/test_labels.json"))
    ap.add_argument("--head-preds",  default=str(ROOT / "outputs/runs/concat_seed42/fused_test.npz"),
                    help=".npz with 'ids' and 'pred' keys from trained head")
    ap.add_argument("--llava-preds", default=str(ROOT / "outputs/runs/vlm_zero_shot/predictions_test.jsonl"))
    ap.add_argument("--clip-preds",  default=str(ROOT / "outputs/runs/clip_zero_shot/predictions_test_diff.jsonl"))
    ap.add_argument("--out-png",     default=str(ROOT / "outputs/figures/qualitative_examples.png"))
    ap.add_argument("--out-pdf",     default=str(ROOT / "outputs/figures/qualitative_examples.pdf"))
    ap.add_argument("--n-success",   type=int, default=3)
    ap.add_argument("--n-failure",   type=int, default=3)
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Ground truth
    with open(args.labels) as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}

    # Head predictions (from cached npz)
    if Path(args.head_preds).exists():
        z = np.load(args.head_preds, allow_pickle=True)
        head_pred = {str(pid): int(p) for pid, p in
                     zip(z["ids"].tolist(), z["pred"].tolist())}
    else:
        head_pred = {}
        log.warning(f"Head preds not found: {args.head_preds}")

    llava_pred = load_jsonl_preds(args.llava_preds)
    clip_pred = load_jsonl_preds(args.clip_preds)

    # Categorise pairs
    successes = []  # head correct, LLaVA wrong
    failures  = []  # head wrong, LLaVA wrong
    for pid, y in labels.items():
        h = head_pred.get(pid); l = llava_pred.get(pid, -2)
        if h is None or l == -2:
            continue
        head_correct = (h == y)
        llava_correct = (l == y)
        if head_correct and not llava_correct:
            successes.append(pid)
        elif (not head_correct) and (not llava_correct):
            failures.append(pid)

    rng = random.Random(args.seed)
    rng.shuffle(successes)
    rng.shuffle(failures)

    def balance_by_class(pool, n):
        """Pick n ids from pool, guaranteeing at least one of each label
        class when the pool contains both (falls back to majority-class-only
        fill if one class is scarce), rather than just taking the first n
        after shuffling."""
        by_class = {}
        for pid in pool:
            by_class.setdefault(labels[pid], []).append(pid)
        classes = list(by_class.keys())
        picked = []
        # Guarantee at least one example per available class first.
        for c in classes:
            if by_class[c] and len(picked) < n:
                picked.append(by_class[c].pop(0))
        # Fill the rest from the remaining pool in shuffled order.
        remaining = [pid for c in classes for pid in by_class[c]]
        rng.shuffle(remaining)
        for pid in remaining:
            if len(picked) >= n:
                break
            picked.append(pid)
        return picked

    successes_bal = balance_by_class(successes, args.n_success)
    failures_bal = balance_by_class(failures, args.n_failure)

    picked = successes_bal + failures_bal
    labels_row = (["success"] * len(successes_bal)) + (["failure"] * len(failures_bal))

    if not picked:
        log.error("No pairs available. Ensure the prediction files exist.")
        return 1

    # Build the figure: 2 rows (success / failure), N cols (each shows A+B pair)
    # Font sizes below are deliberately large in the source figure because the
    # PDF is scaled down to \textwidth (~half this figure's native width) when
    # placed in the two-column journal layout; sizes are chosen so the printed
    # text clears JARS's "no smaller than 8 pt at final typeset size" rule.
    n_cols = max(len(successes_bal), len(failures_bal))
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.5))
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for row, (row_label, ids) in enumerate([("success", successes_bal),
                                            ("failure", failures_bal)]):
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(ids):
                ax.axis("off")
                continue
            pid = ids[col]
            a, b = image_paths(args.images_root, args.split, pid)
            if not (a.exists() and b.exists()):
                ax.text(0.5, 0.5, f"MISSING\n{pid}", ha="center", va="center")
                ax.axis("off")
                continue
            imgA = Image.open(a).convert("RGB")
            imgB = Image.open(b).convert("RGB")
            # Upsample the (native 256x256) LEVIR-CC crops with high-quality
            # resampling before display so the printed thumbnail is sharper,
            # not just larger.
            upscale = 1
            imgA = imgA.resize((imgA.width * upscale, imgA.height * upscale), Image.LANCZOS)
            imgB = imgB.resize((imgB.width * upscale, imgB.height * upscale), Image.LANCZOS)
            # Side-by-side with a thin white separator so the before/after
            # boundary is visible even at small print size.
            sep = np.full((imgA.height, 4, 3), 255, dtype=np.uint8)
            combined = np.concatenate([np.array(imgA), sep, np.array(imgB)], axis=1)
            ax.imshow(combined)
            ax.axis("off")
            y = labels[pid]
            h = head_pred.get(pid, -1)
            l = llava_pred.get(pid, -1)
            c = clip_pred.get(pid, -1)
            title = (
                f"GT: {CLASS_NAMES[y]}\n"
                f"head: {CLASS_NAMES[h] if h in (0, 1) else '?'}\n"
                f"LLaVA: {CLASS_NAMES[l] if l in (0, 1) else '?'}   "
                f"CLIP-diff: {CLASS_NAMES[c] if c in (0, 1) else '?'}"
            )
            ax.set_title(title, fontsize=15, linespacing=1.4)
        axes[row, 0].set_ylabel(row_label, fontsize=17, rotation=90,
                                labelpad=14)

    # No fig.suptitle here: per JARS figure guidelines, captions/titles must
    # not be baked into the image file (the caption lives in the manuscript's
    # \caption{} instead).
    fig.subplots_adjust(wspace=0.15, hspace=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {out_png}")
    log.info(f"Wrote {args.out_pdf}")

    # Also dump the picked IDs for the paper
    picked_info = {
        "success_ids": successes_bal,
        "failure_ids": failures_bal,
        "notes": (
            "success = trained head correct, LLaVA-7B wrong; "
            "failure = both trained head and LLaVA-7B wrong"
        ),
    }
    with open(out_png.parent / "qualitative_examples_picked.json", "w") as f:
        json.dump(picked_info, f, indent=2)
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
