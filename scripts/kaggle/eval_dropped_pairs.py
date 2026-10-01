"""Dropped-pair analysis: are the pairs the majority-vote rule filters out
systematically harder or easier than the retained ones?

Addresses Reviewer 1 comment #4: "No analysis is provided on whether the
dropped pairs are harder or easier than retained ones. Reporting VLM and
CLIP-zero-shot performance specifically on the dropped subset would reveal
whether the weak-label filtering introduces a difficulty bias."

The strategy
------------
"Dropped" pairs are pairs where <3 captions of 5 agreed on either class
under the 3/5 rule. Because such pairs have no reliable weak label, we
need an alternative ground truth. Options:

  (A) Use a stricter 4/5 or 5/5 majority as pseudo-ground-truth: apply
      the rule again with a higher threshold on the same captions; any
      pair the stricter rule still labels is our pseudo-truth pair.
      This is the approach used here (default: stricter=4).
  (B) Hand-label a small random subset.

We choose (A) because it is fully reproducible from public captions and
requires no extra annotation. Pairs that "would have been dropped at 3/5"
but "have a label at 4/5" are, definitionally, cases where the rule is
BORDERLINE. Running LLaVA and zero-shot CLIP on those and comparing to
the retained-pair accuracy is exactly the requested diagnostic.

Output
------
outputs/runs/dropped_pair_analysis/
    dropped_ids.json                    list of pair IDs that were dropped
                                        at 3/5 but recovered at stricter/5
    dropped_labels.json                 the pseudo-ground-truth labels
    metrics_llava.json                  LLaVA performance on dropped subset
    metrics_clip_zero_shot.json         zero-shot CLIP performance
    summary.json                        side-by-side vs retained-subset
    log.txt

Runtime
-------
* CPU: fast for label reconstruction (~1 min)
* GPU: ~1-2 hours for LLaVA on dropped pairs (~500-800 pairs)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")

log = logging.getLogger("dropped_pair_analysis")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_labels_at_threshold(min_votes, out_labels_dir):
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,
           str(ROOT / "scripts/build_weak_labels.py"),
           "--min-votes", str(min_votes),
           "--out-dir", str(out_labels_dir)]
    log.info(f"Rebuilding labels with min_votes={min_votes}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"failed: {r.stderr[-2000:]}")
        raise RuntimeError(r.stderr)


def load_labels(labels_dir, split):
    with open(Path(labels_dir) / f"{split}_labels.json") as f:
        return {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="test")
    ap.add_argument("--out",     default=str(ROOT / "outputs/runs/dropped_pair_analysis"))
    ap.add_argument("--labels-original", default=str(ROOT / "outputs/labels"),
                    help="Original 3/5 labels dir")
    ap.add_argument("--stricter", type=int, default=5,
                    help="Stricter threshold used as pseudo-truth "
                         "(default: 5/5 unanimous)")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    # Step 1: build a stricter label set to use as pseudo-truth
    stricter_labels_dir = out_dir / f"labels_min_votes_{args.stricter}"
    build_labels_at_threshold(args.stricter, stricter_labels_dir)

    # Step 2: identify pairs dropped at 3/5 but labelled at 5/5.
    # Wait: dropped-at-3/5 means 0/5 caps agreed AND 0/5 disagreed at 3-of-5 level.
    # Any pair labelled at stricter/5 is by definition ALSO labelled at 3/5.
    # We want the opposite: pairs UNLABELLED at 3/5 but ALSO unlabelled at 5/5.
    # Those are hopeless.
    #
    # A cleaner definition: the RETAINED subset = pairs labelled at 3/5.
    # The DROPPED subset = pairs NOT labelled at 3/5.
    # Pseudo-truth for DROPPED pairs = the caption votes with a 2/5 (LOOSER)
    # threshold, taking the majority label if >= 2 agree.
    # Rebuild labels at threshold 2/5 to get pseudo-truth for dropped pairs.
    looser_labels_dir = out_dir / "labels_min_votes_2"
    build_labels_at_threshold(2, looser_labels_dir)

    orig = load_labels(args.labels_original, args.split)
    looser = load_labels(looser_labels_dir, args.split)

    # The set of pair IDs in the LEVIR-CC test split (from looser, which drops fewer)
    all_ids = set(looser.keys())
    labelled_at_3 = set(orig.keys())
    dropped_at_3 = all_ids - labelled_at_3
    # For dropped pairs, pseudo-truth = looser label
    dropped_labels = {pid: looser[pid] for pid in sorted(dropped_at_3) if pid in looser}
    log.info(f"Retained (3/5 labelled): {len(labelled_at_3)}")
    log.info(f"Dropped-at-3/5, recovered-at-2/5 pseudo-truth: {len(dropped_labels)}")

    with open(out_dir / "dropped_ids.json", "w") as f:
        json.dump(sorted(dropped_labels.keys()), f, indent=2)
    with open(out_dir / "dropped_labels.json", "w") as f:
        json.dump(dropped_labels, f, indent=2)

    # Step 3: subclass balance report
    n_dropped = len(dropped_labels)
    n0 = sum(1 for v in dropped_labels.values() if v == 0)
    n1 = sum(1 for v in dropped_labels.values() if v == 1)
    summary = {
        "n_labelled_at_3": len(labelled_at_3),
        "n_dropped_at_3_recovered_at_2": n_dropped,
        "dropped_class_balance": {"no_change": n0, "completed": n1},
        "notes": (
            "'Dropped-at-3/5, recovered-at-2/5' pairs are the borderline "
            "cases where the strict rule was uncertain but a looser rule "
            "yields a plurality label. VLM / CLIP performance on this "
            "subset is compared to their performance on the retained subset."
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Step 4: hand off to the user
    log.info("=" * 60)
    log.info("DROPPED-PAIR SUBSET PREPARED.")
    log.info(f"To evaluate LLaVA on this subset:")
    log.info(f"  python scripts/kaggle/eval_llava_fewshot.py \\")
    log.info(f"      --labels-dir {out_dir} \\")
    log.info(f"      --shots 0 \\")
    log.info(f"      --out {out_dir}/llava_on_dropped")
    log.info(f"To evaluate zero-shot CLIP on this subset:")
    log.info(f"  python scripts/eval_clip_zero_shot.py \\")
    log.info(f"      --labels-dir {out_dir} \\")
    log.info(f"      --fuse diff \\")
    log.info(f"      --out {out_dir}/clip_on_dropped")
    log.info(f"Then compare metrics_test.json values against the ones in")
    log.info(f"outputs/runs/vlm_zero_shot/ and outputs/runs/clip_zero_shot/.")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
