"""Build caption-derived weak labels for all LEVIR-CC splits.

Run locally after the dataset is downloaded:
    python scripts/build_weak_labels.py --captions data/LEVIR_CC/LevirCCcaptions.json \
                                        --out outputs/labels --min-votes 3

Produces:
    outputs/labels/train_labels.json
    outputs/labels/val_labels.json
    outputs/labels/test_labels.json
    outputs/labels/label_coverage.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.captions import build_weak_labels
from geoconstruct.utils.logging import setup_logger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/LEVIR_CC/LevirCCcaptions.json")
    ap.add_argument("--out", default="outputs/labels")
    ap.add_argument("--min-votes", type=int, default=3)
    args = ap.parse_args()

    setup_logger("geoconstruct", log_file=Path(args.out) / "build_weak_labels.log")
    build_weak_labels(args.captions, args.out, min_votes=args.min_votes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
