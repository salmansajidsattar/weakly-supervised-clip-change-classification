"""Generate paper figures from saved metrics and fused features.

Usage:
    python scripts/make_figures.py --run-dir outputs/runs/full_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.captions import CLASS_NAMES
from geoconstruct.visualization.plots import (
    plot_confusion_matrix,
    plot_history,
    plot_tsne,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = run_dir / "history.json"
    if history.exists():
        plot_history(history, out_dir / "loss_acc.png")

    test_metrics = run_dir / "metrics_test.json"
    if test_metrics.exists():
        with open(test_metrics, "r", encoding="utf-8") as f:
            m = json.load(f)
        plot_confusion_matrix(m["confusion_matrix"], CLASS_NAMES,
                              out_dir / "confusion_matrix.png")

    fused_npz = run_dir / "fused_test.npz"
    if fused_npz.exists():
        data = np.load(fused_npz, allow_pickle=True)
        plot_tsne(data["fused"], data["labels"], CLASS_NAMES,
                  out_dir / "tsne.png")

    print(f"Figures written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
