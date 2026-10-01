"""Evaluate a trained checkpoint on val or test split.

Usage:
    python scripts/eval.py --config configs/default.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.captions import CLASS_NAMES
from geoconstruct.data.dataset import load_cached_split
from geoconstruct.evaluation.metrics import compute_metrics, format_report
from geoconstruct.models.temporal_head import build_head
from geoconstruct.utils.config import load_config
from geoconstruct.utils.logging import setup_logger
from geoconstruct.utils.seed import seed_everything

log = logging.getLogger("eval")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--checkpoint", default=None,
                    help="Override; default = <out_dir>/best.pt")
    ap.add_argument("--save-fused", action="store_true",
                    help="Also dump fused features for t-SNE later")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.run.out_dir)
    setup_logger("geoconstruct", log_file=out_dir / "eval.log")
    setup_logger("eval", log_file=out_dir / "eval.log")

    seed_everything(int(cfg.run.seed), deterministic=True)

    ds = load_cached_split(cfg.paths.features_dir, cfg.paths.labels_dir, args.split)
    loader = DataLoader(ds, batch_size=int(cfg.optim.batch_size), shuffle=False)
    log.info("Eval split=%s size=%d", args.split, len(ds))

    model = build_head(cfg)
    ckpt = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"
    log.info("Loading checkpoint: %s", ckpt)
    state = torch.load(ckpt, map_location=cfg.run.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(cfg.run.device)

    all_pred, all_true, fused_all, ids_all = [], [], [], []
    with torch.no_grad():
        for f1, f2, y, ids in loader:
            f1 = f1.to(cfg.run.device); f2 = f2.to(cfg.run.device)
            logits, fused = model(f1, f2)
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(y.tolist())
            if args.save_fused:
                fused_all.append(fused.cpu().numpy())
                ids_all.extend(list(ids))

    metrics = compute_metrics(all_true, all_pred, num_classes=int(cfg.head.num_classes))
    report = format_report(all_true, all_pred, target_names=CLASS_NAMES)
    log.info("\n%s", report)

    out_metrics = out_dir / f"metrics_{args.split}.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", out_metrics)

    if args.save_fused and fused_all:
        fused = np.concatenate(fused_all, axis=0)
        np.savez(out_dir / f"fused_{args.split}.npz",
                 fused=fused, labels=np.array(all_true), pred=np.array(all_pred),
                 ids=np.array(ids_all, dtype=object))
        log.info("Saved fused features for %s", args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
