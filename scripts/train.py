"""Train a single configuration end-to-end from cached features.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.dataset import load_cached_split
from geoconstruct.losses.classification import build_loss
from geoconstruct.models.temporal_head import build_head
from geoconstruct.trainers.trainer import Trainer
from geoconstruct.utils.config import load_config
from geoconstruct.utils.logging import setup_logger
from geoconstruct.utils.seed import seed_everything

log = logging.getLogger("train")


def _build_scheduler(name: str, optimizer, n_epochs: int, patience: int):
    """Returns (scheduler, scheduler_type)."""
    name = (name or "none").lower()
    if name == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        return sch, "cosine"
    if name in ("plateau", "reduce_on_plateau"):
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=max(2, patience // 2), min_lr=1e-6,
        )
        return sch, "plateau"
    return None, "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.run.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("geoconstruct", log_file=out_dir / "train.log")
    setup_logger("train", log_file=out_dir / "train.log")
    log.info("Loaded config: %s", args.config)

    seed_everything(int(cfg.run.seed), deterministic=bool(cfg.run.deterministic))

    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    train_ds = load_cached_split(cfg.paths.features_dir, cfg.paths.labels_dir, "train")
    val_ds = load_cached_split(cfg.paths.features_dir, cfg.paths.labels_dir, "val")
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.optim.batch_size), shuffle=True,
        num_workers=int(cfg.optim.get("num_workers", 0)), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.optim.batch_size), shuffle=False,
        num_workers=int(cfg.optim.get("num_workers", 0)),
    )

    model = build_head(cfg)
    loss_fn = build_loss(
        cfg,
        train_labels_json=Path(cfg.paths.labels_dir) / "train_labels.json",
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )

    n_epochs = int(cfg.optim.n_epochs)
    patience = int(cfg.optim.get("early_stopping_patience", 0))
    scheduler, sch_type = _build_scheduler(
        cfg.optim.get("scheduler", "none"),
        optimizer, n_epochs, patience,
    )
    log.info("Scheduler: %s | early_stopping_patience: %d", sch_type, patience)

    trainer = Trainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        device=cfg.run.device, out_dir=out_dir,
        num_classes=int(cfg.head.num_classes),
        scheduler=scheduler, scheduler_type=sch_type,
        early_stopping_patience=patience,
    )
    trainer.fit(n_epochs=n_epochs)
    log.info("Training done. Best macro-F1=%.4f (epoch %d)",
             trainer.best_f1, trainer.best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
