"""CPU-friendly trainer for the cached-feature setup.

Supports:
  - Standard step-based schedulers (cosine) called via `scheduler.step()` after each epoch
  - ReduceLROnPlateau called with the val metric after each epoch
  - Early stopping on val macro-F1 with configurable patience

Best checkpoint (by val macro-F1) is saved as <out_dir>/best.pt.
Per-epoch history is saved as <out_dir>/history.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_metrics
from ..utils.logging import setup_logger

log = logging.getLogger(__name__)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    lr: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        out_dir: str | Path = "outputs/run",
        num_classes: int = 2,
        scheduler=None,
        scheduler_type: str = "none",       # "none" | "cosine" | "plateau"
        early_stopping_patience: int = 0,    # 0 disables early stopping
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device) if hasattr(loss_fn, "to") else loss_fn
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = int(early_stopping_patience)

        self.history: list[EpochResult] = []
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.epochs_since_best = 0
        self.best_ckpt = self.out_dir / "best.pt"

        setup_logger("geoconstruct", log_file=self.out_dir / "train.log")

    def _current_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        total_n = 0
        all_pred: list[int] = []
        all_true: list[int] = []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for f1, f2, y, _ in loader:
                f1 = f1.to(self.device)
                f2 = f2.to(self.device)
                y = y.to(self.device).long()
                logits, _ = self.model(f1, f2)
                loss = self.loss_fn(logits, y)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                total_loss += loss.item() * y.size(0)
                total_n += y.size(0)
                all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
                all_true.extend(y.cpu().tolist())
        return total_loss / max(1, total_n), all_pred, all_true

    def _step_scheduler(self, val_metric: float) -> None:
        if self.scheduler is None:
            return
        if self.scheduler_type == "plateau":
            self.scheduler.step(val_metric)
        else:
            self.scheduler.step()

    def fit(self, n_epochs: int) -> list[EpochResult]:
        log.info(
            "Starting training: max %d epochs on %s | scheduler=%s | "
            "early_stopping_patience=%d",
            n_epochs, self.device, self.scheduler_type,
            self.early_stopping_patience,
        )
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss, _, _ = self._run_epoch(self.train_loader, train=True)
            val_loss, vp, vy = self._run_epoch(self.val_loader, train=False)
            metrics = compute_metrics(vy, vp, self.num_classes)
            self._step_scheduler(metrics["macro_f1"])

            res = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=metrics["accuracy"],
                val_macro_f1=metrics["macro_f1"],
                lr=self._current_lr(),
            )
            self.history.append(res)
            elapsed = time.time() - t0
            log.info(
                "Epoch %3d | train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                "val_macroF1=%.4f lr=%.2e | %.1fs",
                epoch, train_loss, val_loss, metrics["accuracy"],
                metrics["macro_f1"], res.lr, elapsed,
            )

            if metrics["macro_f1"] > self.best_f1:
                self.best_f1 = metrics["macro_f1"]
                self.best_epoch = epoch
                self.epochs_since_best = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "best_f1": self.best_f1,
                    },
                    self.best_ckpt,
                )
                log.info("  ↑ new best macroF1=%.4f saved to %s",
                         self.best_f1, self.best_ckpt)
            else:
                self.epochs_since_best += 1

            if (self.early_stopping_patience > 0
                    and self.epochs_since_best >= self.early_stopping_patience):
                log.info(
                    "Early stopping: no val macro-F1 improvement for %d epochs "
                    "(best=%.4f at epoch %d).",
                    self.epochs_since_best, self.best_f1, self.best_epoch,
                )
                break

        history_path = self.out_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump([asdict(h) for h in self.history], f, indent=2)
        log.info("History saved to %s", history_path)
        return self.history
