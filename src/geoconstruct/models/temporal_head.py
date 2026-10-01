"""The trainable head. ~1-1.5M params, runs comfortably on CPU.

Variants for ablation (selected via config.head.variant):
  - "t1_only"   : MLP on f1 only
  - "t2_only"   : MLP on f2 only
  - "concat"    : MLP on [f1; f2]
  - "diff"      : MLP on (f2 - f1)
  - "full"      : [f1; f2; f2-f1] -> 1-head cross-attention -> MLP (default)

All variants share the same MLP head architecture so trainable param counts
are comparable and the ablation is honest about the temporal-fusion delta.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _mlp(in_dim: int, hidden: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, num_classes),
    )


class _T1OrT2Only(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, num_classes: int, dropout: float, use: str):
        super().__init__()
        assert use in {"t1", "t2"}
        self.use = use
        self.mlp = _mlp(embed_dim, hidden, num_classes, dropout)

    def forward(self, f1, f2):  # type: ignore[override]
        x = f1 if self.use == "t1" else f2
        return self.mlp(x), x  # return logits and the fused feature


class _Concat(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.mlp = _mlp(embed_dim * 2, hidden, num_classes, dropout)

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=-1)
        return self.mlp(x), x


class _Diff(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.mlp = _mlp(embed_dim, hidden, num_classes, dropout)

    def forward(self, f1, f2):
        x = f2 - f1
        return self.mlp(x), x


class _Full(nn.Module):
    """[f1; f2; f2-f1] projected to d_model, then 1 transformer block over the
    3-token sequence, mean-pool, MLP head."""

    def __init__(
        self,
        embed_dim: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        hidden: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.proj_t1 = nn.Linear(embed_dim, d_model)
        self.proj_t2 = nn.Linear(embed_dim, d_model)
        self.proj_diff = nn.Linear(embed_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.mlp = _mlp(d_model, hidden, num_classes, dropout)

    def forward(self, f1, f2):
        diff = f2 - f1
        t1 = self.proj_t1(f1)
        t2 = self.proj_t2(f2)
        td = self.proj_diff(diff)
        seq = torch.stack([t1, t2, td], dim=1)   # (B, 3, d_model)
        out = self.attn(seq)                     # (B, 3, d_model)
        pooled = out.mean(dim=1)                 # (B, d_model)
        return self.mlp(pooled), pooled


class TemporalHead(nn.Module):
    """Top-level wrapper selected by variant."""

    def __init__(self, variant: str, **kwargs):
        super().__init__()
        self.variant = variant
        if variant == "t1_only":
            self.net = _T1OrT2Only(use="t1", **_take(kwargs, "embed_dim", "hidden", "num_classes", "dropout"))
        elif variant == "t2_only":
            self.net = _T1OrT2Only(use="t2", **_take(kwargs, "embed_dim", "hidden", "num_classes", "dropout"))
        elif variant == "concat":
            self.net = _Concat(**_take(kwargs, "embed_dim", "hidden", "num_classes", "dropout"))
        elif variant == "diff":
            self.net = _Diff(**_take(kwargs, "embed_dim", "hidden", "num_classes", "dropout"))
        elif variant == "full":
            self.net = _Full(**_take(
                kwargs, "embed_dim", "d_model", "n_heads", "ff_dim",
                "hidden", "num_classes", "dropout"
            ))
        else:
            raise ValueError(f"Unknown head variant: {variant}")

    def forward(self, f1, f2):
        return self.net(f1, f2)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _take(d: dict, *keys: str) -> dict:
    return {k: d[k] for k in keys if k in d}


def build_head(cfg) -> TemporalHead:
    """Build TemporalHead from a Config object (utils.config.Config)."""
    head = TemporalHead(
        variant=cfg.head.variant,
        embed_dim=cfg.head.embed_dim,
        d_model=cfg.head.get("d_model", 256),
        n_heads=cfg.head.get("n_heads", 1),
        ff_dim=cfg.head.get("ff_dim", 512),
        hidden=cfg.head.get("hidden", 128),
        num_classes=cfg.head.num_classes,
        dropout=cfg.head.get("dropout", 0.1),
    )
    log.info("Built head variant=%s | trainable params=%d",
             cfg.head.variant, head.n_params())
    return head
