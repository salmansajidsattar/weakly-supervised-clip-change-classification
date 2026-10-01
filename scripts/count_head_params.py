"""Print exact trainable parameter counts for every head variant.

Addresses JARS Reviewer 1 comment #5: Table 2 / Table 5 / abstract report
inconsistent parameter counts (~1M vs ~83K vs ~962K) for the same concat
head. This script builds each variant directly from the real configs and
prints ground-truth counts so the paper can report one correct number
everywhere.

Usage:
    python scripts/count_head_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.models.temporal_head import TemporalHead


def count(variant: str, **kwargs) -> int:
    head = TemporalHead(variant=variant, **kwargs)
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def main() -> int:
    embed_dim, hidden, num_classes, dropout = 512, 128, 2, 0.1

    for variant in ("t1_only", "t2_only", "diff", "concat"):
        n = count(variant, embed_dim=embed_dim, hidden=hidden,
                  num_classes=num_classes, dropout=dropout)
        print(f"{variant:10s}: {n:,} params (~{n/1000:.0f}K)")

    n_full = count("full", embed_dim=embed_dim, d_model=256, n_heads=1,
                    ff_dim=512, hidden=hidden, num_classes=num_classes,
                    dropout=dropout)
    print(f"{'full':10s}: {n_full:,} params (~{n_full/1000:.0f}K)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
