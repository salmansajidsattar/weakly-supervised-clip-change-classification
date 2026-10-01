"""One-time CLIP feature extraction for all LEVIR-CC pairs that have captions.

Reads the caption JSON (or the already-built label files) to know which
image IDs to encode. Skips orphan images in the zip that have no caption.

Runs CLIP ViT-B/32 (frozen) over every selected image and saves a per-split
.npz with:
    feat_a: (N, D) float32
    feat_b: (N, D) float32
    ids:    (N,)   str

Usage:
    python scripts/extract_clip_features.py \
        --images-root data/LEVIR_CC/Levir-CC-dataset/images \
        --labels-dir outputs/labels \
        --out outputs/features

Expected time on CPU (10,077 pairs, 20k images, batch 8):
    ~60-120 minutes total depending on machine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.models.clip_backbone import CLIPBackbone
from geoconstruct.utils.logging import setup_logger

log = logging.getLogger("extract_clip_features")


def _ids_from_labels(labels_dir: Path, split: str) -> set[str]:
    """Return the set of image filenames the caption JSON references."""
    p = labels_dir / f"{split}_labels.json"
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d.keys())


def _encode_split(backbone: CLIPBackbone, images_root: Path, split: str,
                  out_dir: Path, batch_size: int,
                  keep_ids: set[str] | None) -> None:
    a_dir = images_root / split / "A"
    available = sorted(p.name for p in a_dir.glob("*.png"))
    if keep_ids is not None:
        ids = [i for i in available if i in keep_ids]
        log.info("Split %s: %d images on disk, %d kept after caption filter",
                 split, len(available), len(ids))
    else:
        ids = available

    if not ids:
        log.warning("No images to encode for split=%s", split); return

    a_paths = [images_root / split / "A" / i for i in ids]
    b_paths = [images_root / split / "B" / i for i in ids]

    log.info("Split %s: encoding %d 'before' images...", split, len(a_paths))
    t0 = time.time()
    feat_a = backbone.encode_images(a_paths, batch_size=batch_size)
    log.info("  before done in %.1fs", time.time() - t0)

    log.info("Split %s: encoding %d 'after' images...", split, len(b_paths))
    t0 = time.time()
    feat_b = backbone.encode_images(b_paths, batch_size=batch_size)
    log.info("  after done in %.1fs", time.time() - t0)

    out_path = out_dir / f"{split}_features.npz"
    np.savez(
        out_path,
        feat_a=feat_a.numpy().astype(np.float32),
        feat_b=feat_b.numpy().astype(np.float32),
        ids=np.array(ids, dtype=object),
    )
    log.info("Saved %s | feat_a=%s feat_b=%s", out_path,
             tuple(feat_a.shape), tuple(feat_b.shape))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root",
                    default="data/LEVIR_CC/Levir-CC-dataset/images")
    ap.add_argument("--labels-dir", default="outputs/labels",
                    help="If set, only encode images that appear in "
                         "{split}_labels.json (saves ~20%% CPU time).")
    ap.add_argument("--out", default="outputs/features")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"],
                    help="Default order encodes test first (smallest) so you "
                         "can validate the pipeline before committing to "
                         "the long train run.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("geoconstruct", log_file=out_dir / "extract_clip_features.log")
    setup_logger("extract_clip_features",
                 log_file=out_dir / "extract_clip_features.log")

    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    backbone = CLIPBackbone(model_name=args.model, pretrained=args.pretrained,
                            device=args.device)

    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    for split in args.splits:
        keep = _ids_from_labels(labels_dir, split) if labels_dir else None
        _encode_split(backbone, Path(args.images_root), split, out_dir,
                      args.batch_size, keep_ids=keep)

    log.info("All splits done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
