"""Zero-shot CLIP baseline: no training, only cosine matching.

For each test pair we form a fused image feature (default: f2 - f1, the
temporal difference, which historically aligns with "what changed"). We
compare its cosine similarity against the per-class text-template embeddings
and take argmax as the predicted class.

This is the fair "no supervision" baseline: same backbone, same templates,
same test split. Whatever performance gap exists between this and our
trained head is *purely* what the weak supervision bought us.

Usage:
    python scripts/eval_clip_zero_shot.py \
        --features-dir outputs/features \
        --labels-dir outputs/labels \
        --out outputs/runs/clip_zero_shot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.captions import CLASS_NAMES, CLASS_TEMPLATES
from geoconstruct.evaluation.metrics import compute_metrics, format_report
from geoconstruct.models.clip_backbone import CLIPBackbone
from geoconstruct.utils.logging import setup_logger
from geoconstruct.utils.seed import seed_everything

log = logging.getLogger("eval_clip_zero_shot")


def _fuse(feat_a: np.ndarray, feat_b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "diff":
        f = feat_b - feat_a
    elif mode == "concat":
        f = np.concatenate([feat_a, feat_b], axis=-1)
    elif mode == "t2":
        f = feat_b
    elif mode == "mean":
        f = 0.5 * (feat_a + feat_b)
    else:
        raise ValueError(f"Unknown fuse mode: {mode}")
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.maximum(norm, 1e-12)


def _encode_class_centroids(backbone: CLIPBackbone, mode: str) -> torch.Tensor:
    """For each class, encode all templates and average → class centroid (D,).

    Returns (K, D) L2-normalized. For 'concat' mode we duplicate the centroid
    so the dimensions match (2D); cosine in concat space is equivalent to
    cosine in (D, D) when templates are the same in both halves.
    """
    centroids: list[torch.Tensor] = []
    for cls in CLASS_NAMES:
        z = backbone.encode_texts(CLASS_TEMPLATES[cls])  # (k, D), normalized
        c = z.mean(dim=0)
        c = c / c.norm()
        centroids.append(c)
    base = torch.stack(centroids, dim=0)  # (K, D)
    if mode == "concat":
        base = torch.cat([base, base], dim=-1) / (2 ** 0.5)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs/features")
    ap.add_argument("--labels-dir", default="outputs/labels")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="outputs/runs/clip_zero_shot")
    ap.add_argument("--fuse", default="diff",
                    choices=["diff", "concat", "t2", "mean"],
                    help="How to combine f1 and f2 before cosine-matching. "
                         "'diff' is the natural choice for change semantics; "
                         "'t2' matches the t2_only ablation regime.")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("geoconstruct", log_file=out_dir / "eval.log")
    setup_logger("eval_clip_zero_shot", log_file=out_dir / "eval.log")
    seed_everything(0, deterministic=True)

    # Load cached test features
    cache_path = Path(args.features_dir) / f"{args.split}_features.npz"
    log.info("Loading features: %s", cache_path)
    cache = np.load(cache_path, allow_pickle=True)
    feat_a, feat_b = cache["feat_a"], cache["feat_b"]
    ids = [str(x) for x in cache["ids"].tolist()]

    labels_path = Path(args.labels_dir) / f"{args.split}_labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    keep = [(i, ids.index(i), int(v)) for i, v in labels.items()
            if int(v) >= 0 and i in ids]
    log.info("Eval samples after filtering: %d", len(keep))
    sel_idx = [k[1] for k in keep]
    y_true = [k[2] for k in keep]
    fa = feat_a[sel_idx]; fb = feat_b[sel_idx]

    fused = _fuse(fa, fb, args.fuse)
    log.info("Fused features shape: %s (mode=%s)", fused.shape, args.fuse)

    # Encode class text centroids
    backbone = CLIPBackbone(model_name=args.model, pretrained=args.pretrained,
                            device="cpu")
    centroids = _encode_class_centroids(backbone, args.fuse).numpy()  # (K, D)
    log.info("Class centroid bank: %s", centroids.shape)

    # Cosine and argmax
    sims = fused @ centroids.T    # (N, K)
    y_pred = sims.argmax(axis=-1).tolist()

    num_classes = len(CLASS_NAMES)
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    report = format_report(y_true, y_pred, target_names=CLASS_NAMES)
    log.info("\n%s", report)
    log.info("macro-F1=%.4f accuracy=%.4f", metrics["macro_f1"], metrics["accuracy"])

    with open(out_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "fuse": args.fuse, "model": args.model,
                   "pretrained": args.pretrained, "n_eval": len(keep)}, f, indent=2)
    log.info("Wrote %s", out_dir / f"metrics_{args.split}.json")

    # Dump per-pair predictions for downstream statistical tests.
    pred_path = out_dir / f"predictions_{args.split}_{args.fuse}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for (pair_id, _, y), yp in zip(keep, y_pred):
            f.write(json.dumps({"id": pair_id, "label": int(y),
                                "pred": int(yp), "fuse": args.fuse}) + "\n")
    log.info("Wrote %s", pred_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
