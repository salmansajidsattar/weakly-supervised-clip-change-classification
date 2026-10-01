"""RemoteCLIP zero-shot evaluation on LEVIR-CC test split.

Addresses Reviewer 1 comment #1: "explicitly compare against at least one
existing remote-sensing CLIP variant (e.g., RemoteCLIP or SatCLIP) used with
the same weak-label protocol."

RemoteCLIP is a CLIP variant continued from OpenAI CLIP weights on a large
aerial image-caption corpus (Liu et al., IEEE TGRS 2024). This script uses
the ViT-B/32 checkpoint so the parameter count is identical to the OpenAI
CLIP baseline in the main paper. Only the pretraining data differs.

What it does
------------
1. Downloads RemoteCLIP-ViT-B-32 weights from HuggingFace
2. Loads them into an open_clip ViT-B/32 skeleton
3. Re-extracts image features for every labelled test pair
4. Encodes the same class-template texts used in the main paper
5. Runs cosine matching in three fusion modes: t2, diff, concat
6. Writes metrics + per-pair predictions in the same format as
   scripts/eval_clip_zero_shot.py, so downstream stats/plotting scripts
   just pick up a new run directory

Output
------
outputs/runs/remoteclip_zero_shot/
    metrics_test_t2.json
    metrics_test_diff.json
    metrics_test_concat.json
    predictions_test_t2.jsonl
    predictions_test_diff.jsonl
    predictions_test_concat.jsonl
    eval.log

Kaggle runtime
--------------
* GPU:  ~10-15 min (feature extraction dominates)
* Disk: ~5 MB total output
* VRAM: <2 GB

Usage
-----
    !cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_remoteclip.py
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
from PIL import Image
from tqdm import tqdm

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found. Run from repo root or clone to /kaggle/working/GeoConstruct-R1")
sys.path.insert(0, str(ROOT / "src"))

from geoconstruct.data.captions import CLASS_NAMES, CLASS_TEMPLATES  # noqa: E402
from geoconstruct.evaluation.metrics import compute_metrics, format_report  # noqa: E402

log = logging.getLogger("eval_remoteclip")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_FILE = "RemoteCLIP-ViT-B-32.pt"
IMG_SIZE = 224


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_remoteclip(device: str):
    """Load RemoteCLIP ViT-B/32 into an open_clip skeleton."""
    import open_clip
    from huggingface_hub import hf_hub_download

    log.info("Downloading RemoteCLIP weights from HuggingFace...")
    ckpt_path = hf_hub_download(REMOTECLIP_REPO, REMOTECLIP_FILE)

    log.info("Loading open_clip ViT-B-32 skeleton...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    log.info(f"Loading RemoteCLIP state dict from {ckpt_path}...")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(f"State dict mismatch. missing={len(missing)} "
                    f"unexpected={len(unexpected)}")

    model = model.to(device).eval()
    return model, preprocess, tokenizer


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_image_features(model, preprocess, image_paths, device, batch_size=32):
    """Encode a list of image paths into L2-normalised feature vectors."""
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size),
                      desc="Encoding images", ncols=80):
            batch_paths = image_paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            batch = torch.stack(imgs).to(device)
            z = model.encode_image(batch)
            z = F.normalize(z, dim=-1)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def encode_class_centroids(model, tokenizer, device, mode: str):
    """Encode class templates and return (K, D) L2-normalised centroids."""
    centroids = []
    with torch.no_grad():
        for cls in CLASS_NAMES:
            tokens = tokenizer(CLASS_TEMPLATES[cls]).to(device)
            z = model.encode_text(tokens)
            z = F.normalize(z, dim=-1)
            c = z.mean(dim=0)
            c = F.normalize(c, dim=-1)
            centroids.append(c.cpu())
    base = torch.stack(centroids, dim=0)          # (K, D)
    if mode == "concat":
        base = torch.cat([base, base], dim=-1) / (2 ** 0.5)
    return base.numpy()


def fuse(fa, fb, mode: str):
    if mode == "diff":
        f = fb - fa
    elif mode == "concat":
        f = np.concatenate([fa, fb], axis=-1)
    elif mode == "t2":
        f = fb
    elif mode == "mean":
        f = 0.5 * (fa + fb)
    else:
        raise ValueError(mode)
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    return f / np.maximum(norm, 1e-12)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--labels-dir",  default=str(ROOT / "outputs/labels"))
    ap.add_argument("--split",       default="test", choices=["val", "test"])
    ap.add_argument("--out",         default=str(ROOT / "outputs/runs/remoteclip_zero_shot"))
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--fuse-modes",  nargs="+", default=["t2", "diff", "concat"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info(f"Device: {device}")

    # Load labels
    labels_path = Path(args.labels_dir) / f"{args.split}_labels.json"
    with open(labels_path) as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
    log.info(f"Loaded {len(labels)} labelled pairs from {labels_path}")

    # Build image path lists
    images_root = Path(args.images_root)
    pair_ids = sorted(labels.keys())
    a_paths, b_paths, y_true = [], [], []
    for pid in pair_ids:
        # LEVIR-CC layout: <split>/A/<id>.png, <split>/B/<id>.png
        a = images_root / args.split / "A" / pid
        b = images_root / args.split / "B" / pid
        if not (a.exists() and b.exists()):
            # Fall back to layout without split subdir
            a = images_root / "A" / pid
            b = images_root / "B" / pid
        if not (a.exists() and b.exists()):
            log.warning(f"Skipping missing pair {pid}")
            continue
        a_paths.append(a); b_paths.append(b); y_true.append(labels[pid])
    log.info(f"Prepared {len(a_paths)} pairs after existence check")

    # Load RemoteCLIP
    model, preprocess, tokenizer = load_remoteclip(device)

    # Extract features
    log.info("Encoding pre-change images (A)...")
    fa = extract_image_features(model, preprocess, a_paths, device, args.batch_size)
    log.info("Encoding post-change images (B)...")
    fb = extract_image_features(model, preprocess, b_paths, device, args.batch_size)
    log.info(f"Feature shapes: fa={fa.shape}, fb={fb.shape}")

    # Cache features for downstream reuse (small, ~15 MB)
    feat_cache = out_dir / f"{args.split}_features.npz"
    np.savez_compressed(feat_cache,
                        feat_a=fa, feat_b=fb,
                        ids=np.array(pair_ids[:len(a_paths)], dtype=object))
    log.info(f"Cached features to {feat_cache}")

    # For each fusion mode, run cosine matching
    for mode in args.fuse_modes:
        log.info(f"--- fuse={mode} ---")
        cent = encode_class_centroids(model, tokenizer, device, mode)
        fused = fuse(fa, fb, mode)
        sims = fused @ cent.T
        y_pred = sims.argmax(axis=-1).tolist()
        metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
        report = format_report(y_true, y_pred, target_names=CLASS_NAMES)
        log.info("\n" + report)
        log.info(f"macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

        with open(out_dir / f"metrics_{args.split}_{mode}.json", "w") as f:
            json.dump({**metrics, "fuse": mode, "backbone": "RemoteCLIP-ViT-B-32",
                       "n_eval": len(y_true)}, f, indent=2)
        with open(out_dir / f"predictions_{args.split}_{mode}.jsonl", "w") as f:
            for pid, y, yp in zip(pair_ids[:len(a_paths)], y_true, y_pred):
                f.write(json.dumps({"id": pid, "label": int(y),
                                    "pred": int(yp), "fuse": mode,
                                    "backbone": "RemoteCLIP"}) + "\n")
        log.info(f"Wrote metrics + predictions for fuse={mode}")

    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
