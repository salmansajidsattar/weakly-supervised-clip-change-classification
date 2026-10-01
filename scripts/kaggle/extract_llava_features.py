"""Extract LLaVA-7B vision-tower features for all pairs.

Addresses Reviewer 1 comment #2 (deeper): a linear probe on LLaVA-7B's visual
features disentangles representation quality from the zero-shot prompt
interface. If the linear probe on LLaVA features reaches close to the CLIP-
supervised head's accuracy, then LLaVA's visual encoder is strong; the
bottleneck is in the language-generation interface, not in vision.

What it does
------------
Loads LLaVA-1.5-7B (4-bit quantised) and, for every labelled pair in
train / val / test, runs both I1 and I2 through the vision tower and pools
to a single feature per image. Features cached to disk.

The train logistic-regression step is deferred to
train_linear_probe_llava.py so that the (slow) GPU feature extraction is
separated from the (fast) CPU probe training. This lets you re-run the
probe with different fusion modes or regularisation without re-extracting.

Output
------
outputs/features_llava/
    train_features.npz    keys: feat_a, feat_b, ids   dtype: float16
    val_features.npz      same
    test_features.npz     same

Kaggle runtime
--------------
* GPU: ~2-3 hours for all splits (~19,000 image encodings)
* Disk: features are ~200 MB total (4096-dim x 20k images x fp16)
    -> comfortably within Kaggle's 19 GB limit
* VRAM: ~10 GB (LLaVA-7B 4-bit)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")

log = logging.getLogger("extract_llava_features")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def image_paths(images_root, split, pair_id):
    a = Path(images_root) / split / "A" / pair_id
    b = Path(images_root) / split / "B" / pair_id
    if not a.exists():
        a = Path(images_root) / "A" / pair_id
        b = Path(images_root) / "B" / pair_id
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--labels-dir",  default=str(ROOT / "outputs/labels"))
    ap.add_argument("--splits",      nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-dir",     default=str(ROOT / "outputs/features_llava"))
    ap.add_argument("--batch-size",  type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {device}")

    log.info(f"Loading {LLAVA_MODEL_ID} in 4-bit...")
    from transformers import (AutoProcessor, LlavaForConditionalGeneration,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16)
    model.eval()

    # Access the vision tower directly for feature extraction
    # LLaVA-1.5 uses CLIP-ViT-L/14 with pooled last-hidden-state as the visual
    # feature (before projection to LLM space). We mean-pool the sequence.
    vision_tower = model.vision_tower

    def encode_images(image_paths_list):
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths_list), args.batch_size),
                          desc="encode", ncols=80):
                batch = image_paths_list[i:i + args.batch_size]
                imgs = [Image.open(p).convert("RGB") for p in batch]
                pixel = processor.image_processor(imgs, return_tensors="pt")["pixel_values"]
                pixel = pixel.to(device, torch.float16)
                vout = vision_tower(pixel, output_hidden_states=False)
                # last_hidden_state: (B, N_patches, D_vision)
                z = vout.last_hidden_state
                # Mean-pool over patch dimension
                z = z.mean(dim=1)                       # (B, D_vision)
                # L2 normalise
                z = torch.nn.functional.normalize(z, dim=-1)
                feats.append(z.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    for split in args.splits:
        log.info(f"===== split: {split} =====")
        with open(Path(args.labels_dir) / f"{split}_labels.json") as f:
            labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
        pair_ids = sorted(labels.keys())
        a_paths, b_paths, ids_ok = [], [], []
        for pid in pair_ids:
            a, b = image_paths(args.images_root, split, pid)
            if a.exists() and b.exists():
                a_paths.append(a); b_paths.append(b); ids_ok.append(pid)
        log.info(f"{split}: {len(a_paths)} pairs after existence check")

        log.info("Encoding A (pre-change)...")
        fa = encode_images(a_paths)
        log.info("Encoding B (post-change)...")
        fb = encode_images(b_paths)
        log.info(f"Feature shapes: fa={fa.shape}, fb={fb.shape}")

        # Cast to fp16 to save disk
        fa = fa.astype(np.float16)
        fb = fb.astype(np.float16)
        cache = out_dir / f"{split}_features.npz"
        np.savez_compressed(cache, feat_a=fa, feat_b=fb,
                            ids=np.array(ids_ok, dtype=object))
        log.info(f"Wrote {cache} ({cache.stat().st_size / 1024**2:.1f} MB)")

    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
