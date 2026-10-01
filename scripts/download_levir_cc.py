"""
Download LEVIR-CC dataset from Hugging Face.

LEVIR-CC = LEVIR Change Captioning: 10,077 bi-temporal 256x256 image pairs +
50,385 human-written captions (5 per pair). Built on top of LEVIR-CD by
Chen-Yang Liu et al.

Hugging Face host: https://huggingface.co/datasets/lcybuaa/LEVIR-CC

Run locally:
    pip install huggingface_hub tqdm
    python scripts/download_levir_cc.py

The dataset will be placed under: data/LEVIR_CC/
Total size: ~2.5 GB. Time: 15-30 min depending on connection.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_levir_cc")


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub tqdm")
        return 1

    # Project root = parent of this script's parent (scripts/ -> project root)
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "data" / "LEVIR_CC"
    target_dir.mkdir(parents=True, exist_ok=True)

    log.info("Target directory: %s", target_dir)
    log.info("Repo: lcybuaa/LEVIR-CC on Hugging Face Hub")
    log.info("Starting download (this can take 15-30 min)...")

    try:
        snapshot_download(
            repo_id="lcybuaa/LEVIR-CC",
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        log.exception("Download failed: %s", exc)
        log.info(
            "If the error is auth-related, try: "
            "`huggingface-cli login` and rerun. "
            "Otherwise check disk space and connection."
        )
        return 2

    captions = target_dir / "LevirCCcaptions.json"
    images_dir = target_dir / "images"
    if captions.exists():
        log.info("Captions file: OK (%s)", captions)
    else:
        log.warning("Captions file not found at expected path: %s", captions)

    if images_dir.exists():
        for split in ("train", "val", "test"):
            split_a = images_dir / split / "A"
            n = len(list(split_a.glob("*.png"))) if split_a.exists() else 0
            log.info("Split %s: %d image pairs", split, n)
    else:
        log.warning("Images directory not found: %s", images_dir)

    log.info("Done. Next step: run `scripts/build_weak_labels.py` (coming next).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
