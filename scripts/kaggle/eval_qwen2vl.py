"""Qwen2.5-VL-7B-Instruct zero-shot baseline on LEVIR-CC bi-temporal pairs.

Addresses JARS Reviewer 1 comment #1 (major revision): "The experiments only
evaluate LLaVA-1.5-7B. More recent models such as Qwen2.5-VL and GeoLLaVA are
not evaluated... The authors should evaluate at least one additional recent
7B VLM."

Uses the EXACT SAME prompt and parsing protocol as scripts/eval_vlm_zero_shot.py
(the script that produced the paper's LLaVA-7B numbers), so the two rows are
directly comparable in the paper.

Model: Qwen/Qwen2.5-VL-7B-Instruct, loaded in 4-bit via bitsandbytes.

Output (resumable, safe to re-run / continue across Kaggle sessions)
----------------------------------------------------------------------
outputs/runs/qwen2vl_zero_shot/
    predictions_test.jsonl   one line per pair: {id, label, raw, pred}
    metrics_test.json        final aggregate metrics (written at the end
                              and every time you re-run once all pairs done)
    eval.log

Kaggle runtime
--------------
* GPU: T4 or P100, ~8-12 sec/pair in 4-bit -> full 1849-pair test split is
  ~5-6 GPU-hours. Fits one 8-hour session with margin, but if the kernel
  dies partway through, just re-run the SAME command: already-scored pairs
  are skipped automatically (resume via the JSONL).
* VRAM: ~9-10 GB in 4-bit.
* Disk: <2 MB output.

Usage
-----
    # Quick sanity check on 100 random pairs first (~15-20 min)
    !cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_qwen2vl.py --sample 100

    # Full test split (resumable; just re-run if the kernel dies)
    !cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_qwen2vl.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

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
    raise SystemExit("Repo root not found. Run from repo root or clone to /kaggle/working/GeoConstruct-R1")
sys.path.insert(0, str(ROOT / "src"))

from geoconstruct.data.captions import CLASS_NAMES  # noqa: E402
from geoconstruct.evaluation.metrics import compute_metrics, format_report  # noqa: E402

log = logging.getLogger("eval_qwen2vl")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# --------------------------------------------------------------------------- #
# Workaround for a broken CUDA install on some Colab/Kaggle images: torch has
# no precompiled CUDA kernel for integer-dtype .prod() (it JIT-compiles one
# via NVRTC on first use), and if libnvrtc-builtins.so is missing/mismatched
# on the box, that JIT compile crashes with a wall of CUDA template source in
# the traceback. Qwen2.5-VL calls image_grid_thw.prod(-1) on a CUDA int64
# tensor internally (just to get per-image patch counts), which triggers
# this. Moving the whole tensor to CPU broke device-consistency elsewhere in
# the model, so instead we patch only the reduction itself: run it on CPU and
# move the (tiny) result back to the original device, keeping every other
# tensor exactly where it was.
_orig_tensor_prod = torch.Tensor.prod


def _cuda_safe_prod(self, *args, **kwargs):
    if self.is_cuda and not torch.is_floating_point(self):
        return _orig_tensor_prod(self.cpu(), *args, **kwargs).to(self.device)
    return _orig_tensor_prod(self, *args, **kwargs)


torch.Tensor.prod = _cuda_safe_prod

# Identical prompt to scripts/eval_vlm_zero_shot.py (the LLaVA baseline script)
# so the two rows are a fair, same-protocol comparison in the paper.
PROMPT = """You are looking at two satellite images of the same place taken at \
different times. The first image is the "before" view and the second is the \
"after" view.

Decide whether new buildings have appeared between the two images.

Answer with EXACTLY one word, lowercase, no punctuation:
- "completed" if one or more new buildings, houses, villas, or similar \
structures have appeared in the after image that were not in the before image.
- "no_change" if the scene is essentially unchanged (no new structures).

Answer:"""


def _parse(text: str) -> int:
    """Same mapping rule as eval_vlm_zero_shot.py::_parse."""
    t = (text or "").strip().lower()
    for ch in [".", ",", "!", "?", "\"", "'", "`", "*", "_", "\n"]:
        t = t.replace(ch, " ")
    t = t.split()[0] if t.split() else t
    if t in ("completed", "completion", "built", "new", "yes", "change", "changed"):
        return 1
    if t in ("no_change", "nochange", "no", "same", "unchanged", "nothing", "none"):
        return 0
    return -1


def load_model(device: str):
    """Load Qwen2.5-VL-7B-Instruct in 4-bit. Tries the 2.5-specific class
    first, falls back to the generic Qwen2VL class for older transformers."""
    from transformers import AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        log.warning("Qwen2_5_VLForConditionalGeneration not found in this "
                    "transformers version; falling back to Qwen2VLForConditionalGeneration. "
                    "Run `pip install -U transformers` if this fails to load the checkpoint.")
        from transformers import Qwen2VLForConditionalGeneration as ModelCls
    model = ModelCls.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16)
    model.eval()
    return model, processor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--labels-dir",  default=str(ROOT / "outputs/labels"))
    ap.add_argument("--split",       default="test", choices=["val", "test"])
    ap.add_argument("--out",         default=str(ROOT / "outputs/runs/qwen2vl_zero_shot"))
    ap.add_argument("--sample",      type=int, default=0,
                    help="If > 0, evaluate a random subset instead of the full split.")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info(f"Device: {device}")

    # Load labels
    labels_path = Path(args.labels_dir) / f"{args.split}_labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    items = [(fname, int(y)) for fname, y in labels.items() if int(y) >= 0]
    log.info(f"Eval pool: {len(items)} (split={args.split})")

    if args.sample > 0:
        rng = random.Random(args.seed)
        items = rng.sample(items, min(args.sample, len(items)))
        log.info(f"Sampled {len(items)} for a quick run")

    images_root = Path(args.images_root) / args.split

    # Resume support: skip pairs already present in the JSONL
    jsonl_path = out_dir / f"predictions_{args.split}.jsonl"
    done_ids: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        log.info(f"Resuming: {len(done_ids)} pairs already scored")

    remaining = [(f, y) for f, y in items if f not in done_ids]
    if not remaining:
        log.info("Nothing left to score; jumping straight to aggregation.")
    else:
        log.info(f"Loading {MODEL_ID} in 4-bit...")
        model, processor = load_model(device)

        t0 = time.time()
        with open(jsonl_path, "a", encoding="utf-8") as fout:
            for i, (fname, y) in enumerate(tqdm(remaining, desc="qwen2.5-vl", ncols=80), 1):
                a_path = images_root / "A" / fname
                b_path = images_root / "B" / fname
                if not (a_path.exists() and b_path.exists()):
                    log.warning(f"Skipping missing pair {fname}")
                    continue
                img_a = Image.open(a_path).convert("RGB")
                img_b = Image.open(b_path).convert("RGB")

                convo = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before image:"},
                        {"type": "image", "image": img_a},
                        {"type": "text", "text": "After image:"},
                        {"type": "image", "image": img_b},
                        {"type": "text", "text": PROMPT},
                    ],
                }]
                prompt_text = processor.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt_text], images=[img_a, img_b],
                                   return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs,
                                         max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                gen = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0]
                pred = _parse(gen)
                rec = {"id": fname, "label": y, "raw": gen, "pred": pred}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()

                if i % 50 == 0:
                    rate = i / max(1.0, time.time() - t0)
                    eta_min = (len(remaining) - i) / max(rate, 1e-6) / 60
                    log.info(f"Progress: {i}/{len(remaining)}  "
                            f"({rate:.2f}/s, ETA {eta_min:.0f} min)")

    # Aggregate over everything scored so far (resume-safe)
    y_true, y_pred = [], []
    n_unparseable = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["pred"] < 0:
                n_unparseable += 1
                continue
            y_true.append(r["label"]); y_pred.append(r["pred"])

    if not y_true:
        log.warning("No scored pairs yet; nothing to aggregate.")
        return 0

    metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    report = format_report(y_true, y_pred, target_names=CLASS_NAMES)
    log.info("\n" + report)
    log.info(f"macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")

    with open(out_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "backend": "hf_transformers", "model": MODEL_ID,
                   "n_scored": len(y_true), "n_unparseable": n_unparseable}, f, indent=2)
    log.info(f"Wrote {out_dir / f'metrics_{args.split}.json'}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
