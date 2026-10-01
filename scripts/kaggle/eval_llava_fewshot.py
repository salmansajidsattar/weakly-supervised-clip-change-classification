"""Few-shot LLaVA-7B evaluation on the balanced 200-pair test subset.

Addresses Reviewer 1 comment #2: "The comparison between LLaVA-7B and the
proposed method is asymmetric: the proposed method benefits from 6,526
training pairs while LLaVA-7B receives none... a few-shot or fine-tuned
LLaVA-7B variant" would be more informative.

This script runs LLaVA-1.5-7B under 0-shot, 4-shot, and 8-shot in-context
prompting. The in-context examples are sampled from the training split.
Uses HuggingFace transformers with 4-bit quantisation to fit LLaVA on a
Kaggle T4 (16 GB VRAM).

Compatible LLaVA checkpoint: llava-hf/llava-1.5-7b-hf, which is the same
base model as Ollama's llava:7b used in the earlier CPU experiments — so
the 0-shot number here should reproduce (within tokenization noise) the
number in the main paper's Table I.

Output
------
outputs/runs/llava_fewshot/
    predictions_{shot}_shot.jsonl
    metrics_{shot}_shot.json     for shot in {0, 4, 8}
    summary.json                 macro-F1 table
    eval.log

Kaggle runtime
--------------
* GPU:  ~3-4 hours for 200 pairs x 3 shot counts on T4
* Disk: <5 MB output
* VRAM: ~10 GB (LLaVA-7B 4-bit + processor + activations)

Usage
-----
    !cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_llava_fewshot.py
    !cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_llava_fewshot.py --shots 4 8   # skip 0-shot if already done
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
sys.path.insert(0, str(ROOT / "src"))

from geoconstruct.evaluation.metrics import compute_metrics, format_report  # noqa: E402

log = logging.getLogger("eval_llava_fewshot")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# The class labels the model must emit
CLASS_NAMES = ["no_change", "completed"]

# Direct prompt (matches the "direct" prompt in the existing paper)
QUESTION = ("Compare the two overhead satellite images of the same location, "
            "taken at different times. Has new construction (buildings, houses, "
            "villas, or similar structures) appeared in the second image that "
            "was not present in the first? Answer with a single word: "
            "'no_change' or 'completed'.")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_balanced_subset(labels_dir, split, n_per_class=100, seed=0):
    """Load a balanced subset of test pairs — the same protocol as the
    prompt-sensitivity study in the main paper."""
    with open(Path(labels_dir) / f"{split}_labels.json") as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
    by_class = {0: [], 1: []}
    for pid, y in labels.items():
        by_class[y].append(pid)
    rng = random.Random(seed)
    for k in by_class:
        rng.shuffle(by_class[k])
    picked = by_class[0][:n_per_class] + by_class[1][:n_per_class]
    rng.shuffle(picked)
    return [(pid, labels[pid]) for pid in picked]


def load_shot_examples(labels_dir, split, n_shots, seed=0):
    """Pick n_shots in-context examples from the training split, balanced."""
    if n_shots == 0:
        return []
    with open(Path(labels_dir) / f"{split}_labels.json") as f:
        labels = {k: int(v) for k, v in json.load(f).items() if int(v) >= 0}
    by_class = {0: [], 1: []}
    for pid, y in labels.items():
        by_class[y].append(pid)
    rng = random.Random(seed)
    for k in by_class:
        rng.shuffle(by_class[k])
    n_each = n_shots // 2
    picked = [(pid, 0) for pid in by_class[0][:n_each]] + \
             [(pid, 1) for pid in by_class[1][:n_each]]
    rng.shuffle(picked)
    return picked


def image_paths(images_root, split, pair_id):
    a = Path(images_root) / split / "A" / pair_id
    b = Path(images_root) / split / "B" / pair_id
    if not a.exists():
        a = Path(images_root) / "A" / pair_id
        b = Path(images_root) / "B" / pair_id
    return a, b


# --------------------------------------------------------------------------- #
# LLaVA-1.5 few-shot prompting
# --------------------------------------------------------------------------- #
def build_conversation(shot_examples, target_answer=None):
    """Build the multi-turn conversation the LLaVA processor expects.

    LLaVA-1.5 uses the conversation format:
      USER: <image>\n<image>\nQUESTION
      ASSISTANT: label
    We concatenate multiple such turns for few-shot in-context learning.
    """
    convos = []
    for pid, y in shot_examples:
        convos.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        })
        convos.append({
            "role": "assistant",
            "content": [{"type": "text", "text": CLASS_NAMES[y]}],
        })
    # Final query
    convos.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": QUESTION},
        ],
    })
    return convos


def parse_response(text: str) -> int:
    """Parse LLaVA's free-form response into 0 (no_change) / 1 (completed) / -1 (unparseable)."""
    t = text.strip().lower()
    # Strip any leading "assistant:" or similar
    for prefix in ("assistant:", "answer:", "output:", "response:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    # First-word match
    first = t.split()[0] if t.split() else ""
    if first.startswith("no_change") or first == "no" or first.startswith("no_"):
        return 0
    if first.startswith("completed") or first.startswith("complete"):
        return 1
    # Fallback: contains
    if "no_change" in t:
        return 0
    if "completed" in t or "new construction" in t or "new building" in t:
        return 1
    return -1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--labels-dir",  default=str(ROOT / "outputs/labels"))
    ap.add_argument("--split",       default="test")
    ap.add_argument("--out",         default=str(ROOT / "outputs/runs/llava_fewshot"))
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--shots",       type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info(f"Device: {device}")

    # Load LLaVA-1.5-7B (4-bit quantised)
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

    # Load test subset (balanced 200-pair — same as prompt-sensitivity study)
    subset = load_balanced_subset(args.labels_dir, args.split,
                                  n_per_class=args.n_per_class, seed=args.seed)
    log.info(f"Balanced test subset: {len(subset)} pairs")

    summary = {}
    for n_shots in args.shots:
        log.info(f"===== {n_shots}-shot =====")
        # Pick support examples from TRAIN split (never from test)
        shot_examples = load_shot_examples(args.labels_dir, "train",
                                           n_shots=n_shots, seed=args.seed)
        log.info(f"Support examples: {len(shot_examples)} from train split")

        y_true, y_pred, records = [], [], []
        n_unparseable = 0
        for pid, y in tqdm(subset, desc=f"{n_shots}-shot", ncols=80):
            a, b = image_paths(args.images_root, args.split, pid)
            if not (a.exists() and b.exists()):
                log.warning(f"Skipping missing pair {pid}")
                continue

            # Load target pair images
            target_a = Image.open(a).convert("RGB")
            target_b = Image.open(b).convert("RGB")

            # Load support pair images
            support_images = []
            for sp_pid, sp_y in shot_examples:
                sa, sb = image_paths(args.images_root, "train", sp_pid)
                if not (sa.exists() and sb.exists()):
                    log.warning(f"Skipping missing support pair {sp_pid}")
                    continue
                support_images.extend([Image.open(sa).convert("RGB"),
                                       Image.open(sb).convert("RGB")])
            all_images = support_images + [target_a, target_b]

            convo = build_conversation(shot_examples)
            prompt = processor.apply_chat_template(convo,
                                                   add_generation_prompt=True)

            inputs = processor(images=all_images, text=prompt,
                               return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                out = model.generate(**inputs,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            gen = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
            pred = parse_response(gen)
            if pred < 0:
                n_unparseable += 1
                records.append({"id": pid, "label": int(y),
                                "raw": gen, "pred": -1, "shots": n_shots})
                continue
            y_true.append(int(y)); y_pred.append(int(pred))
            records.append({"id": pid, "label": int(y),
                            "raw": gen, "pred": int(pred), "shots": n_shots})

        metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
        metrics.update({"n_scored": len(y_true), "n_unparseable": n_unparseable,
                        "shots": n_shots, "backend": "hf_transformers"})
        log.info("\n" + format_report(y_true, y_pred, target_names=CLASS_NAMES))
        log.info(f"macro_f1={metrics['macro_f1']:.4f}  "
                 f"unparseable={n_unparseable}")

        with open(out_dir / f"metrics_{n_shots}_shot.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / f"predictions_{n_shots}_shot.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summary[f"{n_shots}_shot"] = {
            "macro_f1": metrics["macro_f1"],
            "n_unparseable": n_unparseable,
            "n_scored": metrics["n_scored"],
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    log.info("SUMMARY")
    for k, v in summary.items():
        log.info(f"  {k}: macro-F1={v['macro_f1']:.3f}  "
                 f"unparseable={v['n_unparseable']}/{v['n_scored'] + v['n_unparseable']}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
