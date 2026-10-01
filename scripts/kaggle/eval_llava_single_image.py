"""Single-image LLaVA-7B diagnostic.

Addresses Reviewer 1 comment #8: "Running the VLM on individual pre-change
and post-change images separately and comparing to paired performance would
isolate whether the bottleneck is specifically temporal reasoning or domain
mismatch."

For each pair (I1, I2) in the balanced 200-pair test subset:
  1. Prompt LLaVA-7B with I1 alone -> ask "Does this satellite image show
     recently completed construction?"
  2. Prompt LLaVA-7B with I2 alone -> same question.

If LLaVA is fundamentally bad at satellite imagery (domain mismatch), it
will perform poorly on I2 alone even though "completed" pairs have new
construction visible in I2.
If LLaVA is fine on single satellite images but bad at comparing two, the
I2-only accuracy will be high and the paired accuracy will be the low one.

This cleanly separates the two hypotheses.

Output
------
outputs/runs/llava_single_image/
    predictions_I1.jsonl
    predictions_I2.jsonl
    metrics_I1.json
    metrics_I2.json
    summary.json         side-by-side comparison
    eval.log

Kaggle runtime
--------------
* GPU: ~2 hours on T4 (400 forward passes @ ~15 sec each)
* Disk: <2 MB output
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
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
    raise SystemExit("Repo root not found.")
sys.path.insert(0, str(ROOT / "src"))

from geoconstruct.evaluation.metrics import compute_metrics, format_report  # noqa: E402

log = logging.getLogger("eval_llava_single_image")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLASS_NAMES = ["no_change", "completed"]

# Single-image prompts: we ask about the presence of recently built structures.
# Both I1 and I2 use the same prompt for fair comparison.
QUESTION_I1 = ("You are shown a single overhead satellite image. Does this "
               "image show recently completed construction (new buildings, "
               "houses, villas, or similar structures)? Answer with a "
               "single word: 'no_change' or 'completed'.")
QUESTION_I2 = QUESTION_I1  # identical prompt


def load_balanced_subset(labels_dir, split, n_per_class=100, seed=0):
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


def image_paths(images_root, split, pair_id):
    a = Path(images_root) / split / "A" / pair_id
    b = Path(images_root) / split / "B" / pair_id
    if not a.exists():
        a = Path(images_root) / "A" / pair_id
        b = Path(images_root) / "B" / pair_id
    return a, b


def parse_response(text: str) -> int:
    t = text.strip().lower()
    for prefix in ("assistant:", "answer:", "output:", "response:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    first = t.split()[0] if t.split() else ""
    if first.startswith("no_change") or first == "no" or first.startswith("no_"):
        return 0
    if first.startswith("completed") or first.startswith("complete") or first.startswith("yes"):
        return 1
    if "no_change" in t or "no change" in t:
        return 0
    if "completed" in t or "new construction" in t or "new building" in t:
        return 1
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default=str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/images"))
    ap.add_argument("--labels-dir",  default=str(ROOT / "outputs/labels"))
    ap.add_argument("--split",       default="test")
    ap.add_argument("--out",         default=str(ROOT / "outputs/runs/llava_single_image"))
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--seed",        type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info(f"Device: {device}")

    # Load LLaVA
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

    subset = load_balanced_subset(args.labels_dir, args.split,
                                  n_per_class=args.n_per_class, seed=args.seed)
    log.info(f"Balanced test subset: {len(subset)} pairs")

    summary = {}
    for which, question in (("I1", QUESTION_I1), ("I2", QUESTION_I2)):
        log.info(f"===== single-image {which} =====")
        y_true, y_pred, records = [], [], []
        n_unparseable = 0
        for pid, y in tqdm(subset, desc=which, ncols=80):
            a, b = image_paths(args.images_root, args.split, pid)
            img_path = a if which == "I1" else b
            if not img_path.exists():
                log.warning(f"Skipping missing {which} for {pid}")
                continue
            img = Image.open(img_path).convert("RGB")

            convo = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            prompt = processor.apply_chat_template(convo,
                                                   add_generation_prompt=True)
            inputs = processor(images=[img], text=prompt,
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
                records.append({"id": pid, "label": int(y), "which": which,
                                "raw": gen, "pred": -1})
                continue
            y_true.append(int(y)); y_pred.append(int(pred))
            records.append({"id": pid, "label": int(y), "which": which,
                            "raw": gen, "pred": int(pred)})

        metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
        metrics.update({"which": which, "n_scored": len(y_true),
                        "n_unparseable": n_unparseable,
                        "backend": "hf_transformers"})
        log.info("\n" + format_report(y_true, y_pred, target_names=CLASS_NAMES))
        log.info(f"{which}: macro_f1={metrics['macro_f1']:.4f}")

        with open(out_dir / f"metrics_{which}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(out_dir / f"predictions_{which}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summary[which] = metrics

    # Comparison summary
    summary_out = {
        "single_image_I1": summary["I1"]["macro_f1"],
        "single_image_I2": summary["I2"]["macro_f1"],
        "notes": (
            "I2-only recall on 'completed' class is the key diagnostic: "
            "high recall implies LLaVA can see the buildings but fails at "
            "temporal reasoning; low recall implies domain-mismatch dominates."
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    log.info("=" * 60)
    log.info("SUMMARY")
    for k, v in summary_out.items():
        log.info(f"  {k}: {v}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
