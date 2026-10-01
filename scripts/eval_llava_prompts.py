"""LLaVA-7B prompt-sensitivity study on a small balanced subsample.

Evaluates three prompt variants on the same 200 test pairs (100 no_change,
100 completed), all served through the local Ollama OpenAI-compatible
endpoint. Reports macro-F1, parse failure rate, and per-class recall for
each prompt.

Usage (with Ollama running and llava:7b pulled):
    python scripts/eval_llava_prompts.py \\
        --images-root data/LEVIR_CC/Levir-CC-dataset/images \\
        --labels-dir outputs/labels \\
        --out outputs/prompt_sensitivity
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from geoconstruct.data.captions import CLASS_NAMES
from geoconstruct.evaluation.metrics import compute_metrics
from geoconstruct.utils.logging import setup_logger

log = logging.getLogger("eval_llava_prompts")

PROMPTS = {
    "direct": (
        "You are looking at two satellite images of the same place. The first "
        "is the BEFORE view and the second is the AFTER view. Have one or more "
        "new buildings, houses, or villas appeared in the after image that were "
        "not in the before image?\n"
        "Answer with EXACTLY one word, lowercase, no punctuation:\n"
        "- \"completed\" if new structures appeared.\n"
        "- \"no_change\" if the scene is essentially unchanged.\n"
        "Answer:"
    ),
    "json": (
        "Compare the two satellite images (before / after). Decide whether new "
        "buildings have appeared.\n"
        "Respond with a single JSON object on one line, no markdown:\n"
        "{\"label\": \"completed\"}  or  {\"label\": \"no_change\"}\n"
        "JSON:"
    ),
    "fewshot": (
        "Decide whether new buildings appeared between the before and after "
        "satellite views.\n"
        "Example 1: scene is identical bare land -> label = no_change\n"
        "Example 2: bare land in before, several new villas in after -> label = completed\n"
        "Now classify the current pair. Reply with exactly: \"completed\" or "
        "\"no_change\". No other words.\n"
        "Answer:"
    ),
}


def _b64(p: Path) -> str:
    return base64.standard_b64encode(p.read_bytes()).decode("ascii")


def _parse(text: str) -> int:
    if text is None:
        return -1
    t = text.strip().lower()
    # try JSON shape first
    if "{" in t and "label" in t:
        try:
            obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
            v = str(obj.get("label", "")).lower().strip()
            t = v or t
        except Exception:
            pass
    for ch in [".", ",", "!", "?", "\"", "'", "`", "*", "_", "\n"]:
        t = t.replace(ch, " ")
    t = t.split()[0] if t.split() else t
    if t in {"completed", "completion", "built", "new", "yes", "change", "changed"}:
        return 1
    if t in {"no_change", "nochange", "no", "same", "unchanged", "nothing", "none"}:
        return 0
    return -1


def _call(client, prompt: str, ia: Path, ib: Path,
          max_retries: int = 3, base_wait: float = 2.0) -> str:
    content = [
        {"type": "text", "text": "Before image:"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_b64(ia)}"}},
        {"type": "text", "text": "After image:"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_b64(ib)}"}},
        {"type": "text", "text": prompt},
    ]
    last = None
    for k in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="llava:7b", max_tokens=20,
                messages=[{"role": "user", "content": content}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            log.warning("Call failed (%d/%d): %s", k + 1, max_retries, e)
            time.sleep(base_wait * (2 ** k))
    raise RuntimeError(f"All retries failed: {last}")


def _build_balanced_sample(labels_path: Path, n_per_class: int, seed: int):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    pools = {0: [], 1: []}
    for fname, lab in labels.items():
        lab = int(lab)
        if lab in pools:
            pools[lab].append(fname)
    rng = random.Random(seed)
    sample = []
    for c in (0, 1):
        rng.shuffle(pools[c])
        sample.extend((f, c) for f in pools[c][:n_per_class])
    rng.shuffle(sample)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root",
                    default="data/LEVIR_CC/Levir-CC-dataset/images")
    ap.add_argument("--labels-dir", default="outputs/labels")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/prompt_sensitivity")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("eval_llava_prompts", log_file=out_dir / "eval.log")

    try:
        import openai
    except ImportError:
        log.error("pip install openai"); return 1
    client = openai.OpenAI(base_url=args.base_url, api_key="ollama")

    sample = _build_balanced_sample(
        Path(args.labels_dir) / f"{args.split}_labels.json",
        args.n_per_class, args.seed,
    )
    log.info("Balanced sample: %d pairs (%d per class).",
             len(sample), args.n_per_class)
    images_root = Path(args.images_root) / args.split

    aggregate: dict = {}
    for pname, prompt in PROMPTS.items():
        jsonl = out_dir / f"predictions_{pname}.jsonl"
        done = set()
        if jsonl.exists():
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try: done.add(json.loads(line)["id"])
                    except Exception: pass
        log.info("Prompt=%s | already done: %d", pname, len(done))

        with open(jsonl, "a", encoding="utf-8") as fout:
            for i, (fname, y) in enumerate(sample, 1):
                if fname in done: continue
                ia = images_root / "A" / fname
                ib = images_root / "B" / fname
                try:
                    raw = _call(client, prompt, ia, ib)
                except Exception as e:
                    log.error("Hard fail on %s: %s", fname, e); continue
                pred = _parse(raw)
                fout.write(json.dumps({"id": fname, "label": y,
                                       "raw": raw, "pred": pred}) + "\n")
                fout.flush()
                if i % 25 == 0:
                    log.info("  %s: %d / %d", pname, i, len(sample))

        # Aggregate
        y_true, y_pred = [], []; n_fail = 0
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["pred"] < 0:
                    n_fail += 1
                else:
                    y_true.append(r["label"]); y_pred.append(r["pred"])
        m = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
        log.info("Prompt=%s | macro-F1=%.4f | parse-fail=%d/%d (%.1f%%)",
                 pname, m["macro_f1"], n_fail, len(sample),
                 100.0 * n_fail / max(1, len(sample)))
        aggregate[pname] = {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "per_class_recall": m["per_class"]["recall"],
            "parse_fail": n_fail,
            "parse_fail_rate": n_fail / max(1, len(sample)),
            "n_scored": len(y_true),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    # LaTeX-style row for Table V
    print("\n% Paste these into the Table V stub.")
    print(r"% Prompt style & Macro-F1 & Parse-fail \% & no\_change recall \\")
    print(r"% \midrule")
    nice = {"direct": "Direct binary", "json": "Strict JSON output",
            "fewshot": "Few-shot (2 examples)"}
    for k in ("direct", "json", "fewshot"):
        a = aggregate[k]
        nc_recall = a["per_class_recall"][0]
        print(f"% {nice[k]:<25} & {a['macro_f1']:.3f} & "
              f"{100*a['parse_fail_rate']:.1f} & {nc_recall:.3f} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
