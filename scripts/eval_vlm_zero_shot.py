"""Frontier VLM zero-shot baseline.

Sends each (before, after) image pair to a hosted vision-language model
through an API, asks it to classify the pair as "no_change" or "completed",
parses the answer, and reports test-set metrics.

Supported backends:
    claude   - Anthropic (Claude Sonnet 4.6 by default)
    openai   - OpenAI (gpt-4o-mini by default)
    gemini   - Google (gemini-2.5-flash by default; FREE TIER available)

API key is read from the corresponding env var:
    ANTHROPIC_API_KEY   for backend=claude
    OPENAI_API_KEY      for backend=openai
    GEMINI_API_KEY      for backend=gemini  (alias: GOOGLE_API_KEY)

Cost / quota for the full LEVIR-CC test split (~1849 pairs after filtering):
    Claude 3.5/4 Sonnet:    ~ $15-40
    GPT-4o:                 ~ $20-50
    GPT-4o-mini:            ~ $1-5
    Gemini 2.5 Flash:       FREE on free tier (subject to daily/RPM limits)
                            or ~ $1-5 on paid tier

Usage:
    # Sanity check on 100 random pairs (FREE on Gemini)
    python scripts/eval_vlm_zero_shot.py --backend gemini --sample 100

    # Full test set
    python scripts/eval_vlm_zero_shot.py --backend gemini

Predictions and raw responses are saved incrementally to a JSONL file so a
crash or interrupt doesn't lose progress; rerun with the same --out and the
script skips pairs that are already in the JSONL.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from geoconstruct.data.captions import CLASS_NAMES
from geoconstruct.evaluation.metrics import compute_metrics, format_report
from geoconstruct.utils.logging import setup_logger

log = logging.getLogger("eval_vlm_zero_shot")

PROMPT = """You are looking at two satellite images of the same place taken at \
different times. The first image is the "before" view and the second is the \
"after" view.

Decide whether new buildings have appeared between the two images.

Answer with EXACTLY one word, lowercase, no punctuation:
- "completed" if one or more new buildings, houses, villas, or similar \
structures have appeared in the after image that were not in the before image.
- "no_change" if the scene is essentially unchanged (no new structures).

Answer:"""


def _b64(p: Path) -> str:
    return base64.standard_b64encode(p.read_bytes()).decode("ascii")


def _parse(text: str) -> int:
    """Map a free-text response to a class index. -1 means unparseable."""
    t = (text or "").strip().lower()
    for ch in [".", ",", "!", "?", "\"", "'", "`", "*", "_", "\n"]:
        t = t.replace(ch, " ")
    t = t.split()[0] if t.split() else t
    if t in ("completed", "completion", "built", "new", "yes",
             "change", "changed"):
        return 1
    if t in ("no_change", "nochange", "no", "same", "unchanged",
             "nothing", "none"):
        return 0
    return -1


def _call_claude(client, model, img_a_path, img_b_path,
                 max_retries=3, retry_wait=2.0):
    content = [
        {"type": "text", "text": "Before image:"},
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/png",
                                      "data": _b64(img_a_path)}},
        {"type": "text", "text": "After image:"},
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/png",
                                      "data": _b64(img_b_path)}},
        {"type": "text", "text": PROMPT},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text
        except Exception as e:
            last_err = e
            log.warning("Claude call failed (%d/%d): %s",
                        attempt + 1, max_retries, e)
            time.sleep(retry_wait * (2 ** attempt))
    raise RuntimeError(f"Claude failed: {last_err}")


def _call_openai(client, model, img_a_path, img_b_path,
                 max_retries=3, retry_wait=2.0):
    content = [
        {"type": "text", "text": "Before image:"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_b64(img_a_path)}"}},
        {"type": "text", "text": "After image:"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{_b64(img_b_path)}"}},
        {"type": "text", "text": PROMPT},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": content}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            log.warning("OpenAI call failed (%d/%d): %s",
                        attempt + 1, max_retries, e)
            time.sleep(retry_wait * (2 ** attempt))
    raise RuntimeError(f"OpenAI failed: {last_err}")


def _call_gemini(client, model, img_a_path, img_b_path,
                 max_retries=3, retry_wait=2.0):
    """Google Gemini via google-genai SDK."""
    from google.genai import types as gtypes
    img_a = gtypes.Part.from_bytes(data=img_a_path.read_bytes(),
                                    mime_type="image/png")
    img_b = gtypes.Part.from_bytes(data=img_b_path.read_bytes(),
                                    mime_type="image/png")
    contents = [
        gtypes.Part.from_text(text="Before image:"),
        img_a,
        gtypes.Part.from_text(text="After image:"),
        img_b,
        gtypes.Part.from_text(text=PROMPT),
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.0,
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            log.warning("Gemini call failed (%d/%d): %s",
                        attempt + 1, max_retries, e)
            time.sleep(retry_wait * (2 ** attempt))
    raise RuntimeError(f"Gemini failed: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root",
                    default="data/LEVIR_CC/Levir-CC-dataset/images")
    ap.add_argument("--labels-dir", default="outputs/labels")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="outputs/runs/vlm_zero_shot")
    ap.add_argument("--backend", default="gemini",
                    choices=["claude", "openai", "gemini"])
    ap.add_argument("--model", default=None,
                    help="Override default model per backend. "
                         "Defaults: claude=claude-sonnet-4-6, "
                         "openai=gpt-4o-mini, gemini=gemini-2.5-flash")
    ap.add_argument("--sample", type=int, default=0,
                    help="If > 0, evaluate on a random subset.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-url", default=None,
                    help="Override API base URL. Use "
                         "http://localhost:11434/v1 for a local Ollama server. "
                         "Only applies to --backend openai.")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds to sleep between calls. Use 4.0 to stay "
                         "well under Gemini free-tier 15 RPM.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("geoconstruct", log_file=out_dir / "eval.log")
    setup_logger("eval_vlm_zero_shot", log_file=out_dir / "eval.log")

    # Build backend
    if args.backend == "claude":
        try:
            import anthropic
        except ImportError:
            log.error("pip install anthropic"); return 1
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.error("Set ANTHROPIC_API_KEY"); return 1
        client = anthropic.Anthropic()
        model = args.model or "claude-sonnet-4-6"
        call = lambda a, b: _call_claude(client, model, a, b)
    elif args.backend == "openai":
        try:
            import openai
        except ImportError:
            log.error("pip install openai"); return 1
        # If pointing at a local server, allow a dummy/missing key.
        kwargs = {}
        if args.base_url:
            kwargs["base_url"] = args.base_url
            kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "ollama"
            log.info("Using custom base_url: %s", args.base_url)
        else:
            if not os.environ.get("OPENAI_API_KEY"):
                log.error("Set OPENAI_API_KEY (or pass --base-url for a local server)"); return 1
        client = openai.OpenAI(**kwargs)
        model = args.model or "gpt-4o-mini"
        call = lambda a, b: _call_openai(client, model, a, b)
    else:  # gemini
        try:
            from google import genai
        except ImportError:
            log.error("pip install google-genai"); return 1
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log.error("Set GEMINI_API_KEY (or GOOGLE_API_KEY)"); return 1
        client = genai.Client(api_key=api_key)
        model = args.model or "gemini-2.5-flash"
        call = lambda a, b: _call_gemini(client, model, a, b)

    log.info("Backend=%s model=%s", args.backend, model)

    # Load labels and image paths
    labels_path = Path(args.labels_dir) / f"{args.split}_labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    items = [(fname, int(y)) for fname, y in labels.items() if int(y) >= 0]
    log.info("Eval pool: %d (split=%s)", len(items), args.split)

    if args.sample > 0:
        rng = random.Random(args.seed)
        items = rng.sample(items, min(args.sample, len(items)))
        log.info("Sampled %d for cheap run", len(items))

    images_root = Path(args.images_root) / args.split

    # Resume support via JSONL
    jsonl_path = out_dir / f"predictions_{args.split}.jsonl"
    done_ids: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        log.info("Resuming: %d already done", len(done_ids))

    n_call = 0; n_skip = 0
    t0 = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for i, (fname, y) in enumerate(items, 1):
            if fname in done_ids:
                n_skip += 1; continue
            a_path = images_root / "A" / fname
            b_path = images_root / "B" / fname
            try:
                raw = call(a_path, b_path)
            except Exception as e:
                log.error("Hard failure on %s: %s", fname, e); continue
            pred = _parse(raw)
            rec = {"id": fname, "label": y, "raw": raw, "pred": pred}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n_call += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if n_call % 25 == 0:
                rate = n_call / max(1.0, time.time() - t0)
                log.info("Progress: %d called, %d skipped (%.1f/s)",
                         n_call, n_skip, rate)

    # Aggregate
    y_true: list[int] = []; y_pred: list[int] = []
    n_unparseable = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["pred"] < 0:
                n_unparseable += 1; continue
            y_true.append(r["label"]); y_pred.append(r["pred"])

    log.info("Total scored: %d | unparseable: %d", len(y_true), n_unparseable)
    metrics = compute_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))
    report = format_report(y_true, y_pred, target_names=CLASS_NAMES)
    log.info("\n%s", report)
    log.info("macro-F1=%.4f accuracy=%.4f", metrics["macro_f1"], metrics["accuracy"])

    with open(out_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "backend": args.backend, "model": model,
                   "n_scored": len(y_true), "n_unparseable": n_unparseable}, f, indent=2)
    log.info("Wrote %s", out_dir / f"metrics_{args.split}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
