"""Tiny CLI for hand-validating a sample of weak labels.

Picks 50 random test-set pairs (default), opens each pair side-by-side in
your system image viewer, asks you to enter the true class, and writes
a comparison report when you finish.

Usage:
    python scripts/validate_weak_labels.py --n 50

Press k for completed, j for no_change, s to skip (mark ambiguous),
q to quit. Progress is written incrementally to a JSONL so you can
resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from geoconstruct.utils.logging import setup_logger
log = setup_logger("validate_weak_labels")

CLASSES = {"j": 0, "k": 1, "s": -1, "q": "QUIT"}


def _open_pair(images_root: Path, split: str, fname: str,
               out_dir: Path) -> None:
    """Render a small side-by-side HTML view of the pair and open it.

    Resolves both image paths to absolute paths (required by as_uri on
    Windows) and writes one HTML file we keep overwriting so the user
    just refreshes a single browser tab per pair.
    """
    a = (images_root / split / "A" / fname).resolve()
    b = (images_root / split / "B" / fname).resolve()
    viewer = (out_dir / "_viewer.html").resolve()
    viewer.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{fname}</title>
<style>body{{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:8px}}
.row{{display:flex;gap:8px}} .col{{flex:1;text-align:center}}
img{{max-width:100%;height:auto;border:1px solid #444}}
h2{{margin:4px 0;font-size:14px;font-weight:normal}}</style></head>
<body><h1 style='font-size:15px;margin:0 0 6px 0'>{fname}</h1>
<div class='row'>
<div class='col'><h2>BEFORE</h2><img src='{a.as_uri()}'></div>
<div class='col'><h2>AFTER</h2><img src='{b.as_uri()}'></div>
</div></body></html>""",
        encoding="utf-8",
    )
    webbrowser.open(viewer.as_uri(), new=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="outputs/labels")
    ap.add_argument("--images-root",
                    default="data/LEVIR_CC/Levir-CC-dataset/images")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/label_validation")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    journal = out / f"manual_{args.split}.jsonl"
    done = set()
    if journal.exists():
        with open(journal, "r", encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["id"])
                except Exception: pass
        log.info("Resuming: %d already done", len(done))

    with open(Path(args.labels_dir) / f"{args.split}_labels.json", "r",
              encoding="utf-8") as f:
        labels = json.load(f)
    pool = [(k, int(v)) for k, v in labels.items() if int(v) >= 0]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = [(k, v) for (k, v) in pool[:args.n] if k not in done]

    print(f"Validating {len(sample)} remaining pairs.")
    print("Press j=no_change, k=completed, s=skip(ambiguous), q=quit.\n")

    images_root = Path(args.images_root)
    with open(journal, "a", encoding="utf-8") as fout:
        for idx, (fname, weak) in enumerate(sample, 1):
            _open_pair(images_root, args.split, fname, out)
            print(f"[{idx}/{len(sample)}] {fname}  weak={weak}", end=" ")
            ch = input("your label> ").strip().lower()
            if ch == "q":
                print("Quit. Resume by rerunning.")
                break
            if ch not in CLASSES:
                print("  ignored, retry next time.")
                continue
            manual = CLASSES[ch]
            fout.write(json.dumps({"id": fname, "weak": weak,
                                   "manual": manual}) + "\n")
            fout.flush()

    # Report
    rows = []
    with open(journal, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    valid = [r for r in rows if r["manual"] >= 0]
    if not valid:
        print("No manual decisions to score yet."); return 0

    agree = sum(1 for r in valid if r["weak"] == r["manual"])
    n = len(valid)
    print(f"\nManual / weak agreement on {n} judged pairs: "
          f"{agree}/{n} = {agree/n:.3f}")

    # Cohen's kappa
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    y_w = [r["weak"]   for r in valid]
    y_m = [r["manual"] for r in valid]
    kappa = cohen_kappa_score(y_m, y_w)
    cm = confusion_matrix(y_m, y_w, labels=[0, 1])
    print(f"Cohen's kappa: {kappa:.3f}")
    print("Confusion (rows = manual, cols = weak):")
    print(cm)

    summary = {
        "n_judged": n,
        "agreement": agree / n,
        "cohens_kappa": kappa,
        "confusion": cm.tolist(),
        "ambiguous_skipped": sum(1 for r in rows if r["manual"] == -1),
    }
    with open(out / f"summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out / f'summary_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
