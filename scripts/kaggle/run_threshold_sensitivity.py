"""Weak-label majority-vote threshold sensitivity analysis.

Addresses Reviewer 1 comment #5: "The five-variant fusion ablation in Table
III is informative, but the ablation does not investigate the effect of the
3/5 majority-vote threshold... A sensitivity analysis on the vote threshold
alone would make the weak-label protocol more convincing."

For each threshold t in {2, 3, 4} out of 5:
  1. Rebuild binary labels from LEVIR-CC captions using the same keyword
     rule but with the required-agreement count changed.
  2. Retrain the concat MLP head on the resulting labels.
  3. Report train/val/test class balance and test macro-F1.
  4. Report the drop rate (fraction of pairs left unlabelled).

Note: this rebuilds labels via geoconstruct.data.captions.build_labels
directly. It does not touch outputs/labels/*.json — the results are written
to a separate directory so the main paper's labels are preserved.

Output
------
outputs/runs/threshold_sensitivity/
    labels_min_votes_{t}/{split}_labels.json
    metrics_min_votes_{t}.json     macro-F1 for each threshold
    summary.json                   trade-off table
    log.txt

Runtime
-------
* CPU: ~30 minutes total (3 label rebuilds + 3 MLP trainings)
* GPU: not required (train.py can use CPU or GPU)
* Disk: <30 MB output
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)
if ROOT is None:
    raise SystemExit("Repo root not found.")
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("threshold_sensitivity")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _write_config(base_config_path, labels_dir, out_dir, dest_path):
    """train.py/eval.py have no --override flag; materialise a real config
    YAML instead, copied from the base ablation config with paths swapped."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["labels_dir"] = str(labels_dir)
    cfg["run"]["out_dir"] = str(out_dir)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return dest_path


def rebuild_labels(min_votes, out_labels_dir):
    """Call build_weak_labels.py with the given majority-vote threshold."""
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,
           str(ROOT / "scripts/build_weak_labels.py"),
           "--captions", str(ROOT / "data/LEVIR_CC/Levir-CC-dataset/LevirCCcaptions.json"),
           "--min-votes", str(min_votes),
           "--out", str(out_labels_dir)]
    log.info(f"Rebuilding labels with min_votes={min_votes}: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"build_weak_labels failed: {r.stderr[-2000:]}")
        raise RuntimeError("label build failed")
    log.info(r.stdout[-500:])


def train_head(labels_dir, run_out):
    """Train the concat MLP on the given labels."""
    config_path = Path(run_out) / "train_config.yaml"
    _write_config(ROOT / "configs/ablation_concat.yaml", labels_dir, run_out, config_path)
    cmd = [sys.executable,
           str(ROOT / "scripts/train.py"),
           "--config", str(config_path)]
    log.info(f"Training head: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"train.py failed: {r.stderr[-2000:]}")
        raise RuntimeError("train failed")
    log.info(r.stdout[-500:])


def evaluate(run_out, labels_dir):
    """Test the trained checkpoint on the corresponding test labels.

    eval.py reads paths/config straight from the YAML the run was trained
    with (config_used.json is only a dumped record, not consumable by
    load_config, which expects YAML) -- reuse the same train_config.yaml
    written in train_head(), which already points at the right labels_dir
    and out_dir (so --checkpoint defaults to <out_dir>/best.pt correctly).
    """
    config_path = Path(run_out) / "train_config.yaml"
    cmd = [sys.executable,
           str(ROOT / "scripts/test.py"),
           "--config", str(config_path)]
    log.info(f"Testing: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(f"test.py failed: {r.stderr[-2000:]}")
        raise RuntimeError("test failed")
    log.info(r.stdout[-500:])


def load_split_counts(labels_dir):
    """Count no_change, completed per split."""
    counts = {}
    for split in ("train", "val", "test"):
        p = Path(labels_dir) / f"{split}_labels.json"
        if not p.exists():
            counts[split] = {"n": 0, "no_change": 0, "completed": 0}
            continue
        with open(p) as f:
            d = json.load(f)
        d = {k: int(v) for k, v in d.items() if int(v) >= 0}
        counts[split] = {
            "n": len(d),
            "no_change": sum(1 for v in d.values() if v == 0),
            "completed": sum(1 for v in d.values() if v == 1),
        }
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "outputs/runs/threshold_sensitivity"))
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_root / "log.txt")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    results = {}
    for t in args.thresholds:
        log.info(f"===== min_votes = {t}/5 =====")
        labels_dir = out_root / f"labels_min_votes_{t}"
        run_out = out_root / f"train_min_votes_{t}"
        try:
            rebuild_labels(t, labels_dir)
            counts = load_split_counts(labels_dir)
            log.info(f"class counts: {counts}")
            train_head(labels_dir, run_out)
            evaluate(run_out, labels_dir)
            metrics_path = run_out / "metrics_test.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                macro = metrics.get("macro_f1", None)
            else:
                macro = None
            results[str(t)] = {
                "min_votes": t,
                "counts": counts,
                "macro_f1_test": macro,
            }
        except Exception as e:
            log.error(f"threshold {t} failed: {e}")
            results[str(t)] = {"min_votes": t, "error": str(e)}

    with open(out_root / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 60)
    log.info("SUMMARY threshold sensitivity")
    for t, r in results.items():
        if "error" in r:
            log.info(f"  min_votes={t}: FAILED ({r['error']})")
            continue
        n_train = r["counts"]["train"]["n"]
        macro = r["macro_f1_test"]
        log.info(f"  min_votes={t}: n_train={n_train}  macro_F1={macro}")
    log.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
