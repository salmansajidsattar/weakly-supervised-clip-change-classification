"""Run all ablation variants × 3 seeds each, then report a unified table.

Trains and evaluates every variant (t1_only, t2_only, concat, diff, full)
at seeds {42, 123, 999} for fair head-to-head comparison with mean ± std.

Reuses cached CLIP features — each individual run takes 3-8 minutes on CPU.
Total: 15 runs × ~5 min = ~50-75 min wall-clock.

Output:
    outputs/ablation_summary.json    machine-readable
    outputs/ablations.log            human-readable
    outputs/runs/<variant>_seed<N>/  per-run artifacts (history, best.pt, metrics_test.json)
"""

from __future__ import annotations

import json
import logging
import statistics as st
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from geoconstruct.utils.logging import setup_logger

log = setup_logger("run_ablations", log_file=PROJECT / "outputs" / "ablations.log")

VARIANTS = [
    ("t1_only", PROJECT / "configs/ablation_t1_only.yaml"),
    ("t2_only", PROJECT / "configs/ablation_t2_only.yaml"),
    ("concat",  PROJECT / "configs/ablation_concat.yaml"),
    ("diff",    PROJECT / "configs/ablation_diff.yaml"),
    ("full",    PROJECT / "configs/default.yaml"),
]
SEEDS = [42, 123, 999]


def _run(cmd: list[str]) -> int:
    log.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd)


def _write_seeded_config(base_cfg_path: Path, variant: str, seed: int,
                         tmp_path: Path) -> Path:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["run"]["seed"] = seed
    cfg["run"]["out_dir"] = f"outputs/runs/{variant}_seed{seed}"
    out = tmp_path / f"{variant}_seed{seed}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main() -> int:
    tmp = PROJECT / "outputs" / "_tmp_configs"
    tmp.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[float]] = {}

    for variant, base_cfg in VARIANTS:
        summary[variant] = []
        for seed in SEEDS:
            seeded_cfg = _write_seeded_config(base_cfg, variant, seed, tmp)
            rc_train = _run([sys.executable, str(PROJECT / "scripts/train.py"),
                             "--config", str(seeded_cfg)])
            if rc_train != 0:
                log.error("Training failed for %s seed=%d", variant, seed)
                continue
            rc_eval = _run([sys.executable, str(PROJECT / "scripts/test.py"),
                            "--config", str(seeded_cfg)])
            if rc_eval != 0:
                log.error("Test failed for %s seed=%d", variant, seed)
                continue
            metrics_file = (PROJECT / "outputs" / "runs"
                            / f"{variant}_seed{seed}" / "metrics_test.json")
            with open(metrics_file, "r", encoding="utf-8") as f:
                m = json.load(f)
            summary[variant].append(m["macro_f1"])

    # Print and persist summary table
    log.info("=" * 70)
    log.info("Ablation summary (test macro-F1, 3 seeds each)")
    log.info("%-12s | %-22s | %s", "variant", "mean ± std", "individual seeds")
    log.info("-" * 70)
    out_table: dict[str, dict] = {}
    for variant, scores in summary.items():
        if not scores:
            log.info("%-12s | %-22s | (no runs)", variant, "n/a")
            out_table[variant] = {"mean": None, "std": None, "scores": []}
            continue
        m = st.mean(scores)
        s = st.pstdev(scores) if len(scores) > 1 else 0.0
        seeds_str = ", ".join(f"{x:.4f}" for x in scores)
        log.info("%-12s | %.4f ± %.4f       | %s", variant, m, s, seeds_str)
        out_table[variant] = {"mean": m, "std": s, "scores": scores}

    out = PROJECT / "outputs" / "ablation_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_table, f, indent=2)
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
