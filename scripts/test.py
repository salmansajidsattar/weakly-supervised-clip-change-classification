"""Test-set runner. Same as eval.py but forces --split test and always saves
fused features for downstream t-SNE figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    cmd = [sys.executable, str(Path(__file__).parent / "eval.py"),
           "--config", args.config, "--split", "test", "--save-fused"]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
