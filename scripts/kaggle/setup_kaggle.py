"""Kaggle environment setup and sanity check.

Run this ONCE at the start of any Kaggle notebook session for this project.
It verifies that GPU, dataset, and dependencies are ready before you invest
Kaggle GPU-hours on the actual experiments.

What it does
------------
1. Prints Python / torch / CUDA versions
2. Verifies GPU is available and reports memory
3. Reports disk usage in /kaggle/working (Kaggle output folder, 19 GB limit)
4. Installs missing Python packages (transformers, accelerate, bitsandbytes,
   open_clip_torch, huggingface_hub, scikit-learn)
5. Verifies dataset structure (LEVIR-CC images, label JSONs, feature cache)
6. Prints a short GO/NO-GO summary

Usage on Kaggle
---------------
    !python scripts/kaggle/setup_kaggle.py

If everything is green, you're safe to run any of the downstream Priority 1
scripts (RemoteCLIP, few-shot LLaVA, single-image LLaVA).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Kaggle notebooks run in /kaggle/working. We support two layouts:
#   1. Repo cloned into /kaggle/working/GeoConstruct-R1
#   2. Repo cloned into current working directory (local runs)
ROOT_CANDIDATES = [
    Path("/kaggle/working/GeoConstruct-R1"),
    Path.cwd(),
    Path(__file__).resolve().parent.parent.parent,
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "src" / "geoconstruct").exists()), None)

REQUIRED_PACKAGES = [
    # (import name, pip name)
    ("torch",              "torch"),
    ("torchvision",        "torchvision"),
    ("transformers",       "transformers>=4.42"),
    ("accelerate",         "accelerate>=0.30"),
    ("bitsandbytes",       "bitsandbytes>=0.43"),
    ("open_clip",          "open_clip_torch>=2.24"),
    ("huggingface_hub",    "huggingface_hub>=0.23"),
    ("sklearn",            "scikit-learn>=1.3"),
    ("PIL",                "pillow"),
    ("numpy",              "numpy"),
    ("pandas",             "pandas"),
    ("matplotlib",         "matplotlib"),
    ("tqdm",               "tqdm"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_python() -> None:
    section("1. Python and system")
    print(f"Python:  {sys.version.split()[0]} ({sys.executable})")
    print(f"cwd:     {Path.cwd()}")
    print(f"repo:    {ROOT if ROOT else 'NOT FOUND'}")


def check_torch_gpu() -> tuple[bool, str]:
    section("2. Torch / CUDA / GPU")
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed")
        return False, "torch missing"
    print(f"torch:            {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[WARN] No GPU visible. Enable GPU in Kaggle notebook settings.")
        return False, "no GPU"
    n = torch.cuda.device_count()
    print(f"GPU count:        {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB")
    return True, "GPU ok"


def check_disk() -> tuple[bool, str]:
    section("3. Disk usage (Kaggle output limit: 19 GB)")
    for target in ("/kaggle/working", "/kaggle/input", str(Path.cwd())):
        if Path(target).exists():
            usage = shutil.disk_usage(target)
            free_gb = usage.free / 1024**3
            used_gb = (usage.total - usage.free) / 1024**3
            total_gb = usage.total / 1024**3
            print(f"{target}:  {used_gb:.1f} GB used / {total_gb:.1f} GB total "
                  f"({free_gb:.1f} GB free)")
    return True, "disk ok"


def install_packages() -> tuple[bool, str]:
    section("4. Package check / install")
    missing = []
    for imp, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(imp)
            print(f"  [ok]      {imp}")
        except ImportError:
            print(f"  [missing] {imp}  -> will install {pip_name}")
            missing.append(pip_name)
    if not missing:
        print("All required packages already present.")
        return True, "packages ok"
    print(f"\nInstalling: {' '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--no-warn-script-location"] + missing
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[FAIL] pip install returned non-zero.")
        print(result.stdout[-2000:]); print(result.stderr[-2000:])
        return False, "pip install failed"
    print("Installation ok.")
    return True, "packages ok"


def check_dataset() -> tuple[bool, str]:
    section("5. Dataset / labels / features presence")
    if ROOT is None:
        print("[FAIL] Repo root not found. Clone the repo into /kaggle/working "
              "or cd into it before running this script.")
        return False, "repo missing"
    files_to_check = [
        ROOT / "src/geoconstruct/data/captions.py",
        ROOT / "outputs/labels/train_labels.json",
        ROOT / "outputs/labels/val_labels.json",
        ROOT / "outputs/labels/test_labels.json",
        ROOT / "outputs/features/train_features.npz",
        ROOT / "outputs/features/val_features.npz",
        ROOT / "outputs/features/test_features.npz",
    ]
    missing = [p for p in files_to_check if not p.exists()]
    for p in files_to_check:
        tag = "[ok]     " if p.exists() else "[missing]"
        print(f"  {tag} {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    if missing:
        print(f"\n[WARN] {len(missing)} files missing.")
        print("If features are missing, you can regenerate with "
              "scripts/extract_clip_features.py (needs GPU).")
        return False, f"{len(missing)} files missing"
    # Report label class balance
    print("\nWeak-label class balance (from JSON):")
    for split in ("train", "val", "test"):
        p = ROOT / f"outputs/labels/{split}_labels.json"
        with open(p) as f:
            d = json.load(f)
        n0 = sum(1 for v in d.values() if v == 0)
        n1 = sum(1 for v in d.values() if v == 1)
        print(f"  {split:5s}: n={len(d)}  no_change={n0}  completed={n1}")
    return True, "dataset ok"


def summarise(results: list[tuple[str, bool, str]]) -> None:
    section("SUMMARY")
    all_ok = True
    for name, ok, msg in results:
        tag = "[ok]  " if ok else "[FAIL]"
        print(f"  {tag} {name}: {msg}")
        all_ok = all_ok and ok
    print()
    if all_ok:
        print(">>> Environment is READY. You can now run downstream scripts.")
    else:
        print(">>> Some checks failed. Fix them before running experiments.")


def main() -> int:
    check_python()
    r1 = check_torch_gpu()
    r2 = check_disk()
    r3 = install_packages()
    r4 = check_dataset()
    summarise([
        ("GPU",       *r1),
        ("Disk",      *r2),
        ("Packages",  *r3),
        ("Dataset",   *r4),
    ])
    return 0 if all(r[0] for r in (r1, r2, r3, r4)) else 1


if __name__ == "__main__":
    sys.exit(main())
