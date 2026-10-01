# Weakly Supervised CLIP Outperforms a 7B Open-Source VLM on Bi-Temporal Satellite Change Classification

Official code, weak labels, and trained checkpoints for the IEEE GRSL letter:

> **Salman Sajid.** *Weakly Supervised CLIP Outperforms a 7B Open-Source Vision–Language Model on Bi-Temporal Satellite Change Classification.* IEEE Geoscience and Remote Sensing Letters, under review, 2026.
> Preprint: `arXiv:2606.XXXXX` *(to be assigned)*
> Affiliation: School of Electrical Engineering and Computer Science (SEECS), National University of Sciences and Technology (NUST), Islamabad, Pakistan
> Contact: `salmansajidsattar@gmail.com`

---

## TL;DR

We ask whether a current open-source 7B-parameter Vision-Language Model (VLM) can classify bi-temporal change on satellite imagery zero-shot. On a controlled 1849-pair test split from LEVIR-CC, we compare three supervision regimes that share the same labels and the same metric:

| Method | Trainable params | Macro-F1 [95% CI] |
|---|---|---|
| Random baseline | 0 | 0.501 [0.478, 0.523] |
| **LLaVA-7B zero-shot** (parseable only, n=1708) | 0 | 0.501 [0.478, 0.526] |
| LLaVA-7B zero-shot (worst-case) | 0 | 0.464 [0.440, 0.484] |
| Zero-shot CLIP, *f₂* only | 0 | 0.347 |
| Zero-shot CLIP, *f₂ − f₁* | 0 | 0.588 [0.563, 0.611] |
| **Frozen CLIP + concat head (ours)** | ~1 M | **0.882 [0.866, 0.896]** |

McNemar tests reject equality of the trained head with either zero-shot regime (LLaVA: χ²=423.5, p<10⁻¹⁰⁰; CLIP *f₂−f₁*: χ²=401.4, p<10⁻⁹⁹). A 44-pair hand validation of the caption-derived labels yielded κ=0.72 (substantial agreement).

**Read the result as a transfer limitation**, not a verdict on VLMs in general: a 7B open-source VLM does not, on its own, support discriminative bi-temporal satellite change reasoning within the prompt budget we explored, while ~1M parameters of caption-supervised classifier closes most of the gap.

The entire pipeline reproduces on a single CPU in under a day.

---

## What's in this repository

```
.
├── paper/                         # the LaTeX source and PDF of the letter
│   ├── grsl_letter.tex
│   ├── references.bib
│   ├── SUBMISSION_CHECKLIST.md    # internal: pre-submission steps
│   └── REVIEWER_DEFENSE.md        # internal: anticipated reviewer rebuttals
├── configs/                       # YAML configs, one per ablation variant
│   ├── default.yaml               # headline model: frozen CLIP + concat MLP
│   ├── ablation_t1_only.yaml
│   ├── ablation_t2_only.yaml
│   ├── ablation_concat.yaml
│   ├── ablation_diff.yaml
│   └── ablation_full.yaml         # cross-attention head, 12× larger
├── scripts/                       # runnable entry points
│   ├── download_levir_cc.py       # one-time dataset download
│   ├── build_weak_labels.py       # caption rule → binary labels
│   ├── extract_clip_features.py   # one-time CLIP feature cache
│   ├── train.py                   # train one config
│   ├── eval.py                    # evaluate one checkpoint
│   ├── test.py                    # test-split eval + dump fused features
│   ├── run_ablations.py           # 5 variants × 3 seeds
│   ├── run_classical_baselines.py # LogReg, Linear SVM, kNN on cached features
│   ├── eval_clip_zero_shot.py     # zero-shot CLIP baseline (no training)
│   ├── eval_vlm_zero_shot.py      # zero-shot LLaVA via Ollama
│   ├── eval_llava_prompts.py      # prompt-sensitivity study
│   ├── validate_weak_labels.py    # interactive 50-pair hand validation
│   ├── run_stats.py               # bootstrap CIs + McNemar on Table I
│   └── make_figures.py            # paper figures from a run directory
├── src/geoconstruct/              # the actual library
│   ├── data/                      # caption rules, weak-label builder, cached dataset
│   ├── models/                    # frozen CLIP, temporal head, retrieval explainer
│   ├── losses/                    # class-weighted cross entropy
│   ├── trainers/                  # CPU-friendly training loop
│   ├── evaluation/                # macro-F1 + sklearn report
│   ├── visualization/             # matplotlib-only figure helpers
│   └── utils/                     # seed, logging, YAML config
├── outputs/                       # all generated artefacts
│   ├── labels/                    # released weak labels (train/val/test JSON)
│   ├── stats/                     # bootstrap CIs + McNemar (Table I numbers)
│   └── runs/                      # checkpoints + metrics
├── requirements.txt
└── README.md                      # this file
```

---

## Reproducing the headline numbers

The full pipeline runs end-to-end on a laptop CPU; no GPU is required. Total wall-clock time is roughly 6–8 hours, dominated by the LLaVA-7B zero-shot evaluation. Everything else is well under one hour.

### 0. Environment

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

The full requirement list is in `requirements.txt`. Key packages:

- `torch` (CPU build is sufficient)
- `open_clip_torch` for the frozen CLIP ViT-B/32 backbone
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- `huggingface_hub`, `transformers` for dataset download
- `pyyaml`, `tqdm`

For the LLaVA-7B baseline you additionally need [Ollama](https://ollama.com):

```bash
# install Ollama (see https://ollama.com), then:
ollama pull llava:7b
ollama serve   # in a separate terminal
```

### 1. Dataset (~30 min, ~2.5 GB)

```bash
python scripts/download_levir_cc.py
```

This pulls LEVIR-CC from Hugging Face into `data/LEVIR_CC/`. The dataset is by Liu et al. (TGRS 2022); please cite their paper if you use it.

### 2. Build weak labels (1 min)

```bash
python scripts/build_weak_labels.py
```

Applies the caption-rule majority vote (≥3 of 5 captions per pair must agree on a class; ambiguous pairs are dropped). Writes per-split JSONs to `outputs/labels/`.

**The released label files** (`outputs/labels/{train,val,test}_labels.json`) are the artefact most readers will want. They map each pair ID to its binary label (`0` = no_change, `1` = completed) and are produced by the rules in `src/geoconstruct/data/captions.py`.

### 3. Extract frozen CLIP features (~20 min)

```bash
python scripts/extract_clip_features.py
```

Runs CLIP ViT-B/32 (OpenAI weights, via `open_clip`) once over both images of every labelled pair. Cached features are written to `outputs/features/{train,val,test}_features.npz`. All downstream training reads from this cache; the backbone is never re-run.

### 4. Train the headline head (3–8 min)

```bash
python scripts/train.py --config configs/default.yaml
```

Trains the `[f₁; f₂]` concat MLP on the weak labels with AdamW, lr=1e-3, batch=64, label smoothing 0.05, `ReduceLROnPlateau`, early stopping at patience 10. Best checkpoint is saved to `outputs/runs/concat_seed42/best.pt`.

### 5. Run the 5-variant × 3-seed ablation (~45 min)

```bash
python scripts/run_ablations.py
```

Produces Table III of the paper (fusion ablation: `f1`, `f2`, `f2-f1`, `[f1;f2]`, cross-attention).

### 6. Classical baselines (~2 min)

```bash
python scripts/run_classical_baselines.py
```

Trains Logistic Regression, Linear SVM, and k-NN on the same frozen features. Produces Table IV.

### 7. Zero-shot CLIP baselines (~3 min)

```bash
python scripts/eval_clip_zero_shot.py --fuse t2
python scripts/eval_clip_zero_shot.py --fuse diff
```

Per-pair predictions are saved to `outputs/runs/clip_zero_shot/predictions_test_*.jsonl`.

### 8. Zero-shot LLaVA-7B (~5 hr CPU)

```bash
# in one terminal: ollama serve
python scripts/eval_vlm_zero_shot.py
```

Predictions saved to `outputs/runs/vlm_zero_shot/predictions_test.jsonl`. The script handles parsing of LLaVA's free-form responses; 7.6% of responses fall outside the binary label vocabulary on our run and are reported separately.

### 9. Prompt-sensitivity study (~1 hr CPU)

```bash
python scripts/eval_llava_prompts.py
```

Evaluates LLaVA-7B under three prompt formulations (direct, JSON, few-shot) on a balanced 200-pair subsample of the test split. Produces Table V.

### 10. Hand validation of weak labels (~30 min of your time)

```bash
python scripts/validate_weak_labels.py --n 50
```

Opens each pair in your browser and asks you to vote `j` (no_change), `k` (completed), `s` (skip), or `q` (quit). Reports Cohen's κ at the end. Our run yielded κ = 0.72 on 44 judged pairs.

### 11. Statistical tests (~30 sec)

```bash
python scripts/run_stats.py
```

Reads cached predictions from steps 4, 7, 8 and produces:

- **Bootstrap 95% CIs** (1000 resamples) on macro-F1 for every Table I row
- **McNemar tests** between the trained head and each zero-shot regime
- **Worst-case LLaVA** macro-F1 (unparseable responses counted as errors)

Outputs are written to `outputs/stats/{headline_ci.json, mcnemar.json, llava_worstcase.json, latex_snippet.tex}`.

---

## Released artefacts

These are the files that constitute the paper's reproducibility contribution. All are in this repository:

| Artefact | Location | What it is |
|---|---|---|
| Caption rule set | `src/geoconstruct/data/captions.py` | The regex patterns + majority-vote logic that turn LEVIR-CC captions into binary labels. |
| Per-pair weak labels | `outputs/labels/{train,val,test}_labels.json` | The labels themselves: a dictionary `{pair_id: 0 or 1}`. |
| Coverage report | `outputs/labels/label_stats.json` | Per-split class balance and drop rate. |
| Trained checkpoint (headline) | `outputs/runs/concat_seed42/best.pt` | The model that achieves Macro-F1 0.882 on the test split. |
| 3-seed ablation results | `outputs/runs/*_seed{42,123,999}/metrics_test.json` | Five fusion variants × three seeds. |
| Hand-validated subset + κ | `outputs/label_validation/summary_test.json` | The 44-pair hand-validation results behind the κ=0.72 claim. |
| Bootstrap CIs + McNemar | `outputs/stats/*.json` | The statistical numbers in Table I. |

---

## Citation

If you use any of these artefacts (code, weak labels, trained checkpoints, or the caption rules) in academic work, please cite:

```bibtex
@article{sajid2026weakclipvlm,
  author  = {Sajid, Salman},
  title   = {Weakly Supervised {CLIP} Outperforms a 7B Open-Source
             Vision--Language Model on Bi-Temporal Satellite Change Classification},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Under review. Preprint: arXiv:2606.XXXXX}
}
```

And please also cite the underlying dataset:

```bibtex
@article{liu2022levircc,
  author  = {Liu, Chenyang and Zhao, Rui and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  title   = {Remote Sensing Image Change Captioning with Dual-Branch Transformers:
             A New Method and a Large Scale Dataset},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {60},
  pages   = {1--20},
  year    = {2022},
  doi     = {10.1109/TGRS.2022.3218921}
}
```

---

## License

Code is released under the **MIT License** (see `LICENSE`).

The released weak labels (`outputs/labels/*.json`) are derivative work from the LEVIR-CC captions and are released under the **CC-BY-4.0** license, consistent with the source dataset.

---

## Acknowledgments

This work was carried out at SEECS, NUST, Islamabad. Thanks to the LEVIR-CC authors for releasing the captioned dataset that made this study possible, and to the Ollama and `open_clip` maintainers for the open-source infrastructure that enabled a CPU-only reproduction.

---

## Contact

For questions about the code or paper, please open a GitHub issue or email `salmansajidsattar@gmail.com`.

For reviewing the paper itself, the latest PDF is in `paper/grsl_letter.pdf` and the source in `paper/grsl_letter.tex`.
