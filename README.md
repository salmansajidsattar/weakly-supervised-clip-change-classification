# Weakly Supervised CLIP Outperforms a 7B Open-Source Vision-Language Model on Bi-Temporal Satellite Change Classification: A Systematic Empirical Benchmark

Official code, weak labels, and trained checkpoints for the paper (under major revision):

> **Salman Sajid.** *Weakly Supervised CLIP Outperforms a 7B Open-Source Vision-Language Model on Bi-Temporal Satellite Change Classification: A Systematic Empirical Benchmark.* SPIE Journal of Applied Remote Sensing (JARS), manuscript JARS-260830-1, under review, 2026.
> Affiliation: School of Electrical Engineering and Computer Science (SEECS), National University of Sciences and Technology (NUST), Islamabad, Pakistan
> Contact: `salmansajidsattar@gmail.com`

---

## TL;DR

Can an open-source vision-language model (VLM) classify bi-temporal change on satellite imagery zero-shot, and does the answer depend on which VLM? This study benchmarks five supervision regimes on a shared 1849-pair LEVIR-CC test split, all evaluated under the same metric:

| Method | Trainable params | Macro-F1 [95% CI] |
|---|---|---|
| LLaVA-1.5-7B zero-shot | 0 | 0.501 [0.478, 0.526] |
| Zero-shot CLIP ViT-B/32 (temporal diff) | 0 | 0.588 [0.563, 0.611] |
| RemoteCLIP zero-shot (temporal diff) | 0 | 0.603 [0.582, 0.626] |
| Qwen2.5-VL-7B zero-shot | 0 | 0.807 [0.789, 0.824] |
| **Frozen CLIP + concat head (ours)** | ~139.6K | **0.882 [0.866, 0.896]** |

The trained head significantly outperforms Qwen2.5-VL-7B on the full weak-labelled test set (p = 5.5x10^-13) and LLaVA-1.5-7B (p < 10^-99). RemoteCLIP is not significantly different from generic CLIP (p = 0.27) — the gain over zero-shot CLIP comes from weak supervision, not a better-adapted encoder. A five-variant fusion ablation shows simple concatenation is the best head design; a cross-attention variant with ~962K params (nearly 7x more) does not help.

**Labels are caption-derived, not ground truth.** A 183-pair hand-validated subset (79.8% agreement with the weak labels, kappa = 0.596) is used to re-score every method: the trained head drops to 0.794 [0.729, 0.850], Qwen2.5-VL-7B reaches 0.749 [0.682, 0.809] — a gap that is no longer statistically significant at this sample size (p = 0.22) — while LLaVA-1.5-7B remains at chance under both label sources (0.515 [0.440, 0.591] on hand labels).

**Read this as a generation-dependent result, not a fixed 7B-scale limitation**: an older instruction-tuned VLM (LLaVA-1.5-7B) fails outright at this task, a newer one (Qwen2.5-VL) approaches the cost of full supervision, and a ~140K-parameter caption-supervised classifier remains competitive with or ahead of both at a fraction of the inference cost. The supervised pipeline runs end-to-end on a single CPU.

---

## What's in this repository

This repo contains the reproducibility code only — the paper source, generated data, and large run artefacts are not tracked here (see `.gitignore`).

```
.
├── configs/                        # YAML configs, one per ablation variant
│   ├── default.yaml                # cross-attention head ("full" variant, used in the ablation table)
│   ├── ablation_concat.yaml        # headline model: frozen CLIP + concat MLP (~139.6K params)
│   ├── ablation_t1_only.yaml       # single-stream, image 1 only (~74K params)
│   ├── ablation_t2_only.yaml       # single-stream, image 2 only (~74K params)
│   └── ablation_diff.yaml          # temporal-difference fusion
├── scripts/                        # CPU-only runnable entry points
│   ├── download_levir_cc.py        # one-time dataset download
│   ├── build_weak_labels.py        # caption rule → binary labels
│   ├── extract_clip_features.py    # one-time CLIP feature cache
│   ├── train.py                    # train one config
│   ├── eval.py                     # evaluate one checkpoint
│   ├── test.py                     # test-split eval + dump fused features
│   ├── run_ablations.py            # 5 variants x 3 seeds
│   ├── run_classical_baselines.py  # LogReg, Linear SVM, kNN on cached features
│   ├── eval_clip_zero_shot.py      # zero-shot CLIP baseline (no training)
│   ├── eval_vlm_zero_shot.py       # zero-shot LLaVA-1.5-7B via Ollama
│   ├── eval_llava_prompts.py       # prompt-sensitivity study
│   ├── eval_on_hand_validated_subset.py  # re-score every method on the 183-pair hand-labelled subset
│   ├── validate_weak_labels.py     # interactive hand-validation tool
│   ├── count_head_params.py        # verifies the exact trainable-parameter counts quoted in the paper
│   ├── run_stats.py                # bootstrap CIs + McNemar tests
│   ├── make_figures.py             # paper figures from a run directory
│   └── kaggle/                     # GPU-dependent revision experiments (see scripts/kaggle/README_KAGGLE.md)
│       ├── eval_qwen2vl.py         # Qwen2.5-VL-7B zero-shot baseline
│       ├── eval_remoteclip.py      # RemoteCLIP zero-shot baseline
│       ├── run_threshold_sensitivity.py  # 2/5, 3/5, 4/5 caption-agreement threshold sweep
│       ├── make_qualitative_figure.py    # success/failure qualitative examples figure
│       ├── make_tsne_figure.py     # t-SNE separability figure
│       ├── eval_llava_fewshot.py, eval_llava_single_image.py, extract_llava_features.py,
│       │   train_linear_probe_llava.py, eval_dropped_pairs.py, setup_kaggle.py
│       └── README_KAGGLE.md        # Kaggle session playbook for the GPU-dependent experiments
├── src/geoconstruct/                # the library
│   ├── data/                        # caption rules, weak-label builder, cached dataset
│   ├── models/                      # frozen CLIP, temporal head (concat/diff/single-stream/cross-attn), retrieval explainer
│   ├── losses/                      # class-weighted cross entropy
│   ├── trainers/                    # CPU-friendly training loop
│   ├── evaluation/                  # macro-F1 + sklearn report
│   ├── visualization/                # matplotlib-only figure helpers
│   └── utils/                       # seed, logging, YAML config
├── requirements.txt
└── README.md                        # this file
```

Running the scripts locally will generate `data/`, `outputs/` (labels, cached features, checkpoints, stats, figures) — all gitignored, since they're either large, regenerable, or (for `data/`) third-party.

---

## Reproducing the headline numbers

The core pipeline (everything except the Qwen2.5-VL and RemoteCLIP baselines, which need a GPU — see `scripts/kaggle/`) runs end-to-end on a laptop CPU.

### 0. Environment

```bash
pip install -r requirements.txt
```

Key packages: `torch` (CPU build is sufficient), `open_clip_torch` (frozen CLIP ViT-B/32 backbone), `numpy`/`pandas`/`scikit-learn`/`matplotlib`/`seaborn`, `huggingface_hub`/`transformers`, `pyyaml`, `tqdm`.

For the LLaVA-1.5-7B baseline you additionally need [Ollama](https://ollama.com):

```bash
ollama pull llava:7b
ollama serve   # in a separate terminal
```

### 1. Dataset

```bash
python scripts/download_levir_cc.py
```

Pulls LEVIR-CC from Hugging Face into `data/LEVIR_CC/`. Dataset by Liu et al. (TGRS 2022); please cite their paper if you use it.

### 2. Build weak labels

```bash
python scripts/build_weak_labels.py
```

Applies the caption-rule majority vote (default: >=3 of 5 captions must agree; ambiguous pairs dropped). Writes per-split JSONs to `outputs/labels/`.

### 3. Extract frozen CLIP features

```bash
python scripts/extract_clip_features.py
```

Runs CLIP ViT-B/32 (OpenAI weights) once over every labelled pair; cached to `outputs/features/`. All downstream training reads from this cache.

### 4. Train the headline head

```bash
python scripts/train.py --config configs/ablation_concat.yaml
```

Trains the `[f1; f2]` concatenation MLP (~139.6K trainable params) on the weak labels. Best checkpoint saved to `outputs/runs/concat_seed42/best.pt`.

### 5. Fusion ablation (5 variants x 3 seeds)

```bash
python scripts/run_ablations.py
```

Reproduces the ablation table: single-stream (f1 or f2 only, ~74K params each), temporal-difference, concatenation, and cross-attention (~962K params).

### 6. Classical baselines

```bash
python scripts/run_classical_baselines.py
```

### 7. Zero-shot CLIP / RemoteCLIP baselines

```bash
python scripts/eval_clip_zero_shot.py --fuse diff
python scripts/kaggle/eval_remoteclip.py   # GPU
```

### 8. Zero-shot VLM baselines

```bash
# LLaVA-1.5-7B (CPU, via Ollama — ~5 hr)
python scripts/eval_vlm_zero_shot.py

# Qwen2.5-VL-7B (GPU, 4-bit NF4 — see scripts/kaggle/README_KAGGLE.md)
python scripts/kaggle/eval_qwen2vl.py
```

### 9. Hand-validation re-scoring

```bash
python scripts/validate_weak_labels.py
python scripts/eval_on_hand_validated_subset.py
```

Reproduces the 183-pair hand-validated re-scoring (79.8% agreement, kappa = 0.596) behind the paper's Limitations analysis.

### 10. Threshold sensitivity

```bash
python scripts/kaggle/run_threshold_sensitivity.py
```

Reproduces the 2/5, 3/5, 4/5 caption-agreement sweep (macro-F1 0.882 / 0.882 / 0.887).

### 11. Statistical tests

```bash
python scripts/run_stats.py
```

Bootstrap 95% CIs (1000 resamples) and McNemar tests for every reported comparison.

---

## Citation

```bibtex
@article{sajid2026weakclipvlm,
  author  = {Sajid, Salman},
  title   = {Weakly Supervised {CLIP} Outperforms a 7B Open-Source
             Vision--Language Model on Bi-Temporal Satellite Change
             Classification: A Systematic Empirical Benchmark},
  journal = {Journal of Applied Remote Sensing},
  year    = {2026},
  note    = {Manuscript JARS-260830-1, under review}
}
```

Please also cite the underlying dataset:

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

The released weak labels (generated locally to `outputs/labels/*.json`) are derivative work from the LEVIR-CC captions and are released under **CC-BY-4.0**, consistent with the source dataset.

---

## Acknowledgments

This work was carried out at SEECS, NUST, Islamabad. Thanks to the LEVIR-CC authors for releasing the captioned dataset that made this study possible, and to the Ollama, `open_clip`, and Hugging Face `transformers` maintainers for the open-source infrastructure.

---

## Contact

For questions about the code or paper, please open a GitHub issue or email `salmansajidsattar@gmail.com`.
