# Kaggle Revision Playbook

This folder contains the eight new experiments needed for the IEEE GRSL
revision, plus one setup script and this README.

Each experiment is a standalone `.py` file that:
- reads only what it needs
- writes outputs to `outputs/runs/<experiment_name>/`
- finishes within a single 8-hour Kaggle session
- is idempotent — safe to re-run

## Kaggle constraints (as of 2026-08)

- **GPU**: T4 (16 GB VRAM) or P100 (16 GB VRAM). Enable in notebook settings.
- **Session length**: 8 hours continuous, then kernel dies.
- **Output disk**: 19 GB in `/kaggle/working/` — this is our working folder.
- **Free tier**: 30 GPU-hours per week (auto-refreshed).

The full Tier C revision needs **~15-18 GPU-hours** in total. Comfortably within one week's free-tier budget.

---

## Setup (one-time per session)

Every notebook session should start with:

```bash
!git clone https://github.com/<your-user>/geoconstruct-vl.git /kaggle/working/GeoConstruct-R1
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/setup_kaggle.py
```

The setup script:
- verifies GPU is enabled
- installs missing packages (`transformers`, `accelerate`, `bitsandbytes`, `open_clip_torch`)
- checks that the LEVIR-CC images, weak labels, and cached CLIP features are present
- prints a GO/NO-GO summary

If it prints `>>> Environment is READY`, you're good to go.

If features are missing, either:
1. Upload `outputs/features/*.npz` as a Kaggle dataset and mount it, OR
2. Re-run `python scripts/extract_clip_features.py` (needs GPU, ~20 min).

---

## Recommended order (matches reviewer priority)

Run the experiments in this order. Each addresses a specific reviewer concern.

### Session 1 — Priority 1 experiments (~4-5 hours)

Addresses the two most-cited reviewer concerns: RS-specific CLIP variant, and asymmetric comparison.

| # | Script | GPU-hours | Reviewer concern addressed |
|---|---|---|---|
| 1 | `eval_remoteclip.py` | 0.3 | R1#1: RS-specific CLIP variant (RemoteCLIP) |
| 2 | `eval_llava_fewshot.py` | 3-4 | R1#2: few-shot LLaVA closes the asymmetric-comparison gap |
| 3 | `eval_llava_single_image.py` | 1-2 | R1#8: temporal reasoning vs. domain mismatch diagnostic |

```bash
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_remoteclip.py
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_llava_fewshot.py
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_llava_single_image.py
```

After session 1, you have RemoteCLIP + few-shot LLaVA + single-image diagnostic results. **This alone addresses about 50% of the reviewer criticism.**

### Session 2 — Priority 2 experiments (~5-6 hours)

Addresses linear-probe question, threshold sensitivity, and dropped-pair bias.

| # | Script | GPU-hours | Reviewer concern addressed |
|---|---|---|---|
| 4a | `extract_llava_features.py` | 2-3 | R1#2 (deeper): extract LLaVA visual features |
| 4b | `train_linear_probe_llava.py` | 0.1 (CPU) | R1#2 (deeper): logistic regression on LLaVA features |
| 5 | `run_threshold_sensitivity.py` | 0.5 (CPU) | R1#5: 2/5, 3/5, 4/5 majority-vote sensitivity |
| 6 | `eval_dropped_pairs.py` + reruns | 1-2 | R1#4: are dropped pairs harder? |

```bash
# Extract LLaVA vision features (long-running, GPU)
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/extract_llava_features.py

# Train probes (fast, CPU)
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/train_linear_probe_llava.py

# Threshold sensitivity — takes ~30 min because it retrains the head 3 times
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/run_threshold_sensitivity.py

# Prepare dropped subset, then run existing evaluators on it
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/eval_dropped_pairs.py
# (this prints two follow-up commands to run LLaVA and CLIP zero-shot on the dropped subset)
```

### Session 3 — Priority 3 figures (~30 minutes, CPU-only)

These are visualization scripts that make the paper look much better.

| # | Script | Time | Reviewer concern addressed |
|---|---|---|---|
| 7 | `make_tsne_figure.py` | ~15 min CPU | R1#6: t-SNE showing linear separability |
| 8 | `make_qualitative_figure.py` | ~1 min CPU | R1#6: qualitative examples grid |

```bash
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/make_tsne_figure.py
!cd /kaggle/working/GeoConstruct-R1 && python scripts/kaggle/make_qualitative_figure.py
```

---

## What each experiment adds to the paper

After Session 1:
- **Table I row: RemoteCLIP zero-shot (diff)** → answers "is your gain from weak labels or from the specific encoder?"
- **Table V rows: 4-shot LLaVA, 8-shot LLaVA** → answers "did you give LLaVA a fair chance?"
- **New paragraph in Discussion + one figure**: LLaVA on single images to isolate the failure mode

After Session 2:
- **Table I row: linear-probe on LLaVA vision features** → "is the bottleneck vision or language?"
- **New Table: 2/5, 3/5, 4/5 majority-vote sensitivity** → "how brittle is the labeling threshold?"
- **New paragraph on dropped-pair analysis** → "are the pairs you drop harder than the pairs you keep?"

After Session 3:
- **New Figure 3**: three-panel t-SNE showing linear separability
- **New Figure 4**: six-panel qualitative examples

The paper goes from 4 pages / 2 figures / 5 tables (current)
to 4 pages / 4 figures / 6-7 tables (revised).

---

## Common pitfalls and fixes

### Kaggle session dies mid-run

The scripts write intermediate outputs (features, metrics, predictions) as they go. On re-run:
- **eval_remoteclip.py**: safe to re-run, will overwrite (fast, ~15 min).
- **eval_llava_fewshot.py**: re-run with `--shots 4 8` if 0-shot is done.
- **extract_llava_features.py**: will overwrite; run per split (`--splits test`).

### Out of VRAM on LLaVA

Kaggle T4 has 16 GB VRAM. LLaVA-7B in 4-bit uses about 10 GB. If you OOM:
- Reduce `--batch-size` to 4 or lower
- Clear notebook state between long runs (Kernel → Restart)

### Persisting outputs across sessions

Kaggle wipes `/kaggle/working/` when the session ends. To keep results:
1. In your notebook, at the end, run: `!zip -r results.zip outputs/runs outputs/figures outputs/features_llava`
2. Download `results.zip` before session end
3. Alternatively, save intermediate outputs as Kaggle **datasets** (persistent)

### Regenerating the LaTeX paper with new results

All new metric files are in `outputs/runs/<name>/*.json` — same schema as the existing runs. The paper's table-building logic in `scripts/run_stats.py` picks them up automatically. Just add the new rows to `paper/grsl_letter.tex`.

---

## Expected outcomes and what they mean

For each experiment, here's what a REALISTIC set of results looks like and how to interpret each:

| Experiment | Best-case outcome | Neutral outcome | Worst-case outcome |
|---|---|---|---|
| RemoteCLIP zero-shot (diff) | 0.60-0.65 (competitive) | 0.55-0.60 (slightly better than OpenAI CLIP) | 0.65+ (much better than OpenAI CLIP — dilutes your story) |
| Few-shot LLaVA (4-shot) | 0.55-0.60 (small improvement) | 0.50-0.55 (chance-level) | 0.65+ (closes the gap — weakens your story) |
| Single-image LLaVA (I2 only) | 0.55-0.60 (low; supports domain-mismatch) | 0.65-0.70 (medium; both hypotheses partly true) | 0.80+ (LLaVA good on single images; temporal bottleneck confirmed) |
| Linear-probe on LLaVA features | 0.87-0.89 (matches CLIP+MLP) | 0.80-0.85 (weaker but respectable) | 0.90+ (LLaVA visual encoder is superior) |
| Threshold sensitivity | 3/5 is best | 2/5, 3/5, 4/5 within 3 pts | Very different curves — rule is brittle |
| Dropped-pair analysis | LLaVA does WORSE on dropped pairs (not systematic bias) | Roughly equal | LLaVA does BETTER on dropped pairs — filtering IS biased |

Each of these outcomes has a way to frame it in the paper. Even "worst-case" results are publishable if honest.

---

## Response letter checklist

When you receive results, prepare a point-by-point response letter that references the new experiments:

- [ ] R1#1 → RemoteCLIP row in Table I
- [ ] R1#2 → few-shot LLaVA in Table V, linear-probe in Table I
- [ ] R1#3 → new column in Table I: GPU-hours per method
- [ ] R1#4 → new paragraph: dropped-pair analysis results
- [ ] R1#5 → new sub-table: threshold sensitivity
- [ ] R1#6 → new Figures 3 (t-SNE) and 4 (qualitative)
- [ ] R1#7 → trim redundant numbers from abstract
- [ ] R1#8 → single-image LLaVA diagnostic paragraph
- [ ] R1#9 → **polite decline** (SNN citations off-topic)
- [ ] R2#1 → justify LLaVA-7B choice in Section II
- [ ] R2#2 → include actual regex patterns in Section III
- [ ] R2#3 → include exact text templates for zero-shot CLIP
- [ ] R2#4 → report class distribution of the 44 validated pairs
- [ ] R2#5 → cite LoRA fine-tuning as future work (out of scope for CPU-friendly study)
- [ ] R2#6 → discussion paragraph: why caption-then-compare is not equivalent

---

## Contact

If a script fails on Kaggle in a way that isn't covered here, the fastest debug loop is:
1. Print the full stack trace
2. Reduce `--n-per-class` or `--batch-size` to run faster during debugging
3. Check the log file: `outputs/runs/<experiment>/eval.log`

Good luck with the revision. The paper has real substance and the reviewer feedback is genuinely useful — this revision will produce a much stronger paper regardless of the final venue.
