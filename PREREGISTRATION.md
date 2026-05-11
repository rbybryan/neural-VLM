# Pre-registration — Language-Conditioned Recurrent VLM

**Authors.** [Byrong et al.]
**Date locked.** 2026-05-11
**Repo lock.** git tag `v0-preregistration` in this repository; the tagged
commit fixes the analysis code, configs, and this document. For an
external lock (OSF / Zenodo), push this repo and reference the public
commit hash at the tag.

This document locks the analysis plan for the empirical claims in the paper
*before any Stage-2 model is trained*. Any deviation from the plan below will
be flagged explicitly as exploratory in the manuscript.

The corresponding spec is `SPEC_v1.md`. The corresponding analysis code is
`analysis/`.

---

## 1. Models trained for confirmatory analysis

| ID | Description | Source |
|---|---|---|
| **M (main)** | ConvNeXt-V2-Base + Qwen2.5-VL-7B + top-down adapters at stages 3,4 + 32 persistent context tokens, K=2, LoRA r=16 on LLM, full-tune CNN stages 3-4 | `configs/main.yaml` |
| **C1** | FLOP-matched feedforward (no recurrence, deeper CNN) | `configs/controls.yaml :: C1` |
| **C2** | Recurrence-without-LLM (lateral CNN KV) | `configs/controls.yaml :: C2` |
| **C3** | LLM-without-recurrence (K=1) | `configs/controls.yaml :: C3` |
| **C4** | Random-LLM-keys (frozen random projection) | `configs/controls.yaml :: C4` |

Eval-time controls (no separate training): C5 shuffled-caption, C6
iteration-scrambling, C7 memory-lesion, C8 K-sweep, C9 ensembling baseline.

All models use the *identical* training data, optimizer, schedule, and seed
budget. Differences are limited to the architectural toggles in
`RecurrentVLMConfig`.

---

## 2. Confirmatory hypotheses

### H1 — Compositionality specificity
On Winoground (group score), SugarCrepe (overall accuracy), ARO (relation +
attribute), the main model M outperforms:
  - **C1** (FLOP-matched feedforward) at p < 0.01, paired bootstrap over
    items, n_bootstrap = 10,000.
  - **C2** (recurrence without LLM) at p < 0.05.
  - **C3** (LLM without recurrence) at p < 0.05.
  - **C9** (ensembling baseline) at p < 0.05.

A failure of any of these contrasts disconfirms H1.

### H2 — Per-iteration variance dissociation on NSD
Using `analysis/variance_partition.py` with COCO captions as language input:

> The ROI interaction contrast
>   (unique_iter2 − unique_iter1)_late_ventral
>     − (unique_iter2 − unique_iter1)_early_visual
> is positive at p < 0.01 (paired bootstrap over voxels, 10,000 resamples).

ROIs:
  - **Early visual** = V1 + V2 + V3 (NSD pRF-defined).
  - **Late ventral** = LOC + anterior IT + FFA + PPA (NSD floc-defined).

The same contrast computed on **C6** (iteration-scrambled) must NOT be
positive at p < 0.05. If it is, H2 is disconfirmed (the iterations did not
encode distinct information).

### H3 — Temporal generalization on THINGS-MEG
Using `analysis/temporal_generalization.py`:

> (iter2 − iter1) decoding accuracy averaged over 150–300 ms is greater than
> over 80–150 ms, at p < 0.01 (paired bootstrap over time bins, 10,000
> resamples).

Time windows are pre-defined and not re-tuned post-hoc. The same contrast on
the iteration-scrambled control (C6) must NOT be positive at p < 0.05.

### H4 — Caption specificity
Using shuffled-caption control (C5) at NSD eval time:

> The drop in NSD encoding R^2 from correct → shuffled captions is
> significantly larger in the Fedorenko language-network mask than in V1–V3,
> at p < 0.01 (paired bootstrap over voxels, 10,000 resamples).

This tests that the model is using language *content* rather than any
high-dimensional signal correlated with images.

### H5 — RT specificity
Using `analysis/rt_correlation.py`:

> The image-averaged, feedforward-difficulty-partialed, noise-ceiling-
> normalized Spearman correlation between iterations-to-converge and human
> RT is positive (r > 0.2) for the main model M, AND is significantly larger
> for M than for both C2 and C3 at p < 0.05 (image-bootstrap, 10,000 resamples).

The "iterations-to-converge" metric is pre-specified as
`KL(softmax(logits_iter1) || softmax(logits_iter2))`, computed at the answer
token position. Feedforward difficulty is pre-specified as the entropy of the
ConvNeXt-V2 ImageNet linear head on the same image.

Realistic effect-size target: image-averaged partial r ≈ 0.3–0.4. We do NOT
claim r > 0.5.

---

## 3. Multiple-comparisons control

The five primary hypotheses H1–H5 are corrected with Holm–Bonferroni at
family-wise α = 0.05. A hypothesis is considered confirmatory only if it
survives correction.

Within H1, the four sub-contrasts (vs C1, C2, C3, C9) are corrected with
Bonferroni at α = 0.05 / 4 = 0.0125 within the family.

Within H2 and H3, the ROI / time-window contrasts are pre-specified and
single-test.

---

## 4. Exclusions and stopping rules

- **Eval-set decontamination.** Before any benchmark number is reported,
  training data is filtered against eval images via perceptual-hash near-
  duplicate detection at threshold ≤ 4 bits. Items with matches in training
  data are excluded from eval scores. The decontamination procedure and
  thresholds are locked here.
- **Subject exclusion (NSD).** All 8 subjects are used. No subject-level
  exclusion based on model fit.
- **Trial exclusion (RT).** RT outliers are removed per-subject at ±3 SD on
  log RT. Locked.
- **Training stopping.** Stage-2 training stops at one full epoch over the
  data mix in `SPEC_v1.md §3`. No early-stop on eval signal.

---

## 5. Power and feasibility

- Winoground (n=400 image pairs): paired-bootstrap detects effects ≥ 3% at
  α=0.01 with >0.8 power.
- NSD voxel-level paired bootstrap with ~10k voxels per ROI: detects
  per-voxel mean differences ≥ 0.005 R^2 with negligible variance.
- THINGS-MEG (n=22k+ images, 4 subjects): time-bin bootstrap detects
  iteration × window interactions of ≥ 0.01 decoding R^2 at α=0.01.
- RT correlation: with n ≥ 1000 images and split-half ceiling typically
  0.7–0.85, partial r ≥ 0.25 is detectable at α=0.05.

If any benchmark falls below the power threshold above, the corresponding
hypothesis is reported as exploratory.

---

## 6. Deviation policy

Any analysis NOT specified in §2 is exploratory and labeled as such in the
manuscript. If a confirmatory test fails, we report the failure; we do not
substitute a related test post-hoc and call it confirmatory.

The frozen analysis code, config files, and this document are committed to
the repository at the hash recorded in §0 above. All confirmatory analyses
are run from that frozen code.
