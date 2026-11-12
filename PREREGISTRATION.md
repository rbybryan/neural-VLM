# Research hypotheses and analysis plan

**Author:** Boyan Rong

**Original plan date:** 2026-05-11

**Original snapshot:** git tag `v0-preregistration`

This records the intended comparisons for the recurrent VLM project. The tag preserves the original in-repository plan; no external registration is linked here. These are hypotheses, not completed results.

The original model description below is retained as part of that plan. The current default is a text-only Qwen2.5-1.5B model paired with ConvNeXt-V2, and the control implementations still have gaps. See the [design notes](SPEC_v1.md) and [current status](docs/RESEARCH_STATUS.md) before using the plan as a run specification.

H3 uses THINGS-EEG2 under [Amendment 01](docs/preregistration_amendment_01.md). [Amendment 02](docs/preregistration_amendment_02.md) records the documentation corrections and withdrawal of unsupported power estimates. H1–H5 below retain their original directions and thresholds; their statistical procedures still need validation before confirmatory use.

---

## 1. Models proposed for confirmatory analysis

| ID | Description | Source |
|---|---|---|
| **M (main)** | ConvNeXt-V2-Base + Qwen2.5-VL-7B + top-down adapters at stages 3,4 + 32 persistent context tokens, K=2, LoRA r=16 on LLM, full-tune CNN stages 3-4 | `configs/main.yaml` |
| **C1** | FLOP-matched feedforward (no recurrence, deeper CNN) | `configs/controls.yaml :: C1` |
| **C2** | Recurrence-without-LLM (lateral CNN KV) | `configs/controls.yaml :: C2` |
| **C3** | LLM-without-recurrence (K=1) | `configs/controls.yaml :: C3` |
| **C4** | Random-LLM-keys (frozen random projection) | `configs/controls.yaml :: C4` |

Eval-time controls (no separate training): C5 shuffled-caption, C6
iteration-scrambling, C7 memory-lesion, C8 K-sweep, C9 ensembling baseline.

The planned comparison uses the same training data, optimizer, schedule, and seed
budget across models. This has not yet been demonstrated in real-data runs.

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

No reproducible power calculation or simulation is included in this repository.
The numerical power assurances in the original plan are withdrawn from this
version, as recorded in [Amendment 02](docs/preregistration_amendment_02.md).

Before confirmatory analysis, the sampling units, dependence structure, dataset
sizes, effect-size assumptions, and multiple-comparison procedure need review.
The desired effect sizes in the hypotheses are planning thresholds, not measured
performance or demonstrated detection power.

---

## 6. Deviation policy

Any analysis NOT specified in §2 is exploratory and labeled as such in the
manuscript. If a confirmatory test fails, we report the failure; we do not
substitute a related test post-hoc and call it confirmatory.

The original code and plan are preserved at `v0-preregistration`. No completed
confirmatory analysis is reported here. Any future analysis must record the code
commit, data version, and amendments used; the current working tree should not be
assumed identical to the tagged plan.
