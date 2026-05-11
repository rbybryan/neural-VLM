# Pre-registration Amendment 01 — H3 Dataset: THINGS-MEG → THINGS-EEG2

**Date of amendment:** 2026-05-11
**Reason:** Infrastructure discovery before any Stage-2 model is trained.
**Scope:** H3 only (temporal generalization). H1, H2, H4, H5 are unaffected.

---

## What the pre-registration said

H3 specified THINGS-MEG (Gifford, Cichy et al. 2022; 4 subjects, ~22k images,
MEG sensor data) as the neural dataset for the temporal generalization analysis
(King & Dehaene 2014).

## What is available on this cluster

`/scratch/ccn_datasets/things_eeg_2` contains **THINGS-EEG2** (Gifford, Cichy
et al. 2023; 10 subjects, 16540 training + 200 test images, EEG).

Preprocessed test-set shape: (200 images, 80 repeats, 17 channels, 100 timepoints).
Temporal resolution: 10 ms/bin. Coverage: −200 ms to +790 ms relative to stimulus onset.

THINGS-MEG is not available on this cluster.

## Decision

H3 is amended to use **THINGS-EEG2** in place of THINGS-MEG.

### Why this is scientifically valid

1. **Same stimuli.** Both datasets use images from the THINGS object database.
2. **Adequate temporal resolution.** 10 ms/bin covers the pre-registered
   windows (80–150 ms, 150–300 ms) with 7 and 15 bins respectively.
3. **Temporal generalization is standard for EEG.** King & Dehaene 2014
   introduced TDGA on MEG/EEG interchangeably; its application to EEG is
   common (e.g., Gifford et al. 2023 applied it to THINGS-EEG2 themselves).
4. **Better statistical power.** 10 subjects vs. 4; 80 repeats for test images
   vs. ~12 in THINGS-MEG, yielding higher SNR per image.

### What changes

| Item | Pre-registration | Amendment |
|------|-----------------|-----------|
| Dataset | THINGS-MEG | THINGS-EEG2 |
| Modality | MEG | EEG |
| Subjects | 4 | 10 |
| Sensor count | ~300 | 17 (preprocessed) |
| Temporal res. | ~1 ms | 10 ms |
| Data path | (not available) | `/scratch/ccn_datasets/things_eeg_2` |
| Loader | — | `analysis/things_eeg2_loader.py` |

### What does NOT change

- Pre-registered time windows: 80–150 ms (early), 150–300 ms (late). Unchanged.
- Statistical test: paired bootstrap over time bins, n=10,000, α=0.01. Unchanged.
- Iteration-scrambling control (C6) requirement. Unchanged.
- Analysis code: `analysis/temporal_generalization.py`. Unchanged (dataset-agnostic).
- H3 hypothesis direction and success criterion. Unchanged.

## Manuscript language

The manuscript will state:

> "We originally pre-registered THINGS-MEG as the neural dataset for H3.
> Prior to training any Stage-2 model, we discovered that THINGS-EEG2
> (Gifford et al. 2023) — which uses the same THINGS stimuli, covers the
> same temporal windows, and offers superior statistical power (10 subjects,
> 80 repeats per test image) — was available on our computing cluster while
> THINGS-MEG was not. We therefore amended H3 to use THINGS-EEG2. The
> hypothesis, time windows, statistical test, and control requirements are
> unchanged. This amendment predates any model training and is logged in
> `docs/preregistration_amendment_01.md`."

## Amendment authority

This amendment is made before any Stage-2 model is trained. Per §6 of
PREREGISTRATION.md, deviations from the locked plan must be flagged
explicitly. This document serves as that flag. The amendment is committed to
the repository after the `v0-preregistration` tag so that the pre-registered
and amended versions are both versioned in git.
