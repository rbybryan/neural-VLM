# Dataset Access Requirements

This document lists every dataset needed to reproduce the confirmatory analyses
(H1–H5) in `PREREGISTRATION.md`, the access mechanism for each, and the
expected local path or environment variable.

---

## A. Benchmark evaluation datasets

### A1. Winoground
- **What**: 400 image-caption pairs testing compositional understanding.
- **Access**: Gated on Hugging Face. Accept the terms of use at
  https://huggingface.co/datasets/facebook/winoground (requires a free HF
  account). After approval generate an API token at
  https://huggingface.co/settings/tokens and export it:
  ```bash
  export HF_TOKEN=hf_...
  ```
- **Download**:
  ```bash
  python scripts/prepare_eval_data.py winoground \
      --output-dir data/eval/winoground
  ```
- **Local path**: `data/eval/winoground/` (gitignored; created on demand).

### A2. SugarCrepe
- **What**: 7-subset compositional benchmark built on MS-COCO val2017 images.
- **Access (SugarCrepe metadata)**: Public on Hugging Face
  (`HuggingFaceM4/SugarCrepe`), no token required.
- **Access (MS-COCO val2017 images)**: Open license (CC BY 4.0). Download
  once (~1 GB):
  ```bash
  wget http://images.cocodataset.org/zips/val2017.zip
  unzip val2017.zip -d /path/to/coco/
  ```
- **Download**:
  ```bash
  python scripts/prepare_eval_data.py sugarcrepe \
      --coco-val-dir /path/to/coco/val2017 \
      --output-dir data/eval/sugarcrepe
  ```
- **Local path**: manifest in `data/eval/sugarcrepe/`; image_root is the
  shared COCO val2017 directory.

### A3. ARO (Attribution, Relation, Order)
- **What**: VG-Relation, VG-Attribution, COCO-Order, Flickr-Order subsets.
- **Access**: Public. Repository:
  https://github.com/mertyg/vision-language-models-are-bows
- **Download**: follow the ARO repo README to download VG relation/attribute
  JSON and Flickr30k / COCO images.
- **Local path**: `data/eval/aro/` (gitignored).

---

## B. Neural alignment datasets

### B1. NSD (Natural Scenes Dataset)
- **What**: 8-subject 7T fMRI responses to ~73k MS-COCO images. Required for
  H2 (per-iteration variance partitioning) and H4 (caption specificity).
- **Access**: Data sharing agreement required. Register at
  https://naturalscenesdataset.org and sign the DUA. Download via the AWS S3
  bucket described in the NSD data manual.
- **Minimum files needed**:
  - `nsddata/ppdata/subj*/func*/betas_fithrf_GLMdenoise_RR/` — beta volumes
  - `nsddata/experiments/nsd/nsd_stim_info_merged.csv` — image metadata
  - `nsddata/freesurfer/subj*/label/` — pRF + floc ROI labels
  - `nsddata/nsddata_betas/ppdata/subj*/nativesurface/betas_fithrf_GLMdenoise_RR/` — surface betas
- **ROI masks** (used by `analysis/variance_partition.py`):
  - Early visual: V1, V2, V3 (pRF-defined, NSD release)
  - Late ventral: LOC, anterior IT, FFA, PPA (floc-defined, NSD release)
- **Local path**: Set `NSD_ROOT` environment variable to the NSD download root.

### B2. THINGS-MEG
- **What**: MEG recordings from 4 subjects viewing ~22k images from the THINGS
  database. Required for H3 (temporal generalization).
- **Access**: Open access. OSF repository:
  https://osf.io/3jk45/ (Gifford et al. 2022; Hebart et al. 2023).
  Download via `osfclient`:
  ```bash
  pip install osfclient
  osf -p 3jk45 clone things_meg/
  ```
- **Citation**: Gifford et al. (2022), "EEG-ImageNet" / Hebart et al. (2023),
  "THINGS-MEG". Cite both when using.
- **Local path**: Set `THINGS_MEG_ROOT` environment variable.

### B3. THINGS-fMRI (optional)
- **What**: fMRI responses to THINGS images (Theven et al. 2023). May
  supplement NSD for additional ventral-stream ROI coverage.
- **Access**: Open access via OpenNeuro: https://openneuro.org/datasets/ds004192
- **Local path**: `data/neural/things_fmri/` (gitignored).

---

## C. Training data

### C1. LLaVA-665k instruction-tuning corpus
- **What**: ~665k multimodal instruction-following examples.
- **Access**: Public on Hugging Face (`liuhaotian/LLaVA-Instruct-150K` for the
  subset; full 665k details at the LLaVA-v1.5 release page).
- **Images**: COCO train2017 + GQA + OCR-VQA + TextVQA + VisualGenome (all
  publicly licensed; follow the LLaVA data preparation script for download).
- **Local path**: `--data-json` and `--image-root` flags to `scripts/train.py`.

---

## D. Access checklist

| Dataset | Status | Priority | Blocker if missing |
|---------|--------|----------|--------------------|
| Winoground | Need HF token | High (H1) | H1 cannot run |
| MS-COCO val2017 | Open download | High (H1 via SugarCrepe) | SugarCrepe cannot run |
| ARO | Open download | High (H1) | H1 ARO sub-score |
| NSD | DUA required | High (H2, H4) | H2 and H4 cannot run |
| THINGS-MEG | Open (OSF) | High (H3) | H3 cannot run |
| THINGS-fMRI | Open (OpenNeuro) | Low (exploratory) | Exploratory only |
| LLaVA-665k | Open | High (training) | No training |
| COCO train2017 | Open | High (training) | No LLaVA images |

---

## E. Environment variables summary

```bash
export HF_TOKEN=hf_...          # Winoground download
export NSD_ROOT=/path/to/nsd    # analysis/variance_partition.py
export THINGS_MEG_ROOT=/path/to/things_meg  # analysis/temporal_generalization.py
```

Add these to your shell profile or a `.env` file (which is gitignored).
