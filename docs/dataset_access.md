# Dataset Access and Local Paths

All datasets that can be pre-staged are available at `/scratch/ccn_datasets`.
Status column reflects current state on this cluster.

---

## A. Benchmark evaluation datasets

### A1. Winoground  — **ACTION REQUIRED**
- **Status:** Not yet downloaded. Gated HuggingFace dataset.
- **Access:** Accept terms at https://huggingface.co/datasets/facebook/winoground
  (free HF account). Generate a token at https://huggingface.co/settings/tokens.
  ```bash
  export HF_TOKEN=hf_...
  python scripts/prepare_eval_data.py winoground \
      --output-dir data/eval/winoground
  ```
- **Local path (after download):** `data/eval/winoground/` (gitignored).
- **Required for:** H1.

### A2. SugarCrepe  — **COCO images available; metadata download needed**
- **Status:** MS-COCO val2017 images are at `/scratch/ccn_datasets/ms_coco/val2017/`
  (5000 images). The SugarCrepe HF metadata still needs to be downloaded.
  ```bash
  python scripts/prepare_eval_data.py sugarcrepe \
      --coco-val-dir /scratch/ccn_datasets/ms_coco/val2017 \
      --output-dir data/eval/sugarcrepe
  ```
  No HF token needed (public repo: `HuggingFaceM4/SugarCrepe`).
- **Local path (after download):** manifest in `data/eval/sugarcrepe/`;
  `image_root` = `/scratch/ccn_datasets/ms_coco/val2017`.
- **Required for:** H1.

### A3. ARO (Attribution, Relation, Order)  — **ACTION REQUIRED**
- **Status:** Not downloaded.
- **Access:** Clone https://github.com/mertyg/vision-language-models-are-bows
  and follow the data prep instructions (VG relation/attribute JSON + Flickr30k/COCO images).
- **Set env var:** `export ARO_ROOT=/path/to/aro`
- **Required for:** H1.

---

## B. Neural alignment datasets

### B1. NSD (Natural Scenes Dataset)  — **AVAILABLE**
- **Status:** Fully available at `/scratch/ccn_datasets/natural-scenes-dataset`.
- **All 8 subjects present.** Betas (HDF5 + NIfTI), ROI masks, stimuli, experiment metadata.
- **Key paths** (also in `analysis/paths.py`):

  | Resource | Path |
  |----------|------|
  | Beta HDF5 | `…/nsddata_betas/ppdata/subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_sessionNN.hdf5` |
  | pRF ROI masks | `…/nsddata/ppdata/subjXX/func1pt8mm/roi/lh.prf-visualrois.nii.gz` |
  | floc ROI masks | `…/nsddata/ppdata/subjXX/func1pt8mm/roi/lh.floc-{faces,places,bodies}.nii.gz` |
  | Stim info CSV | `…/nsddata/experiments/nsd/nsd_stim_info_merged.csv` |
  | Stimuli HDF5 | `…/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` |

- **Loaders:** `analysis/nsd_loader.py` — `load_early_visual_mask()`,
  `load_late_ventral_mask()`, `load_betas_for_subject()`, `load_repeats_for_subject()`.
- **Required for:** H2, H4.

### B2. THINGS-EEG2 *(amended from THINGS-MEG — see Amendment 01)*  — **AVAILABLE**
- **Status:** Available at `/scratch/ccn_datasets/things_eeg_2`.
- **10 subjects.** Test: (200 images, 80 repeats, 17 channels, 100 times @ 10 ms/bin,
  −200 ms to +790 ms). Train: (16540 images, 4 repeats, …).
- **Loaders:** `analysis/things_eeg2_loader.py` — `eeg_for_temporal_generalization()`.
- **Amendment:** Pre-registration H3 originally specified THINGS-MEG.
  See `docs/preregistration_amendment_01.md` for full justification.
- **Required for:** H3.

### B3. MS-COCO (training images for NSD cross-reference)  — **AVAILABLE**
- **Path:** `/scratch/ccn_datasets/ms_coco/val2017` (5000 images).
- `nsd_stim_info_merged.csv` links NSD stimuli to COCO IDs.

---

## C. Training data

### C1. LLaVA-665k instruction-tuning corpus  — **ACTION REQUIRED**
- **Status:** Not downloaded.
- **Access:** Follow the LLaVA-v1.5 data preparation script at
  https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md
- Pass paths via `--data-json` and `--image-root` flags to `scripts/train.py`.
- COCO train2017 images (required by LLaVA) are not yet confirmed on this cluster.

---

## D. Status summary

| Dataset | Scratch location | Status | Action needed |
|---------|-----------------|--------|---------------|
| NSD | `/scratch/ccn_datasets/natural-scenes-dataset` | ✅ Available | None |
| THINGS-EEG2 | `/scratch/ccn_datasets/things_eeg_2` | ✅ Available | None |
| MS-COCO val2017 | `/scratch/ccn_datasets/ms_coco/val2017` | ✅ Available | None |
| Winoground | — | ❌ Gated HF | Accept terms + `prepare_eval_data.py winoground` |
| SugarCrepe metadata | — | ⚠️ Partial | `prepare_eval_data.py sugarcrepe` (COCO images present) |
| ARO | — | ❌ Not downloaded | Clone ARO repo + download VG/Flickr data |
| LLaVA-665k | — | ❌ Not downloaded | Follow LLaVA data prep |

---

## E. Environment variables

```bash
# Override default scratch paths if needed (defaults already point to scratch):
export NSD_ROOT=/scratch/ccn_datasets/natural-scenes-dataset
export THINGS_EEG2_ROOT=/scratch/ccn_datasets/things_eeg_2

# Required for gated HF datasets:
export HF_TOKEN=hf_...
```

All path constants are centralized in `analysis/paths.py`.
