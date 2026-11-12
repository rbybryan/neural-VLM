# Data access checklist

> Historical planning checklist. Current portable paths and the H3 THINGS-EEG2 amendment are described in [dataset setup](dataset_access.md). Confirm current access conditions with each provider.

Datasets the H1–H5 pipeline depends on, ordered by access lead time and
criticality. Request gated access early — some applications take weeks.

## Required for the confirmatory hypotheses (PREREGISTRATION.md §2)

### Winoground — required for H1
- **Source**: https://huggingface.co/datasets/facebook/winoground
- **Access**: HuggingFace account + accept dataset terms on the page +
  create a read-scope access token
- **Size**: ~30 MB
- **Lead time**: instant after terms accepted
- **Loader**: `scripts/prepare_eval_data.py winoground --output-dir ...`
- **Notes**: 400 image pairs; group score is the H1 headline metric.

### SugarCrepe — required for H1 (compositionality detail)
- **Source captions**: https://huggingface.co/datasets/HuggingFaceM4/SugarCrepe
  (other mirrors: `xinli/SugarCrepe`, official SugarCrepe github)
- **Source images**: MS-COCO val2017 download
  (http://images.cocodataset.org/zips/val2017.zip, ~1 GB)
- **Access**: public on both
- **Lead time**: download time only
- **Loader**: `scripts/prepare_eval_data.py sugarcrepe --coco-val-dir ...`
- **Notes**: 7 subsets × ~1000 items each. Per-category accuracy is
  reported by the BinaryDiscrimination benchmark.

### ARO — optional second compositionality dataset for H1
- **Source captions**: https://github.com/mertyg/vision-language-models-are-bows
- **Source images**: Visual Genome
  (https://homes.cs.washington.edu/~ranjay/visualgenome/api.html, ~15 GB)
- **Access**: public
- **Lead time**: download time
- **Loader**: not yet built (same shape as SugarCrepe — small wiring job
  when needed)

### Natural Scenes Dataset (NSD) — required for H2 and H4
- **Source**: http://naturalscenesdataset.org/
- **Access**: requires Data Use Agreement application; reviewed by NSD
  team, usually <2 weeks turnaround
- **Size**: ~5 TB raw; ~50 GB for the preprocessed COCO-aligned subset
  used by BrainScore / Algonauts pipelines
- **Lead time**: 2–4 weeks for approval + download
- **Companion**: COCO captions for language conditioning (per SPEC §5.B.4,
  NSD subjects did continuous-recognition memory, no native captions;
  we condition the model on COCO captions for the corresponding images
  and explicitly disclose this in the methods).
- **Atlas needed**: NSD-provided floc / pRF ROIs for V1–V3, LOC, anterior
  IT (already bundled with the dataset release).

### THINGS-MEG — required for H3
- **Source**: https://openneuro.org/datasets/ds004212/
- **Access**: public via OpenNeuro
- **Size**: ~200 GB raw
- **Lead time**: download time
- **Notes**: 22 248 images, 4 subjects. Sized for the King–Dehaene
  temporal-generalization analysis (`analysis/temporal_generalization.py`).

### THINGS-fMRI — optional secondary for H2
- **Source**: https://openneuro.org/datasets/ds004192/
- **Access**: public via OpenNeuro
- **Size**: ~150 GB
- **Lead time**: download time

## Required for H5 RT specificity

There is no natively available human RT dataset on Winoground / SugarCrepe
stimuli. Two paths:

- **(a) Custom RT collection on Prolific / MTurk.** ~$200–500 for n ≥ 30
  subjects on a few hundred images. ~2 weeks turnaround. Recommended once
  Stage-2 model is trained and you know which images you want RT on.
- **(b) Use Geirhos model-vs-human benchmark** (https://github.com/bethgelab/model-vs-human)
  for an initial cut — it has 17 perturbation conditions with human
  accuracy and timing-like measurements on ImageNet stimuli. Not native
  to our benchmark suite but acceptable for a first H5 pass.

## Atlas / mask files needed for analyses

### Fedorenko language network mask — required for H4
- **Source**: https://evlab.mit.edu/funcloc/
- **Access**: public download (atlas + mask in MNI space)
- **Size**: tens of MB
- **Used by**: H4 shuffled-caption ROI specificity test in
  `analysis/variance_partition.py::roi_dissociation_test`.

### NSD floc / pRF ROI atlases — required for H2
- **Source**: bundled with NSD distribution
- **Used by**: H2 early-visual (V1–V3) vs late-ventral (LOC + anterior IT)
  variance contrast.

## Training data (already discussed in SPEC §3)

### ShareGPT4V-Pretrain — Stage-1 alignment
- **Source**: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V
- **Size**: 1.2M items; needs companion images
- **Access**: public on HF

### LLaVA-1.5-mix-665K — Stage-2 SFT
- **Source**: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
  (canonical) or `lmms-lab/LLaVA-NeXT-Data` (richer)
- **Size**: ~665k conversations
- **Access**: public on HF

### LLaVA-OneVision-SI / Cambrian-7M subsets — Stage-2 SFT diversity
- **Sources**:
  - https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data
  - https://huggingface.co/datasets/nyu-visionx/Cambrian-10M
- **Access**: public on HF

### COCO 2017 train — for training-data decontamination index
- **Source**: http://images.cocodataset.org/zips/train2017.zip
- **Size**: ~18 GB
- **Used by**: `data/decontaminate.py` builds a PHash index of all
  training-corpus images; any benchmark image within Hamming distance
  4 bits triggers exclusion.

## Action order (by lead time)

1. **Apply for NSD now** — 2–4 weeks. Single-page DUA at
   naturalscenesdataset.org/data/. Do this *first*; it's the slowest.
2. **Apply for Winoground access on HF + create token** — today.
3. **Download SugarCrepe captions** — today (public).
4. **Start MS-COCO val2017 download** — today (~1 GB).
5. **Start THINGS-MEG download** — in parallel with Stage-1 training (no
   blocking dependency).
6. **Plan RT data collection** — after Stage-2 model is trained and
   we know which images carry the iteration-KL signal worth measuring
   human RT on.

## Storage budget summary

| Item | Size |
|---|---|
| ShareGPT4V-Pretrain + images | ~30 GB |
| LLaVA-1.5-mix-665K + images | ~30 GB |
| LLaVA-OneVision subset | ~50 GB |
| Cambrian subset | ~50 GB |
| Winoground | 30 MB |
| SugarCrepe captions + COCO val2017 | ~1 GB |
| ARO + Visual Genome | ~15 GB |
| NSD preprocessed subset | ~50 GB |
| THINGS-MEG | ~200 GB |
| THINGS-fMRI | ~150 GB |
| Model checkpoints (trainable-only, ~10 controls × ~5 ckpts) | ~5 GB |
| **Total realistic working set** | **~600 GB** |

Storage must be provisioned on the machine used for the experiment.
