"""Configurable paths for all datasets used in H1-H5 analyses.

Override any path via environment variable (see each constant's comment).
"""

from __future__ import annotations

import os
from pathlib import Path

_SCRATCH = Path(os.environ.get("NEURAL_VLM_DATA_ROOT", "datasets")).expanduser()

# -------------------------------------------------------------------
# NSD  (H2, H4)
# -------------------------------------------------------------------
NSD_ROOT = Path(os.environ.get("NSD_ROOT", _SCRATCH / "natural-scenes-dataset"))

NSD_BETAS_ROOT = NSD_ROOT / "nsddata_betas" / "ppdata"   # subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/
NSD_DATA_ROOT  = NSD_ROOT / "nsddata"
NSD_PPDATA     = NSD_DATA_ROOT / "ppdata"                 # subj01/func1pt8mm/roi/
NSD_STIM_INFO  = NSD_DATA_ROOT / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
NSD_STIMULI_HDF5 = NSD_ROOT / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"

NSD_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]  # subj01 .. subj08
NSD_RES = "func1pt8mm"

# ROI file names (under ppdata/subjXX/func1pt8mm/roi/)
NSD_ROI_FILES = {
    "prf_visualrois": "lh.prf-visualrois.nii.gz",   # V1=1,V2=2,V3=3 per NSD convention
    "floc_faces":     "lh.floc-faces.nii.gz",        # FFA
    "floc_places":    "lh.floc-places.nii.gz",        # PPA
    "floc_bodies":    "lh.floc-bodies.nii.gz",        # EBA / LOC-body
    "floc_words":     "lh.floc-words.nii.gz",         # VWFA
}

# -------------------------------------------------------------------
# MS-COCO  (SugarCrepe eval, NSD stimuli cross-reference)
# -------------------------------------------------------------------
COCO_VAL2017 = Path(os.environ.get("COCO_VAL2017", _SCRATCH / "ms_coco" / "val2017"))
COCO_TRAIN2017 = Path(os.environ.get("COCO_TRAIN2017", _SCRATCH / "ms_coco" / "train2017"))

# -------------------------------------------------------------------
# THINGS-EEG2  (H3 — temporal generalization; EEG, 10 subjects)
# Pre-registration Amendment 01: H3 originally specified THINGS-MEG;
# The implementation follows the THINGS-EEG2 amendment.
# See docs/preregistration_amendment_01.md.
# -------------------------------------------------------------------
THINGS_EEG2_ROOT = Path(os.environ.get("THINGS_EEG2_ROOT",
                                        _SCRATCH / "things_eeg_2"))

THINGS_EEG2_PREPROCESSED = THINGS_EEG2_ROOT / "preprocessed_data"
THINGS_EEG2_IMAGES_TEST  = THINGS_EEG2_ROOT / "image_set" / "test_images"
THINGS_EEG2_IMAGES_TRAIN = THINGS_EEG2_ROOT / "image_set" / "training_images"
THINGS_EEG2_METADATA     = THINGS_EEG2_ROOT / "image_set" / "image_metadata.npy"
THINGS_EEG2_SUBJECTS     = [f"sub-{i:02d}" for i in range(1, 11)]  # sub-01..sub-10

# -------------------------------------------------------------------
# THINGS image database
# -------------------------------------------------------------------
THINGS_DB_ROOT = Path(os.environ.get("THINGS_DB_ROOT", _SCRATCH / "THINGS-database"))

# -------------------------------------------------------------------
# Winoground  (H1) — gated HuggingFace dataset
# Not pre-downloaded; use prepare_eval_data.py + HF_TOKEN.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# ARO  (H1) — open; clone https://github.com/mertyg/vision-language-models-are-bows
# and set ARO_ROOT if downloaded.
# -------------------------------------------------------------------
ARO_ROOT = Path(os.environ.get("ARO_ROOT", _SCRATCH / "aro"))


def assert_nsd_available() -> None:
    """Raise FileNotFoundError with a helpful message if NSD is not mounted."""
    missing = []
    for p in [NSD_BETAS_ROOT, NSD_STIM_INFO, NSD_STIMULI_HDF5]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "NSD not found at expected paths:\n  " + "\n  ".join(missing)
            + f"\nSet NSD_ROOT env var to override (default: {NSD_ROOT})."
        )


def assert_things_eeg2_available() -> None:
    """Raise FileNotFoundError with a helpful message if THINGS-EEG2 is not mounted."""
    if not THINGS_EEG2_PREPROCESSED.exists():
        raise FileNotFoundError(
            f"THINGS-EEG2 not found at {THINGS_EEG2_PREPROCESSED}. "
            f"Set THINGS_EEG2_ROOT env var to override."
        )
