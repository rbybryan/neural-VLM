"""NSD data loader utilities for H2 (variance partition) and H4 (caption specificity).

Layout assumed (under the configured NSD_ROOT):

  nsddata_betas/ppdata/subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/
      betas_session01.hdf5 ... betas_session40.hdf5
      (each HDF5 contains 'betas': float32 (n_trials_in_session, x, y, z))

  nsddata/ppdata/subjXX/func1pt8mm/roi/
      lh.prf-visualrois.nii.gz  (V1=1, V2=2, V3=3, hV4=4, ...)
      lh.floc-faces.nii.gz
      lh.floc-places.nii.gz
      lh.floc-bodies.nii.gz

  nsddata/experiments/nsd/nsd_stim_info_merged.csv
      cocoId, nsdId, subjectX columns used to map stimuli to trials
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from analysis.paths import (
    NSD_BETAS_ROOT,
    NSD_DATA_ROOT,
    NSD_PPDATA,
    NSD_ROI_FILES,
    NSD_RES,
    NSD_STIM_INFO,
    NSD_SUBJECTS,
    assert_nsd_available,
)


# ---------------------------------------------------------------------------
# ROI masks
# ---------------------------------------------------------------------------

def load_roi_mask(subject: str, roi_key: str) -> np.ndarray:
    """Return flat boolean mask (n_voxels,) for a single hemisphere.

    roi_key: one of the keys in NSD_ROI_FILES.
    Positive voxels are those with label > 0 in the NIfTI.
    """
    fname = NSD_ROI_FILES[roi_key]
    path = NSD_PPDATA / subject / NSD_RES / "roi" / fname
    img = nib.load(path)
    data = img.get_fdata().astype(np.int32)
    return data.ravel() > 0


def load_early_visual_mask(subject: str) -> np.ndarray:
    """V1 + V2 + V3 mask (pRF-defined, labels 1-3 in prf-visualrois)."""
    path = NSD_PPDATA / subject / NSD_RES / "roi" / NSD_ROI_FILES["prf_visualrois"]
    img = nib.load(path)
    data = img.get_fdata().astype(np.int32)
    return ((data >= 1) & (data <= 3)).ravel()


def load_late_ventral_mask(subject: str) -> np.ndarray:
    """LOC (floc-bodies) + FFA (floc-faces) + PPA (floc-places) mask."""
    masks = []
    for key in ("floc_faces", "floc_places", "floc_bodies"):
        masks.append(load_roi_mask(subject, key))
    return np.any(masks, axis=0)


# ---------------------------------------------------------------------------
# Beta loading
# ---------------------------------------------------------------------------

def _session_files(subject: str) -> list[Path]:
    beta_dir = NSD_BETAS_ROOT / subject / NSD_RES / "betas_fithrf_GLMdenoise_RR"
    files = sorted(beta_dir.glob("betas_session*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 beta files found in {beta_dir}")
    return files


def load_betas_for_subject(
    subject: str,
    nsd_ids: Sequence[int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Load trial-averaged betas for the given NSD stimulus IDs.

    nsd_ids: 0-based nsdId values (from nsd_stim_info_merged.csv).
    mask: optional flat boolean array; if provided, returns (n_stim, n_masked_voxels).
    Returns: float32 array (n_stim, n_voxels_or_masked).

    Each NSD image is shown up to 3 times per subject; this function averages
    across repeats automatically.
    """
    stim_info = _load_stim_info()
    subj_col = f"{subject}_rep0"  # e.g. subject1_rep0 — trial index within that session
    # Map nsdId -> list of (session_file, trial_within_session) tuples
    id_to_trials: dict[int, list[tuple[Path, int]]] = {nid: [] for nid in nsd_ids}

    for rep_suffix in ("_rep0", "_rep1", "_rep2"):
        col = subject + rep_suffix
        if col not in stim_info.columns:
            continue
        rows = stim_info[stim_info["nsdId"].isin(nsd_ids)]
        for _, row in rows.iterrows():
            trial_global = int(row[col])
            if trial_global == 0:
                continue  # not shown in this repeat
            nid = int(row["nsdId"])
            # Global 1-based trial index; sessions have 750 trials each
            session_idx = (trial_global - 1) // 750
            trial_in_session = (trial_global - 1) % 750
            session_files = _session_files(subject)
            if session_idx < len(session_files):
                id_to_trials[nid].append((session_files[session_idx], trial_in_session))

    # Load betas; cache open HDF5 files by path
    open_files: dict[Path, h5py.File] = {}
    try:
        out_list = []
        for nid in nsd_ids:
            trials = id_to_trials.get(nid, [])
            if not trials:
                out_list.append(None)
                continue
            vecs = []
            for fpath, t_idx in trials:
                if fpath not in open_files:
                    open_files[fpath] = h5py.File(fpath, "r")
                beta_vol = open_files[fpath]["betas"][t_idx]  # (x, y, z) float32
                vecs.append(beta_vol.ravel())
            arr = np.stack(vecs, axis=0).mean(axis=0)  # average repeats
            out_list.append(arr)
    finally:
        for f in open_files.values():
            f.close()

    # Fill missing with zeros and stack
    n_vox = next(v for v in out_list if v is not None).shape[0]
    result = np.stack(
        [v if v is not None else np.zeros(n_vox, dtype=np.float32) for v in out_list],
        axis=0,
    ).astype(np.float32)

    if mask is not None:
        result = result[:, mask]
    return result


def load_repeats_for_subject(
    subject: str,
    nsd_ids: Sequence[int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-repeat betas (n_repeats, n_stim, n_voxels) for noise ceiling.

    Only stimuli with all 3 repeats are included; shape is (3, n_stim, n_voxels).
    """
    stim_info = _load_stim_info()
    session_files = _session_files(subject)
    open_files: dict[Path, h5py.File] = {}
    reps_per_stim: list[list[np.ndarray]] = [[] for _ in nsd_ids]
    nid_to_idx = {nid: i for i, nid in enumerate(nsd_ids)}

    try:
        for rep_suffix in ("_rep0", "_rep1", "_rep2"):
            col = subject + rep_suffix
            if col not in stim_info.columns:
                continue
            rows = stim_info[stim_info["nsdId"].isin(nsd_ids)]
            for _, row in rows.iterrows():
                trial_global = int(row[col])
                if trial_global == 0:
                    continue
                nid = int(row["nsdId"])
                idx = nid_to_idx[nid]
                session_idx = (trial_global - 1) // 750
                trial_in_session = (trial_global - 1) % 750
                if session_idx >= len(session_files):
                    continue
                fpath = session_files[session_idx]
                if fpath not in open_files:
                    open_files[fpath] = h5py.File(fpath, "r")
                beta_vol = open_files[fpath]["betas"][trial_in_session].ravel()
                reps_per_stim[idx].append(beta_vol)
    finally:
        for f in open_files.values():
            f.close()

    n_vox = next(r[0] for r in reps_per_stim if r).shape[0]
    max_reps = max((len(r) for r in reps_per_stim), default=0)
    out = np.full((max_reps, len(nsd_ids), n_vox), np.nan, dtype=np.float32)
    for i, reps in enumerate(reps_per_stim):
        for r, vec in enumerate(reps):
            out[r, i] = vec

    if mask is not None:
        out = out[:, :, mask]
    return out


# ---------------------------------------------------------------------------
# Stimulus info
# ---------------------------------------------------------------------------

_stim_info_cache: pd.DataFrame | None = None


def _load_stim_info() -> pd.DataFrame:
    global _stim_info_cache
    if _stim_info_cache is None:
        _stim_info_cache = pd.read_csv(NSD_STIM_INFO)
    return _stim_info_cache


def coco_ids_for_subject(subject: str) -> np.ndarray:
    """Return COCO image IDs seen by this subject (any repeat)."""
    info = _load_stim_info()
    seen = info[
        (info[f"{subject}_rep0"] > 0)
        | (info[f"{subject}_rep1"] > 0)
        | (info[f"{subject}_rep2"] > 0)
    ]
    return seen["cocoId"].values


def shared1000_nsd_ids() -> np.ndarray:
    """Return the 1000 NSD stimuli shared across all 8 subjects."""
    info = _load_stim_info()
    return info[info["shared1000"]]["nsdId"].values
