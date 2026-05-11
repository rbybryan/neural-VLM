"""Per-iteration variance partitioning for fMRI encoding (NSD, THINGS-fMRI).

Implements the Lescroart, Stansbury & Gallant (2015) variance-partitioning
approach to ask which iteration of the recurrent VLM uniquely explains
variance in each cortical voxel.

For each voxel y:
    R2_full   = R^2(y | features_iter1, features_iter2)
    R2_iter1  = R^2(y | features_iter1)
    R2_iter2  = R^2(y | features_iter2)

    unique_iter1 = R2_full - R2_iter2
    unique_iter2 = R2_full - R2_iter1
    shared       = R2_iter1 + R2_iter2 - R2_full

The pre-registered prediction (H2) is that unique_iter2 - unique_iter1 is
positive and significant in late ventral cortex (LOC + anterior IT) and
negative in early visual cortex (V1-V3).

R^2 is reported noise-ceiling-normalized following Schoppe et al. 2016 and
Lage-Castellanos et al. 2019.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


@dataclass
class PartitionResult:
    r2_full: np.ndarray            # (n_voxels,)
    r2_iter1: np.ndarray
    r2_iter2: np.ndarray
    unique_iter1: np.ndarray
    unique_iter2: np.ndarray
    shared: np.ndarray
    noise_ceiling: np.ndarray


def _ridge_r2(X: np.ndarray, Y: np.ndarray, n_folds: int = 5,
              alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)) -> np.ndarray:
    """Cross-validated R^2 per voxel."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    preds = np.zeros_like(Y)
    for train, test in kf.split(X):
        model = RidgeCV(alphas=alphas).fit(X[train], Y[train])
        preds[test] = model.predict(X[test])
    ss_res = ((Y - preds) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-8, None)


def noise_ceiling_schoppe(repeats: np.ndarray) -> np.ndarray:
    """Estimate per-voxel noise ceiling from repeated trials.

    repeats: (n_repeats, n_stimuli, n_voxels)
    Returns the SP-derived ceiling that R^2 should be normalized by.
    """
    n_rep = repeats.shape[0]
    mean_signal = repeats.mean(axis=0)
    var_signal = mean_signal.var(axis=0)
    var_total = repeats.reshape(-1, repeats.shape[-1]).var(axis=0)
    var_noise = (var_total - var_signal).clip(min=1e-8)
    ceiling = var_signal / (var_signal + var_noise / n_rep)
    return ceiling.clip(min=1e-3)


def variance_partition(
    features_iter1: np.ndarray,        # (n_stim, d1)
    features_iter2: np.ndarray,        # (n_stim, d2)
    voxel_responses: np.ndarray,       # (n_stim, n_voxels)
    repeats: np.ndarray | None = None, # (n_rep, n_stim, n_voxels) for noise ceiling
    n_folds: int = 5,
) -> PartitionResult:
    X_full = np.concatenate([features_iter1, features_iter2], axis=1)

    r2_full = _ridge_r2(X_full, voxel_responses, n_folds=n_folds)
    r2_iter1 = _ridge_r2(features_iter1, voxel_responses, n_folds=n_folds)
    r2_iter2 = _ridge_r2(features_iter2, voxel_responses, n_folds=n_folds)

    unique_iter1 = r2_full - r2_iter2
    unique_iter2 = r2_full - r2_iter1
    shared = r2_iter1 + r2_iter2 - r2_full

    if repeats is not None:
        ceiling = noise_ceiling_schoppe(repeats)
    else:
        ceiling = np.ones_like(r2_full)

    return PartitionResult(
        r2_full=r2_full / ceiling,
        r2_iter1=r2_iter1 / ceiling,
        r2_iter2=r2_iter2 / ceiling,
        unique_iter1=unique_iter1 / ceiling,
        unique_iter2=unique_iter2 / ceiling,
        shared=shared / ceiling,
        noise_ceiling=ceiling,
    )


def roi_dissociation_test(
    result: PartitionResult,
    early_visual_mask: np.ndarray,   # (n_voxels,) bool
    late_ventral_mask: np.ndarray,   # (n_voxels,) bool
    n_bootstrap: int = 10_000,
    rng_seed: int = 0,
) -> dict:
    """H2 test: unique_iter2 > unique_iter1 in late ventral; reverse in early.

    Returns paired-bootstrap CIs and a p-value for the interaction contrast
        (uniq_iter2 - uniq_iter1)_late - (uniq_iter2 - uniq_iter1)_early > 0.
    """
    rng = np.random.default_rng(rng_seed)
    diff = result.unique_iter2 - result.unique_iter1
    early_diff = diff[early_visual_mask]
    late_diff = diff[late_ventral_mask]

    observed = late_diff.mean() - early_diff.mean()

    boot = np.empty(n_bootstrap)
    n_e, n_l = early_diff.size, late_diff.size
    for b in range(n_bootstrap):
        e = rng.choice(early_diff, size=n_e, replace=True).mean()
        l = rng.choice(late_diff, size=n_l, replace=True).mean()
        boot[b] = l - e

    p_value = float((boot <= 0).mean())
    return {
        "observed_interaction": float(observed),
        "ci_95": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
        "p_value": p_value,
        "early_mean_diff": float(early_diff.mean()),
        "late_mean_diff": float(late_diff.mean()),
    }
