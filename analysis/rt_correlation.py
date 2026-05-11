"""Image-averaged RT (and MEG decoding latency) correlation with model
iterations-to-converge, with feedforward-difficulty partialing.

The pre-registered analysis (H5) is:
  1. For each image, compute the model's iterations-to-converge metric
     (gate magnitude on iter-2, or KL between iter-1 and iter-2 logits).
  2. Compute image-averaged human RT (or MEG decoding latency).
  3. Compute partial correlation between (1) and (2) controlling for a
     feedforward-difficulty proxy (CNN softmax entropy / accuracy).
  4. Specificity: this partial correlation must be larger for the main
     model than for C2 (no LLM) and C3 (no recurrence).
  5. Report noise-ceiling-normalized partial correlation.

Realistic target: r ≈ 0.3-0.4 image-averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class RTCorrelationResult:
    partial_r: float
    raw_r: float
    feedforward_r: float
    noise_ceiling: float
    n_images: int


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of x,y controlling for z."""
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom < 1e-8:
        return 0.0
    return float((rxy - rxz * ryz) / denom)


def split_half_noise_ceiling(rts_per_subject: np.ndarray, n_iter: int = 1000,
                             rng_seed: int = 0) -> float:
    """rts_per_subject: (n_subjects, n_images). Returns split-half ceiling on
    image-averaged RT — the upper bound any model correlation can reach."""
    rng = np.random.default_rng(rng_seed)
    n_s = rts_per_subject.shape[0]
    rs = np.empty(n_iter)
    for i in range(n_iter):
        perm = rng.permutation(n_s)
        a = rts_per_subject[perm[: n_s // 2]].mean(axis=0)
        b = rts_per_subject[perm[n_s // 2: 2 * (n_s // 2)]].mean(axis=0)
        rs[i] = spearmanr(a, b).correlation
    r_half = np.median(rs)
    # Spearman-Brown step-up to full sample
    return float(2 * r_half / (1 + r_half))


def rt_correlation(
    iterations_metric: np.ndarray,         # (n_images,) e.g. KL(iter1, iter2)
    rt_per_image: np.ndarray,              # (n_images,) image-averaged human RT
    feedforward_difficulty: np.ndarray,    # (n_images,) e.g. CNN entropy
    rts_per_subject: np.ndarray | None = None,
) -> RTCorrelationResult:
    raw_r = float(spearmanr(iterations_metric, rt_per_image).correlation)
    ff_r = float(spearmanr(feedforward_difficulty, rt_per_image).correlation)
    pr = partial_corr(iterations_metric, rt_per_image, feedforward_difficulty)

    if rts_per_subject is not None:
        ceiling = split_half_noise_ceiling(rts_per_subject)
    else:
        ceiling = 1.0

    return RTCorrelationResult(
        partial_r=pr / max(ceiling, 1e-3),
        raw_r=raw_r,
        feedforward_r=ff_r,
        noise_ceiling=ceiling,
        n_images=iterations_metric.size,
    )


def specificity_contrast(
    main_metric: np.ndarray,
    control_metric: np.ndarray,
    rt_per_image: np.ndarray,
    feedforward_difficulty: np.ndarray,
    n_bootstrap: int = 10_000,
    rng_seed: int = 0,
) -> dict:
    """H5: partial r(main) > partial r(control) at p < 0.05.

    Bootstrap over images. Returns observed difference and 95% CI.
    """
    rng = np.random.default_rng(rng_seed)
    n = rt_per_image.size

    obs_main = partial_corr(main_metric, rt_per_image, feedforward_difficulty)
    obs_ctrl = partial_corr(control_metric, rt_per_image, feedforward_difficulty)
    observed = obs_main - obs_ctrl

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = partial_corr(main_metric[idx], rt_per_image[idx], feedforward_difficulty[idx])
        c = partial_corr(control_metric[idx], rt_per_image[idx], feedforward_difficulty[idx])
        boot[b] = m - c

    return {
        "main_partial_r": float(obs_main),
        "control_partial_r": float(obs_ctrl),
        "difference": float(observed),
        "ci_95": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
        "p_value": float((boot <= 0).mean()),
    }
