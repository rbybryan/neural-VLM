"""Temporal generalization for THINGS-MEG (King & Dehaene 2014 framework).

For each model iteration k and each MEG time bin t, train a linear decoder
to predict iteration-k features from MEG sensors at time t (or vice versa).
Decoding accuracy as a function of (k, t) yields a generalization matrix.

Pre-registered prediction (H3):
  - iter-1 features generalize best to early time bins (~80-150 ms)
  - iter-2 features generalize best to late time bins (~150-300 ms)

The iteration-scrambling control (C6) should abolish the dissociation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class GenResult:
    decoding_accuracy: np.ndarray   # (n_iterations, n_timebins)
    times_ms: np.ndarray            # (n_timebins,)


def temporal_generalization(
    iter_features: list[np.ndarray],   # list of (n_stim, d_k) per iteration
    meg: np.ndarray,                   # (n_stim, n_timebins, n_sensors)
    times_ms: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1e3,
) -> GenResult:
    n_stim, n_t, _ = meg.shape
    n_iter = len(iter_features)
    accuracy = np.zeros((n_iter, n_t))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    for k, feats in enumerate(iter_features):
        for t in range(n_t):
            X = meg[:, t, :]
            Y = feats
            preds = np.zeros_like(Y)
            for train, test in kf.split(X):
                Xs = StandardScaler().fit(X[train])
                model = Ridge(alpha=alpha).fit(Xs.transform(X[train]), Y[train])
                preds[test] = model.predict(Xs.transform(X[test]))
            ss_res = ((Y - preds) ** 2).sum()
            ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
            accuracy[k, t] = 1.0 - ss_res / max(ss_tot, 1e-8)

    return GenResult(decoding_accuracy=accuracy, times_ms=times_ms)


def early_vs_late_dissociation(
    result: GenResult,
    early_window_ms: tuple[float, float] = (80.0, 150.0),
    late_window_ms: tuple[float, float] = (150.0, 300.0),
    n_bootstrap: int = 10_000,
    rng_seed: int = 0,
) -> dict:
    """H3 contrast: (iter2 - iter1) accuracy at late window > at early window."""
    rng = np.random.default_rng(rng_seed)
    t = result.times_ms
    early = (t >= early_window_ms[0]) & (t < early_window_ms[1])
    late = (t >= late_window_ms[0]) & (t < late_window_ms[1])

    iter1, iter2 = result.decoding_accuracy[0], result.decoding_accuracy[1]
    diff = iter2 - iter1
    observed = diff[late].mean() - diff[early].mean()

    pool = diff
    boot = np.empty(n_bootstrap)
    n_e = int(early.sum())
    n_l = int(late.sum())
    for b in range(n_bootstrap):
        e = rng.choice(pool[early], size=n_e, replace=True).mean()
        l = rng.choice(pool[late], size=n_l, replace=True).mean()
        boot[b] = l - e

    return {
        "observed_interaction": float(observed),
        "ci_95": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
        "p_value": float((boot <= 0).mean()),
    }
