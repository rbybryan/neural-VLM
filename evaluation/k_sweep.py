"""K-sweep: run a benchmark at multiple iteration counts.

This is the inference-time variant of the recurrence — model weights are
fixed, only `cfg.num_iterations` and `cfg.use_recurrence` change. The
result is the per-K accuracy curve that the SPEC §5.B.3 calls the "money
figure" for the H1 / H5 evidence.

K=1 is the no-recurrence reference (vanilla VLM forward).
K=2..4 reveal whether additional iterations meaningfully change scores
or plateau (biological story predicts diminishing returns).

The model's gate is *not* re-learned per K — we just rerun the same fixed
adapters K times. This is the standard interpretation of "iteration index
as ordinal processing-depth proxy" from SPEC §1.
"""

from __future__ import annotations

from typing import Callable

from .benchmarks import Benchmark
from .scoring import Scorer


def k_sweep(
    model,
    scorer: Scorer,
    benchmark: Benchmark,
    K_values: tuple[int, ...] = (1, 2),
    progress: bool = False,
    log_fn: Callable[[int, dict], None] | None = None,
) -> dict:
    """Run benchmark at each K in K_values; return per-K aggregate metrics
    plus all per-query log-likelihood scores."""
    results_per_K: dict[int, dict[str, float]] = {}
    scores_per_K: dict[int, dict[str, float]] = {}

    original_K = model.cfg.num_iterations
    original_use_rec = model.cfg.use_recurrence

    try:
        for K in K_values:
            model.cfg.num_iterations = K
            model.cfg.use_recurrence = (K > 1)

            scores = scorer.score_many(benchmark.queries(), progress=progress)
            scores_per_K[K] = scores
            results_per_K[K] = benchmark.aggregate(scores)
            if log_fn is not None:
                log_fn(K, results_per_K[K])
    finally:
        model.cfg.num_iterations = original_K
        model.cfg.use_recurrence = original_use_rec

    return {
        "benchmark": benchmark.name,
        "K_values": list(K_values),
        "per_K": results_per_K,
        "scores_per_K": scores_per_K,
    }
