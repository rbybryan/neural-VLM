"""Cross-model specificity driver for H1 / H4.

For each control in C1-C9 plus the main model, build the appropriate
RecurrentVLM (config overrides via configs/controls.yaml), load the
trainable-state checkpoint, run every configured benchmark via k_sweep,
and save the full results plus a contrast table.

The contrast table is the empirical realization of the spec's headline
claim: a *pattern of dissociations* — main beats compute-matched FF (C1)
and language-stripped recurrence (C2) by predicted margins; not just
"main outperforms baseline by X." Reviewers care about the pattern, not
the absolute numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .benchmarks import Benchmark
from .k_sweep import k_sweep
from .scoring import Scorer


@dataclass
class ControlSpec:
    """One row in the specificity suite.

    Args:
        name: human-readable identifier (used as dict key in outputs).
        control_id: key in configs/controls.yaml (e.g. 'C2_recurrence_no_llm').
            None means use the base config unchanged ("main").
        checkpoint_path: path to a trainable_state.pt saved by Trainer.
            None means use the model fresh from the build path (only
            sensible for the smoke test).
    """
    name: str
    control_id: str | None = None
    checkpoint_path: str | None = None


@dataclass
class SuiteResult:
    """One control's results across all benchmarks."""
    name: str
    control_id: str | None
    per_benchmark: dict[str, dict] = field(default_factory=dict)


def compute_specificity_contrasts(
    results: dict[str, SuiteResult],
    main_name: str = "main",
) -> dict:
    """For each (benchmark, K, metric), compute (main - control) deltas.

    A negative delta on a metric where main should win (e.g., Winoground
    group_score) is a *failure* of the specificity claim — the control
    matched or beat main.
    """
    if main_name not in results:
        raise KeyError(f"main control {main_name!r} not in results")
    main = results[main_name]

    contrasts: dict[str, dict] = {}
    for ctrl_name, ctrl in results.items():
        if ctrl_name == main_name:
            continue
        contrasts[ctrl_name] = {}
        for bench_name, main_bench in main.per_benchmark.items():
            ctrl_bench = ctrl.per_benchmark.get(bench_name)
            if ctrl_bench is None:
                continue
            contrasts[ctrl_name][bench_name] = {}
            for K, main_metrics in main_bench["per_K"].items():
                ctrl_metrics = ctrl_bench["per_K"].get(K, {})
                row = {}
                for metric_name, main_val in main_metrics.items():
                    if not isinstance(main_val, (int, float)):
                        continue
                    ctrl_val = ctrl_metrics.get(metric_name, 0.0)
                    row[metric_name] = {
                        "main": main_val,
                        "control": ctrl_val,
                        "delta": main_val - ctrl_val,
                    }
                contrasts[ctrl_name][bench_name][f"K={K}"] = row
    return contrasts


def run_specificity_suite(
    controls: list[ControlSpec],
    benchmarks: list[Benchmark],
    tokenizer,
    build_model_fn: Callable[[ControlSpec], torch.nn.Module],
    output_dir: str | Path | None = None,
    K_values: tuple[int, ...] = (1, 2),
    main_name: str = "main",
    progress: bool = False,
    log_fn: Callable[[str, int, dict], None] | None = None,
) -> dict:
    """Run every benchmark at every K for every control. Returns
    {"results": {control_name: SuiteResult}, "contrasts": {...}}.

    build_model_fn is injected so callers control:
        - whether to load real Qwen2.5 weights vs use a smoke model
        - whether to apply checkpoint state
        - device/dtype placement

    A typical production build_model_fn:

        def build(ctrl):
            cfg = load_config("configs/main.yaml")
            if ctrl.control_id:
                cfg = apply_control_overrides(cfg, ctrl.control_id)
            model = build_main_model(cfg, pretrained=True, dtype=torch.bfloat16)
            model = model.to("cuda")
            if ctrl.checkpoint_path:
                load_trainable_state(model, ctrl.checkpoint_path)
            return model
    """
    results: dict[str, SuiteResult] = {}

    for ctrl in controls:
        if log_fn:
            log_fn(ctrl.name, 0, {"status": "building"})

        model = build_model_fn(ctrl)
        model.eval()
        scorer = Scorer(model=model, tokenizer=tokenizer)

        per_benchmark = {}
        for bench in benchmarks:
            def _bench_log(K, metrics, _ctrl=ctrl.name, _bench=bench.name):
                if log_fn:
                    log_fn(_ctrl, K, {"benchmark": _bench, **metrics})

            result = k_sweep(
                model, scorer, bench,
                K_values=K_values,
                progress=progress,
                log_fn=_bench_log,
            )
            per_benchmark[bench.name] = {
                "per_K": result["per_K"],
                "K_values": result["K_values"],
                # Drop scores_per_K from the cross-suite output to keep
                # JSON small; callers who need per-query scores should
                # call k_sweep() directly.
            }

        results[ctrl.name] = SuiteResult(
            name=ctrl.name,
            control_id=ctrl.control_id,
            per_benchmark=per_benchmark,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    contrasts = compute_specificity_contrasts(results, main_name=main_name)

    final = {
        "results": {
            name: {
                "name": r.name,
                "control_id": r.control_id,
                "per_benchmark": r.per_benchmark,
            }
            for name, r in results.items()
        },
        "contrasts": contrasts,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "specificity_results.json", "w") as f:
            json.dump(final, f, indent=2, default=str)

    return final
