"""End-to-end smoke test for the cross-control specificity driver and the
iteration-KL metric.

Uses the synthetic smoke model so all controls (main, C2, C3) can be
exercised on CPU in seconds. Verifies:

  1. The specificity driver builds different models per control toggle
     and runs all benchmarks per K via k_sweep.
  2. The output JSON has the expected shape (results + contrasts).
  3. At init (gates=0), all controls produce numerically identical
     scores on synthetic benchmarks (since none of the recurrence
     pathways do anything yet). Contrasts therefore evaluate to ~0.
  4. The IterationDivergence metric returns KL == 0 at init.
  5. per_image_metric() correctly aggregates per-query KL into one value
     per image (via group_id).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data import build_eval_transform                                          # noqa: E402
from evaluation import (                                                        # noqa: E402
    BinaryDiscrimination,
    ControlSpec,
    IterationDivergence,
    Winoground,
    run_specificity_suite,
)
from scripts.build_main import build_smoke_model                              # noqa: E402
from scripts.test_data_pipeline import StubTokenizer                          # noqa: E402
from scripts.test_eval import (                                               # noqa: E402
    _make_synthetic_sugarcrepe,
    _make_synthetic_winoground,
)


def _apply_smoke_overrides(model, ctrl: ControlSpec):
    """For the smoke test, we don't have controls.yaml-style HF configs
    to rebuild — we just toggle the relevant RecurrentVLMConfig fields on
    the shared smoke model."""
    if ctrl.control_id is None:
        # main: K=2, language-conditioned, full memory
        model.cfg.num_iterations = 2
        model.cfg.use_recurrence = True
        model.cfg.use_llm_kv = True
        model.cfg.randomize_llm_kv = False
        model.cfg.lesion_memory = False
    elif ctrl.control_id == "C2_recurrence_no_llm":
        model.cfg.num_iterations = 2
        model.cfg.use_recurrence = True
        model.cfg.use_llm_kv = False           # the key C2 toggle
    elif ctrl.control_id == "C3_llm_no_recurrence":
        model.cfg.num_iterations = 1
        model.cfg.use_recurrence = False       # forces K=1
    else:
        raise ValueError(f"unknown control_id for smoke test: {ctrl.control_id}")
    return model


def main():
    torch.manual_seed(0)
    smoke_model = build_smoke_model().eval()
    tokenizer = StubTokenizer(vocab_size=2048)

    def build_for_ctrl(ctrl: ControlSpec):
        # Mutate the shared smoke model; harmless because we save/restore
        # via run_specificity_suite's existing finally-block patterns (or
        # because the smoke test runs each control sequentially anyway).
        return _apply_smoke_overrides(smoke_model, ctrl)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wino_json, wino_root = _make_synthetic_winoground(tmp_path, n=2)
        sc_json, sc_root = _make_synthetic_sugarcrepe(tmp_path, n=2)
        transform = build_eval_transform(image_size=224)

        benchmarks = [
            Winoground(items=wino_json, image_root=wino_root, image_transform=transform),
            BinaryDiscrimination(items=sc_json, image_root=sc_root, image_transform=transform,
                                  name="sugarcrepe_smoke"),
        ]

        controls = [
            ControlSpec(name="main", control_id=None),
            ControlSpec(name="C2", control_id="C2_recurrence_no_llm"),
            ControlSpec(name="C3", control_id="C3_llm_no_recurrence"),
        ]

        def log(name, K, info):
            tag = f"[{name}]"
            if "status" in info:
                print(f"{tag} {info['status']}")
            else:
                bench = info.get("benchmark", "?")
                vals = {k: v for k, v in info.items() if k != "benchmark"}
                print(f"{tag} {bench} K={K}: {vals}")

        print("Running specificity suite on 3 controls x 2 benchmarks x 2 K values")
        result = run_specificity_suite(
            controls=controls,
            benchmarks=benchmarks,
            tokenizer=tokenizer,
            build_model_fn=build_for_ctrl,
            output_dir=tmp_path / "suite_out",
            K_values=(1, 2),
            log_fn=log,
        )

        assert "results" in result and "contrasts" in result
        assert set(result["results"].keys()) == {"main", "C2", "C3"}
        for name in ("main", "C2", "C3"):
            assert "winoground" in result["results"][name]["per_benchmark"]
            assert "sugarcrepe_smoke" in result["results"][name]["per_benchmark"]
        print(f"  results: 3 controls x 2 benchmarks x 2 K values  OK")

        for ctrl_name, ctrl_contrasts in result["contrasts"].items():
            for bench_name, bench_contrasts in ctrl_contrasts.items():
                for K_key, metrics in bench_contrasts.items():
                    for metric_name, vals in metrics.items():
                        delta = vals["delta"]
                        assert abs(delta) < 1e-2, (
                            f"contrast {ctrl_name}/{bench_name}/{K_key}/{metric_name} "
                            f"delta={delta} at init — expected ~0 because gates=0 makes "
                            f"all controls functionally equivalent"
                        )
        print("  contrasts: all main-vs-control deltas <= 1e-2 at init (gates=0)")
        ckpt = tmp_path / "suite_out" / "specificity_results.json"
        assert ckpt.exists(), "JSON output missing"
        print(f"  JSON: {ckpt.stat().st_size} bytes -> {ckpt}")
        print()

        # ----- IterationDivergence smoke -----
        print("Testing IterationDivergence (KL between iter-1 and iter-2)")
        _apply_smoke_overrides(smoke_model, ControlSpec(name="main", control_id=None))
        diverger = IterationDivergence(model=smoke_model, tokenizer=tokenizer)

        wino_queries = list(benchmarks[0].queries())
        kl_results = diverger.kl_many(wino_queries[:4])

        for qid, kl in kl_results.items():
            for k in ("kl_first_token", "kl_mean", "kl_max"):
                assert k in kl
                assert kl[k] >= 0, f"KL must be non-negative, got {kl[k]} for {qid}/{k}"
                assert kl[k] < 1e-3, (
                    f"At init (gates=0), iter-1 == iter-2, so KL must be ~0. "
                    f"Got {qid}/{k} = {kl[k]}"
                )
        max_kl = max(kl["kl_first_token"] for kl in kl_results.values())
        print(f"  KL at init: max kl_first_token across {len(kl_results)} queries = {max_kl:.2e}")

        per_image = IterationDivergence.per_image_metric(kl_results, which="kl_first_token")
        # Wino has 4 queries per group_id; with 4 queries used here, we have 1 group
        assert len(per_image) == 1, f"expected 1 group from 4 queries (1 wino item), got {len(per_image)}"
        print(f"  per_image aggregation: {len(per_image)} group(s) from {len(kl_results)} queries  OK")

        print("test_specificity_suite: PASSED")


if __name__ == "__main__":
    main()
