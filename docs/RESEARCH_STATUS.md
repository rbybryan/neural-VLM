# Current status

The recurrent model and its training/evaluation code are implemented. My next step
is to run a matched baseline and recurrent model on real data.

## What runs now

The CPU suite uses ConvNeXt-V2-atto and a small Qwen2 model, both randomly initialized.
All seven checks pass. They cover:

- Agreement between the single-pass and recurrent paths at zero gate initialization.
- Finite loss and backpropagation through the model.
- Synthetic conversations, batching, and perceptual-hash filtering.
- A ten-step training run that updates the gates and saves a checkpoint.
- Scoring, control sweeps, probe behavior, and dataset-manifest conversion.

This gives me a working starting point for the experiments. The short training run
prints a loss curve, but it isn't a convergence study. The synthetic evaluation
scores aren't benchmark results.

## What still needs testing

Real-weight loading, LoRA integration, GPU/distributed training, and the neural and
behavioral analyses need separate validation. The CPU suite doesn't cover those paths.
I don't yet have results showing that recurrence improves compositional reasoning,
brain alignment, or reaction-time prediction.

Two control definitions also need implementation work before they can support the
comparisons in the research plan:

- **C1, compute-matched feedforward:** `configs/controls.yaml` specifies
  `extra_depth_per_stage`, but [the builder](../scripts/build_main.py) doesn't read
  that field. It currently changes the iteration count without adding the intended
  encoder depth. FLOPs have not been matched.
- **C2, recurrence without language feedback:** `use_llm_kv=False` substitutes visual
  features for the direct adapter context, but [the recurrent loop](../model/recurrent_vlm.py)
  still updates the memory tokens from LLM hidden states. It does not yet remove all
  language feedback.

The [control smoke test](../scripts/test_specificity_suite.py) exercises main, C2,
and C3 on a small model with zero-initialized gates. Agreement there checks the initial
outputs and result format, not whether trained controls isolate the intended effects.

For the first real-data comparison, I want to:

1. Complete and verify the control definitions, then establish a single-pass baseline
   and check the data splits and decontamination.
2. Train the recurrent model under the same setup, recording the config and seeds.
3. Compare the controls to separate the effects of recurrence, language feedback,
   and memory, with uncertainty estimates for the differences.

The [hypotheses](../PREREGISTRATION.md) and [specification](../SPEC_v1.md) are planning
records in this repository; there is no external preregistration record linked here.
Changes to the plan go in dated amendments. The existing
[H3 amendment](preregistration_amendment_01.md) switches the neural dataset from
THINGS-MEG to THINGS-EEG2. The [documentation review](preregistration_amendment_02.md)
records the implementation gaps and the withdrawal of unsupported planning estimates.
