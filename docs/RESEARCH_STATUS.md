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

For the first real-data comparison, I want to:

1. Establish a single-pass baseline and verify the data splits and decontamination.
2. Train the recurrent model under the same setup, recording the config and seeds.
3. Compare the controls to separate the effects of recurrence, language feedback,
   and memory, with uncertainty estimates for the differences.

The [hypotheses](../PREREGISTRATION.md) and [specification](../SPEC_v1.md) are planning
records in this repository; there is no external preregistration record linked here.
Changes to the plan go in dated amendments. The existing
[H3 amendment](preregistration_amendment_01.md) switches the neural dataset from
THINGS-MEG to THINGS-EEG2.
