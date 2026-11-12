# Recurrent VLM: design and experiment plan

The question is whether language-conditioned feedback changes visual representations in a useful way across repeated passes. The implementation combines ConvNeXt-V2 with a Qwen language model and feeds language hidden states back into intermediate visual stages.

This is a plan for experiments, not a report of brain alignment or benchmark gains. The [current status](docs/RESEARCH_STATUS.md) separates the synthetic checks from the work still needed. The original plan is retained at the `v0-preregistration` tag; the [documentation amendment](docs/preregistration_amendment_02.md) records the corrections made here.

## Architecture

The [default config](configs/main.yaml) uses ConvNeXt-V2-Base and the text-only `Qwen/Qwen2.5-1.5B-Instruct`. The 0.5B and 7B alternatives are in `configs/main_dev.yaml` and `configs/main_pub.yaml`. Those filenames describe configuration choices; they do not imply completed training runs or publication results.

| Component | Current implementation |
|---|---|
| Connector | Two-layer MLP from the last visual stage to the language-model hidden dimension |
| Feedback adapters | Cross-attention at encoder stages 2 and 3 in zero-based indexing; queries from visual features, context from language hidden states and memory |
| Adapter initialization | Scalar gate starts at zero; output projection uses ordinary initialization |
| Memory | 32 learnable context tokens in the default config, updated between iterations |
| Recurrence | Two iterations by default; the iteration count is a configuration setting |
| Training options | LoRA on the language model, frozen early visual stages, a separate feedback-parameter learning rate, and auxiliary loss on the first iteration |

See [the recurrent loop](model/recurrent_vlm.py), [adapter](model/top_down_adapter.py), and [builder](scripts/build_main.py).

The zero gate lets the initial recurrent output agree with the single-pass output. Keeping the output projection nonzero gives the gate a gradient so it can start learning. The CPU checks test both initial agreement and gate movement during a short training run.

The original plan mentioned a learned per-example stopping gate and estimated inference costs. The loop currently uses a fixed iteration count, and no end-to-end timing measurement supports those estimates. Parameter counts and runtime should be measured for the exact configuration used.

## Why these comparisons

Recurrence adds computation, so a gain over a single pass would not by itself explain what helped. The intended comparisons are:

| Control | Question |
|---|---|
| C1: compute-matched feedforward | Would the same computation without recurrence be enough? |
| C2: recurrence without language feedback | Does the source of feedback matter? |
| C3: single pass | Does a second pass change the result? |
| C4: randomized feedback | Does the content of the feedback matter? |
| C5: shuffled captions | Does matching the caption to the image matter? |
| C6: iteration scrambling | Do representations from different iterations carry distinct information? |
| C7: memory lesion | Does state carried between iterations matter? |
| C8: iteration sweep | How do accuracy and cost change with the number of passes? |
| C9: single-pass ensemble | Could averaging independent passes explain the gain? |

These are intended controls, not nine validated experimental conditions. In particular, the builder does not consume C1's extra-depth setting, and C2 still has language-derived memory. Those gaps must be resolved before using the controls to make the corresponding claims. The [status notes](docs/RESEARCH_STATUS.md) point to the relevant code.

## Training plan

The proposed first stage aligns the connector to frozen visual and language backbones, using one pass. The proposed second stage enables recurrence, LoRA, and updates to the later visual stages, with an auxiliary first-pass loss weighted at 0.3.

The initial data plan considered ShareGPT4V for alignment, followed by a mixture of LLaVA, OneVision, Cambrian, and language-instruction replay. A final mixture, data access, filtering, and compute budget still need to be established. The earlier schedule and training-duration estimates were not measured results.

Before a large run, the immediate work is to validate real-weight assembly, make the controls do what their names say, check data splits and near-duplicate filtering, and run a small matched comparison. Commands are in [the training guide](docs/USAGE.md).

## Evaluation plan

Start with Winoground and SugarCrepe-style compositional comparisons, recording per-example scores for paired analysis. Evaluate the baseline and recurrent model under the same data and training setup. An iteration sweep should report computation alongside accuracy.

The longer-term plan includes neural and behavioral comparisons:

- NSD encoding and variance partitioning across iterations.
- THINGS-EEG2 temporal analysis, following the [H3 dataset amendment](docs/preregistration_amendment_01.md).
- Reaction-time associations after accounting for image difficulty.
- Probes of feedback modulation and the contents of memory tokens.

The original plan also proposed occlusion/OOD tests and broader VLM benchmarks. Those remain extensions; the repository does not contain completed results for them.

[H1–H5](PREREGISTRATION.md) retain the recorded hypotheses. Their statistical procedures and power assumptions need validation before confirmatory use. The proposed effect sizes are not observations.

## Interpretation

ConvNeXt provides spatial feature maps at which to apply feedback. That engineering choice does not establish biological plausibility. Likewise, an iteration is a processing step, not a calibrated interval in brain time, and persistent tokens are not evidence of working-memory behavior.

A useful result would identify which changes survive matched controls and repeated runs. A null or negative result is still informative if the implementation and comparison are sound.
