# Research status and limitations

## Implemented and exercised by the synthetic CPU suite

- Recurrent model construction with random ConvNeXt-V2-atto and a small Qwen2 configuration.
- Single-pass/recurrent agreement at zero gate initialization, finite loss, and backpropagation.
- Synthetic conversation loading, perceptual-hash filtering, and collation.
- A short optimizer run that updates adapter gates and saves trainable parameters.
- Synthetic scoring, control sweeps, probe interfaces, and dataset-manifest conversion.

The checks use random weights and generated inputs. Their successful execution shows
that these paths run and selected invariants hold; it does not validate scientific hypotheses.
The training smoke check prints early/late losses but does not establish convergence.

## Implemented but requiring separate validation

Real-weight loading and LoRA integration, distributed/GPU training, real dataset evaluation,
and the neural and behavioral analysis modules require their own environment, data, and
validation. A function or script being present is not evidence that a full experiment succeeded.
The CPU suite does not test all these paths.

## Research results not claimed by this release

- Improved Winoground, SugarCrepe, or other benchmark performance.
- Successful H1–H5 confirmatory tests or a trained recurrent-model release.
- Demonstrated biological plausibility, brain alignment, or human reaction-time prediction.
- A general advantage from recurrence, language feedback, or persistent memory.

The plans in `PREREGISTRATION.md` and `SPEC_v1.md` are repository documents.
No external registry record is supplied. Preserve the original plans and record future
changes as dated amendments rather than rewriting their historical hypotheses.
H3 follows `docs/preregistration_amendment_01.md` (THINGS-EEG2).

## Next research milestone

Run a documented real-data baseline and matched recurrent/control comparison, publish
the configuration and split/decontamination record, and report uncertainty and failed
hypotheses alongside positive findings. Record the exact commit, seed, hardware, software
environment, and data/model versions. Keep tuning data separate from confirmatory evaluation.
