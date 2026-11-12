# Documentation review: plans, implementation, and evidence

**Date:** 2026-11-12

This review clarifies the current documentation. It does not report a new experiment or change the recorded directions and decision thresholds of H1–H5. The original plan remains available at the `v0-preregistration` tag.

## What needed correction

The specification presented an intended brain/behavior result as a claim, described an adaptive iteration gate that is not implemented, and gave unmeasured timing and parameter-count estimates. It also used a different model description from the current default config. The design notes now describe the implemented loop and label the rest as planned work.

The power section in `PREREGISTRATION.md` gave numerical assurances without a supporting calculation or simulation in the repository. Those assurances are withdrawn from the current document. A power analysis and a review of the sampling units are still needed before confirmatory use; the old values remain in the tagged historical version.

## Control gaps found during review

- The C1 config contains `extra_depth_per_stage`, but `build_main_model` does not use it when constructing the encoder. The current C1 path is not verified as compute-matched.
- C2 replaces direct language keys/values with projected visual features, while the recurrent loop still updates memory from language hidden states. It does not remove every language-feedback path.
- The synthetic suite exercises main, C2, and C3 at zero gate initialization. Its passing result does not establish the scientific validity of the full control set.

These are implementation gaps to address before the corresponding comparisons, not negative experimental findings. Code and measurements were unchanged by this documentation review.

## Registration status

`v0-preregistration` is an in-repository snapshot dated 2026-05-11. No OSF, Zenodo, or other external registration is linked here. The current implementation has evolved since that snapshot, so a future analysis must name its exact code version and any further plan amendments.
