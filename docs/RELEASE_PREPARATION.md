# Release preparation — 2026-11-12

Prepared from `abe40d8`. Repository visibility is unchanged by this work.
The owner confirmed that the project is theirs to release. Original repository code is
provided under MIT; external models and datasets retain their own terms.

## Validation

- All seven synthetic CPU checks passed on macOS arm64 / Python 3.12 / PyTorch 2.5.1,
  using `requirements-smoke.txt` (35.5 seconds for the complete smoke runner).
- Two dataset-path regression tests passed, covering portable defaults and explicit
  overrides, including paths with spaces.
- Modified Python files compiled, the SLURM template passed `bash -n`, and local Markdown
  links in changed documents resolved.
- Gitleaks 8.30.1 scanned all four commits in the fetched history and the prepared tree;
  no credential matches were reported.

Linux/Python 3.11 execution is covered by the new GitHub workflow. Check its result on the
exact release commit. CPU checks use generated data and random weights; real-weight,
GPU/distributed, benchmark, and neural-data experiments were not rerun during preparation.

## Publication boundary

Paths and the SLURM template are now portable. Historical commits still record the
original machine-specific paths; this preparation does not rewrite history. A clean
source snapshot can be used if those older operational details should remain private.
GitHub issue/PR discussions, Actions logs, and artifacts are outside the local Git scan.
A scan reporting no matches is not proof that every kind of sensitive material is absent.

## Release scope

Publish as an experimental research implementation with runnable synthetic checks.
Do not describe those checks as a trained-model benchmark or evidence for H1–H5.
Preserve the original research plans and their dated amendments.
