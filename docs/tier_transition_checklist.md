# Tier transition checklist

When moving a setup from the 1.5B tier (`configs/main.yaml`) to the 7B
publication tier (`configs/main_pub.yaml`), the architecture decisions
should hold but a brief re-validation suite is required before any
pre-registered confirmatory analysis is run on the 7B model.

This checklist exists because the optimal injection depth, K, and gate
behavior *can* shift with model capability — Bar's PFC->IT story arguably
needs enough PFC capacity to be meaningful, and the LLM's semantic priors
are what get re-injected. A 1.5B model's priors are not the same as a 7B
model's priors.

## Pre-flight checks (before running any 7B Stage-2)

- [ ] Confirm 1.5B Stage-1 alignment loss is below the random-init floor
      (~ln(vocab) ≈ 12 for Qwen2.5; expect Stage-1 to land near 4-6).
- [ ] Confirm 1.5B gate magnitudes at the end of Stage-2 are non-trivial
      across both injection stages (max gate > 0.1, neither stage at
      sustained zero). Gate collapse at 1.5B will reproduce at 7B.
- [ ] Run K-sweep on the 1.5B model first (`--control C8_K_sweep`).
      If K=2 doesn't beat K=1 by at least 1% on SugarCrepe at 1.5B, do
      not commit 7B compute.

## During Stage-1 alignment (7B)

- [ ] Stage-1 alignment loss should reach a comparable normalized level
      to 1.5B (e.g., loss / ln(vocab) within 30%).
- [ ] No exploding-gradient events; LoRA stays stable through warmup.
- [ ] Gate magnitudes still zero at end of Stage-1 (Stage-1 does not
      train them; they unlock in Stage-2).

## During Stage-2 SFT (7B)

- [ ] **Gate magnitudes track the 1.5B trajectory.** If 1.5B gates rose
      to ~0.3 by step 5000, 7B gates should be in the same order of
      magnitude (0.1-0.5). Ten-fold smaller indicates the larger LLM's
      priors are too strong to need top-down — re-think injection depth.
- [ ] **Iteration-1 loss curve.** With aux_iter1_loss_weight=0.3, iter-1
      loss should descend monotonically; if it plateaus while final
      loss descends, the model is learning to require K=2 — broken
      K=1 inference path. Increase aux weight to 0.5.
- [ ] **CNN-stage gradient norms.** Confirm freeze on stages 0-1 holds
      (norm == 0) and stages 2-3 are training (norm > 0).
- [ ] **Catastrophic forgetting check** every 5000 steps: run a 100-prompt
      eval on a held-out instruction set (no images). Loss should rise
      by less than 20% from frozen-LLM baseline. If higher, increase
      replay_ratio.

## Post-Stage-2 sanity (7B)

Before running the full pre-registered H1-H5 suite, re-confirm the
architecture decisions hold at 7B:

- [ ] **Iteration-vs-accuracy curve preserved.** On a held-out
      compositionality slice (~500 examples each from Winoground +
      SugarCrepe), the K=1 vs K=2 accuracy difference at 7B should be
      *at least as large in absolute terms* as at 1.5B. If smaller, the
      recurrence story may be eroding at scale; investigate before
      proceeding.
- [ ] **Gate-stage dissociation preserved.** Inject-at-2 vs
      inject-at-3-only ablations at 7B should reproduce the 1.5B
      anatomy preference. If the optimal injection depth shifted
      (e.g., now favors stage 1), update the pre-registration with
      an exploratory analysis note rather than silently switching.
- [ ] **Standard VLM benchmark parity.** MMBench dev / MMVet should
      land at or above the LoRA-only Qwen2.5-7B-Instruct baseline (no
      visual encoder + recurrence).

## What can break and what to do

| Symptom at 7B but not 1.5B | Most likely cause | Action |
|---|---|---|
| Gate magnitudes 10x smaller | LLM priors saturate without needing top-down | Try injection_layer = 7 (earlier in LLM, less specialized hidden states) |
| K=2 inference no longer helps | Aux iter-1 loss too weak; model learned K=2-only path | Bump aux_iter1_loss_weight to 0.5 |
| Catastrophic forgetting >>1.5B | Larger LLM more fragile to LoRA drift | Increase replay_ratio to 0.15, consider freezing embeddings |
| Stage-2 OOM | Activation memory blew up | Reduce batch_size, add gradient checkpointing on ConvNeXt stages |
| Loss plateaus at high value | Connector under-capacity for larger LLM dim | Increase MLPConnector hidden_mult from 2 to 4 |

## What to record in the paper

Whatever transition checks you run, report them:

> "The architecture, control suite, and analysis pipeline were finalized at
> Qwen2.5-1.5B. Before running confirmatory analyses at 7B, we verified
> [list]. Architecture decisions transferred with [no changes / the
> following changes: ...]."

This pre-empts the "did you cherry-pick the model size?" reviewer concern.
