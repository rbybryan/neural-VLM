# Language-Conditioned Recurrent VLM — v1 Spec

**Working title.** Language-Conditioned Recurrent Modulation in a Vision-Language Model: Specificity Contrasts Against Brain and Behavior.

**One-sentence claim.** A small language-conditioned feedback pathway that re-enters an LLM's context into a convolutional visual encoder produces brain- and behavior-aligned signatures that *specifically* depend on the joint presence of recurrence and language conditioning — not on added compute, ensembling, LLM-prior distillation, or image-difficulty confounds.

The claim is deliberately framed as a **specificity dissociation**, not a brain-alignment headline. Alignment numbers are confirmatory; the lesion pattern across control models is load-bearing.

---

## 1. Framing discipline (what we say and what we do not)

### Use these terms
- "Language-conditioned recurrent modulation"
- "Persistent context tokens" (not "PFC working memory" unless WM properties are tested)
- "Top-down gating" / "feedback modulation"
- "Iteration index as ordinal processing-depth proxy"
- "Hierarchical conditional modulation"

### Do NOT use these terms in the abstract or headline framing
- "Predictive coding" — we have additive gated cross-attention, not signed prediction-error units (Rao & Ballard 1999; Bastos et al. 2012). May appear in the discussion as a *related* framework only.
- "Bar 2003 PFC→IT shortcut" — Bar's pathway is low-spatial-frequency magnocellular gist (~130 ms). LLM hidden states are not that. Drop entirely unless we implement a low-SF fast pathway.
- "Biased competition" — keep only if the gain-modulation-on-competing-features test (§5.B.6) succeeds.
- "PFC working memory" — drop unless WM signatures (capacity limits, interference, content-specific delay decoding) are tested. Otherwise: "persistent context tokens."
- "K=2 corresponds to the 150–250 ms recurrent IT window" — drop the millisecond mapping; iteration index is ordinal.

### ConvNeXt justification
Frame as **structural** ("spatial feature maps geometrically suit spatial top-down injection"), not **biological** ("CNNs are more brain-like"). Cite Conwell et al. 2024 honestly: data > architecture for brain alignment. Plan a ViT-backbone ablation if reviewers push.

---

## 2. Architecture (unchanged from prior plan)

| Component | Choice |
|---|---|
| Visual encoder | ConvNeXt-V2-Base (~90M, ImageNet-22k) |
| LLM | Qwen2.5-VL-7B base (or InternLM2.5-7B) |
| Connector | 2-layer MLP, GELU, CNN stage-4 → LLM dim |
| Top-down adapters | Cross-attention at ConvNeXt **stages 3 and 4**; queries from CNN feature maps; keys/values from LLM layer-14 hidden states + 32 persistent context tokens; **zero-init gated output** — only the *gate* is zero-initialized (`tanh(0)=0` → modulation is zero at init), `out_proj` is normally initialized so the gate has a non-zero gradient and the recurrent pathway can escape from gate=0. Zero-initializing both gate and out_proj would lock all adapter/memory params at zero gradient indefinitely (Flamingo trick, applied correctly). |
| Persistent context tokens | 32 learnable tokens, read/written each iteration |
| Recurrence | K=2 at training; K∈{1,2} at inference via learned per-example gate |
| Total new trainable params | ~30–50M (top-down + tokens + connector) |

Inference cost: ~1.6× a non-recurrent baseline. Per-example gate keeps the average closer to 1.2× in deployment.

---

## 3. Training recipe (unchanged from prior plan)

**Stage 1 — alignment (~2 days, 4×A100):**
- Freeze CNN + LLM. Train connector + top-down adapters + context tokens.
- ShareGPT4V-PT (~1.2M). K=1.

**Stage 2 — SFT with recurrence (~1.5–2 weeks, 4–8×A100):**
- LLM: LoRA r=16, α=32 on `q,k,v,o,gate,up,down`.
- CNN: freeze stages 1–2; full-finetune stages 3–4 at 10× lower LR.
- Top-down + tokens: full-train, separate param group at 5–10× higher LR.
- Data: LLaVA-1.5-mix-665K + LLaVA-OneVision-SI subset + Cambrian-7M subset + 10% original-LLM-instruction replay.
- K=2 with auxiliary CE on iteration-1 (weight 0.3) to keep K=1 inference path valid.
- Eval-set decontamination on every benchmark before any number is reported.

---

## 4. Specificity control suite (FIRST-CLASS, not optional)

Every brain-alignment, behavioral, and benchmark number in the paper is reported for the main model **and** every control below. The headline result is the *pattern of dissociations*, not any single number.

| # | Control | What it isolates | Predicted result if our story is right |
|---|---|---|---|
| C1 | **FLOP-matched feedforward** — same total FLOPs as K=2, no recurrence (e.g., deeper CNN or larger connector) | Compute confound | Underperforms main model on compositionality / occlusion / brain alignment |
| C2 | **Recurrence-without-LLM** — same K=2 loop, but KV come from learned static tokens or lateral CNN features (no LLM hidden states) | Language-conditioning vs. recurrence per se | Captures *some* gain but not the language-specific brain alignment |
| C3 | **LLM-without-recurrence** — K=1 with same LLM cross-attention injection | The loop itself | Captures alignment to language-adjacent regions but not late-ventral iteration-2 signature |
| C4 | **Random-LLM-keys** — frozen random projections of CNN features into the LLM-shaped key space | Whether *language content* matters or just any high-dim signal | Brain alignment to language regions collapses |
| C5 | **Shuffled-caption** (eval-time) — provide a wrong/shuffled caption to NSD images | Stimulus-specific language conditioning | Late-ventral / language-adjacent alignment degrades; early visual alignment does not |
| C6 | **Iteration-scrambling** (eval-time) — swap iter-1 ↔ iter-2 features into encoding regression | Whether iterations encode distinct information | Early/late cortical dissociation collapses |
| C7 | **Memory-token lesion** (eval-time) — zero out persistent tokens at inference | Whether tokens carry information across iterations | Performance degrades on multi-step / context-dependent items only |
| C8 | **K-sweep** — K ∈ {1, 2, 3, 4} | Engineering vs biological story | Diminishing returns (saturate by K=2–3), not monotonic gains |
| C9 | **Ensembling baseline** — K=1 averaged over 2 stochastic forward passes | Implicit ensembling confound | Underperforms K=2 on compositionality and occlusion |

C1–C5 are *required*. C6–C9 are *strongly recommended* and cheap to add post-hoc. Without C1–C5 the paper does not survive review.

---

## 5. Evaluation pipeline (restructured)

### A. Standard VLM benchmarks (sanity, not headline)
MMBench, MMMU, MMVet, MME, OK-VQA, GQA, TextVQA. Goal: match or modestly exceed a Qwen2.5-VL-7B + LoRA baseline. Decontaminate.

### B. Discriminating evals (the actual paper)

**B.1 Compositionality (strongest empirical story)**
- Winoground, SugarCrepe, ARO, EqBen.
- Report main model + every control. Specificity prediction: gains require *both* recurrence and language conditioning (main > C2, C3, C9).

**B.2 Occlusion / OOD robustness**
- ImageNet-A, ImageNet-R, ObjectNet, occlusion-VQA splits.
- Report iteration-vs-accuracy curves on hard subsets.

**B.3 Iteration-vs-accuracy curves**
- For each benchmark, plot accuracy vs K (with the gate disabled) and accuracy vs gate-rate (with the gate enabled).
- The "money plot" if it works: monotonic gains on hard items, flat on easy items.

**B.4 Brain alignment — restructured**

*fMRI (NSD, THINGS-fMRI):*
- Condition the model on **COCO captions** (NSD) or category labels (THINGS) so the language pathway is engaged. Be explicit in the methods that this tests representational content under language conditioning, not the task the subjects performed.
- **Per-iteration variance partitioning** (Lescroart, Stansbury & Gallant 2015): fit encoding models from iter-1 features and iter-2 features *separately* and compute unique variance explained per iteration across the cortical hierarchy.
- **Predicted dissociation:** iter-1 unique variance peaks in posterior ventral cortex (V1–V4); iter-2 unique variance peaks in anterior ventral cortex (LOC, FFA, anterior IT) and language-adjacent regions.
- **Demote PFC claims.** NSD is passive viewing — PFC has low SNR and is task-dependent. Discuss only as exploratory.
- Report **noise-ceiling-normalized** scores (Schoppe et al. 2016; Lage-Castellanos et al. 2019).
- Run shuffled-caption control C5 — predicted to degrade *specifically* in late-ventral / language-adjacent voxels, not in early visual cortex.

*MEG (THINGS-MEG):*
- This is the right modality for recurrence claims (fMRI cannot resolve timing).
- **Temporal generalization** (King & Dehaene 2014): train classifiers on each model layer/iteration's representation, test against MEG patterns at every time bin.
- **Predicted dissociation:** iter-1 features generalize best to early time bins (~80–150 ms); iter-2 features generalize best to late time bins (~150–300 ms).
- Iteration-scrambling control C6 should abolish this dissociation.

*ECoG / intracranial (if accessible):*
- Berezutskaya et al., Hamilton et al. naturalistic ECoG datasets. Optional but elevates the temporal-precision claim if any are workable.

**B.5 Behavioral alignment**

- **Geirhos model-vs-human benchmark**: error consistency, shape bias, OOD distortion robustness across 17 conditions. Report against C1 and C3 to show gains are *not* explained by compute or LLM-priors-without-recurrence.
- **RT correlation (rebuilt money figure):**
  - Image-averaged, not single-trial.
  - Partial out feedforward difficulty (CNN confidence / ViT entropy / human accuracy) — the third-variable killer.
  - Noise-ceiling-normalized.
  - Specificity contrast: correlation should survive in main model, weaken in C2 (no LLM) and C3 (no recurrence).
  - **Realistic target: r ≈ 0.3–0.4**, not 0.5+. Headline is the dissociation, not the magnitude.
- **Illusions and contextual modulation:** Müller-Lyer, Ebbinghaus, scene-congruency (Bar 2004; Brandman & Peelen 2017). Cheap to run; strong narrative if positive.

**B.6 Mechanism probes**
- **Biased-competition test:** present images with target + distractor; condition LLM with target word; measure whether top-down gates up-weight target features and down-weight distractor features in stage-3/4 activations. If positive, "biased competition" framing is earned. If not, drop it.
- **Gate-magnitude monitoring** across training and across image difficulty.
- **Persistent-token decoding:** can the contents of the 32 tokens at iter-2 be decoded for image content? (sanity check that they carry information).
- **Optional WM-property tests** (only if we want to keep "working memory" framing): capacity-limit test under load (load 1 vs 4 vs 8 items into context), interference test with similar items, content-specific delay decoding across iterations.

---

## 6. Pre-registration and falsifiable claims

To preempt reviewer concerns about HARKing, pre-register before Stage-2 training:

1. **H1 (compositionality specificity).** Main > C1 (FLOP-matched FF) on Winoground/SugarCrepe at p < 0.01 with bootstrap CI. Main > C2 (no LLM) and main > C3 (no recurrence) at p < 0.05.
2. **H2 (NSD dissociation).** Iter-2 unique variance > iter-1 unique variance in pre-defined late-ventral ROI (LOC + anterior IT); reverse pattern in V1–V3. Tested with paired-bootstrap.
3. **H3 (MEG temporal generalization).** Iter-2 features generalize to MEG time bins ≥ 150 ms with higher decoding accuracy than iter-1 features; reverse below 150 ms.
4. **H4 (caption specificity).** C5 (shuffled caption) degrades NSD prediction specifically in language-adjacent ROIs (Fedorenko language network mask) more than in early visual cortex (paired-difference test).
5. **H5 (RT correlation specificity).** Image-averaged, difficulty-partialed RT correlation is significantly larger for main model than for C2 and C3.

Anything else is exploratory.

---

## 7. Risks and mitigations (updated)

| Risk | Mitigation |
|---|---|
| Top-down gates collapse to zero | KL-between-iterations penalty, gate-floor regularizer, gate-magnitude logging |
| Gains within noise on standard benchmarks | Pre-registered hypotheses target compositional + occluded subsets, not standard ones |
| "Just more compute" objection | C1 FLOP-matched control is the answer; run *first*, not last |
| "Just ensembling" objection | C9 stochastic-pass baseline |
| "Just LLM priors leaking in" objection | C3 (LLM-without-recurrence) and C4 (random-LLM-keys) |
| Image-difficulty confounds RT/MEG correlations | Partial out feedforward difficulty; noise-ceiling normalize |
| NSD has no language conditioning natively | Use COCO captions + clearly disclose; lead with MEG and variance partitioning |
| PFC alignment claims weak under passive-viewing fMRI | Demote to exploratory; don't headline |
| Eval-set contamination in LLaVA/Cambrian/ShareGPT4V | Decontaminate every eval before reporting |
| WM framing is decoration | Either drop "working memory" terminology or run capacity/interference probes in §5.B.6 |

---

## 8. What changed from the prior plan

- Reframed every neuroscience term that was decorative; kept the engineering plan intact.
- Promoted the specificity control suite (C1–C9) from "ablations" to **first-class experiments** that determine the headline claim.
- Restructured brain alignment around **per-iteration variance partitioning** (fMRI) and **temporal generalization** (MEG), not penultimate-feature ridge regression.
- Demoted PFC alignment claims; led with ventral cortex + MEG.
- Rebuilt the RT money figure: image-averaged, difficulty-partialed, noise-ceiling-normalized, with specificity contrasts. Realistic target r ≈ 0.3–0.4, not 0.5+.
- Iteration index is now **ordinal**, not millisecond-calibrated.
- Added pre-registration of H1–H5 before Stage-2 training.

---

## 9. Immediate next steps (week-by-week)

**Week 1.** Stand up the architecture. Verify zero-init reproduces vanilla VLM. Build the eval harness for Winoground/SugarCrepe/ImageNet-A *first* — these need to be automated before training.

**Week 2.** Stage-1 alignment training on ShareGPT4V-PT.

**Week 3.** Pre-register H1–H5 in a public repo / OSF. Lock the analysis plan.

**Weeks 3–5.** Stage-2 SFT with recurrence. In parallel: train C1 (FLOP-matched FF) on the same data and budget.

**Weeks 5–6.** Train C2, C3, C4 (cheap given Stage-1 reuse). Run C5–C9 at inference time.

**Week 7.** Brain-alignment pipeline: NSD variance partitioning, THINGS-MEG temporal generalization.

**Week 8.** Behavioral evals: Geirhos suite, RT correlations with partialing, illusions battery.

**Week 9+.** Mechanism probes (§5.B.6), writing.

If any of H1–H5 fails, we learn something real. If all five succeed with the predicted specificity pattern, the paper is hard to dismiss.
