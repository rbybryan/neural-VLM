# VLM-custom

Language-conditioned recurrent modulation in a VLM. See `SPEC_v1.md` for the
research plan, `PREREGISTRATION.md` for the locked confirmatory analyses,
`model/` for the architecture skeleton, and `analysis/` for the brain- and
behavior-alignment pipelines.

## Layout

```
SPEC_v1.md              # research plan
PREREGISTRATION.md      # H1-H5 locked before Stage-2 training
model/
  top_down_adapter.py   # cross-attention adapter, zero-init gated output
  memory.py             # persistent context tokens (read/write across iterations)
  encoder.py            # ConvNeXt-V2 wrapper with stage-level access + forward_from
  llm.py                # LLM wrapper (inputs_embeds composition) + apply_lora
  connector.py          # MLP connector (CNN tokens -> LLM dim)
  recurrent_vlm.py      # K-iteration orchestrator + control toggles for C1-C9
data/
  conversation.py       # LLaVA-style schema + JSON-backed Dataset
  transforms.py         # ImageNet-normalized image transforms
  collator.py           # tokenization + supervision masking + padding
  decontaminate.py      # PHash decontamination (CLI + library)
training/
  trainer.py            # Accelerate-based loop with param-group LR split
  scheduler.py          # cosine-with-warmup
  checkpoint.py         # selective save: trainable params only (~50 MB vs ~14 GB)
evaluation/
  scoring.py            # Scorer (log-likelihood of continuation under image+prompt)
  benchmarks.py         # Winoground + BinaryDiscrimination (SugarCrepe/ARO format)
  k_sweep.py            # vary K at inference; produces per-K accuracy curves
  iteration_kl.py       # KL(iter1 || iter2) — the H5 iterations-to-converge metric
  specificity_suite.py  # cross-control driver (main vs C1-C9) + contrast table
  probes/
    biased_competition.py  # SPEC §5.B.6: target vs distractor region modulation
    modulation_analyzer.py # per-image modulation magnitude, difficulty correlation
    token_decoder.py       # linear probe on persistent context tokens
  loaders/
    winoground.py          # HF facebook/winoground -> our JSON manifest format
    sugarcrepe.py          # HF SugarCrepe + local COCO val2017 -> our format
analysis/
  variance_partition.py # NSD per-iteration unique-variance (Lescroart 2015)
  temporal_generalization.py  # THINGS-MEG decoding (King & Dehaene 2014)
  rt_correlation.py     # image-averaged, partialed, noise-ceiling RT correlation
configs/
  main_dev.yaml         # tier-0: Qwen2.5-0.5B (engineering / debug only)
  main.yaml             # tier-1: Qwen2.5-1.5B (default; controls C1-C9, H2-H5)
  main_pub.yaml         # tier-2: Qwen2.5-7B  (publication; main + C1 + C2, H1)
  controls.yaml         # specificity suite C1-C9 (deltas)
scripts/
  setup_env.sh          # one-shot env install (idempotent)
  build_main.py         # build_main_model() + smoke test (--smoke)
  test_data_pipeline.py # end-to-end data pipeline smoke test
  train.py              # training entry (single-GPU or accelerate launch)
  test_training.py      # 10-step training smoke test
  test_eval.py          # eval harness smoke test (Winoground + SugarCrepe + K-sweep)
  verify_real_assembly.py  # real-weight + dtype/device check
```

## Status

Skeleton complete. End-to-end forward + backward verified on CPU via
`scripts/build_main.py --smoke` using ConvNeXt-V2-atto + a 4-layer Qwen2 LM
constructed from config (no weight downloads).

Verified properties:
- Zero-init equivalence: at initialization, K=2 forward exactly equals K=1
  forward (max |Δlogits| = 0). The model degrades gracefully into a vanilla
  VLM if recurrence training fails.
- Gradient unlocking: at step 0 only the adapter gate has nonzero gradient
  (tanh(0)=0 blocks modulation, but out_proj is normally initialized so the
  gate's own gradient is nonzero). At step 1, gate ≈ 1e-3, and `out_proj`,
  `q_proj`, and the memory bank all unlock with nonzero gradients.
- Encoder re-forwarding: when the top-down adapter modulates a non-final
  CNN stage, modulations propagate through subsequent stages via
  `ConvNeXtV2Encoder.forward_from()` so the LLM sees the downstream effect.

Not yet built: real-weight loading (Qwen2.5-7B + ConvNeXt-V2-Base on a GPU
node), mechanism probes (§5.B.6 of `SPEC_v1.md`).

## Smoke tests

```bash
conda activate vlm_custom
python scripts/build_main.py --smoke           # model wiring + zero-init guarantee
python scripts/test_data_pipeline.py           # data layer + decontamination + e2e
python scripts/test_training.py                # 10-step training loop on synthetic data
python scripts/test_eval.py                    # eval harness + K-sweep on synthetic data
python scripts/test_specificity_suite.py       # cross-control driver + iteration-KL metric
python scripts/test_probes.py                  # §5.B.6 mechanism probes (biased-competition, modulation, token decoder)
python scripts/test_loaders.py                 # HF row -> manifest transformers (Winoground + SugarCrepe)
python scripts/verify_real_assembly.py         # real Qwen2.5 + ConvNeXt weights (CPU fp32)
```

All five should end with `PASSED`.

## Training (single GPU)

Three tiers — see `docs/tier_transition_checklist.md` for the dev->pub
transition protocol.

```bash
conda activate vlm_custom

# tier-0 (Qwen2.5-0.5B): engineering / debug only
python scripts/train.py --config configs/main_dev.yaml \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct ...

# tier-1 (Qwen2.5-1.5B): default; all C1-C9 controls + H2/H3/H4/H5 numbers
python scripts/train.py --config configs/main.yaml \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct ...

# tier-2 (Qwen2.5-7B): publication-tier; main + C1 + C2 + H1 Winoground
python scripts/train.py --config configs/main_pub.yaml \
    --tokenizer Qwen/Qwen2.5-7B-Instruct ...
```

Full flag set:

```bash
python scripts/train.py \
    --config configs/main.yaml \
    --data-json /path/to/llava_train.json \
    --image-root /path/to/llava_images \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
    --decontamination-tsv excluded.tsv \
    --output-dir outputs/main_K2 \
    --max-steps 50000 --batch-size 4 --learning-rate 1e-4
```

## Training (multi-GPU)

```bash
accelerate config   # one-time, configure FSDP / DeepSpeed
accelerate launch scripts/train.py [same flags]
```

Each control model uses the same script with `--control C1` etc.

## Preparing eval benchmarks (production)

```bash
# Winoground: gated on HF — accept terms at
# https://huggingface.co/datasets/facebook/winoground and supply a token
HF_TOKEN=hf_xxx python scripts/prepare_eval_data.py winoground \
    --output-dir data/eval/winoground

# SugarCrepe: requires MS-COCO val2017 downloaded separately
# (~1 GB from http://images.cocodataset.org/zips/val2017.zip)
python scripts/prepare_eval_data.py sugarcrepe \
    --coco-val-dir /path/to/coco/val2017 \
    --output-dir data/eval/sugarcrepe
```

Each emits a JSON manifest in the shape consumed by
`evaluation.benchmarks.{Winoground, BinaryDiscrimination}`.

## Decontamination workflow (production)

```bash
# Once: index every eval image (Winoground, SugarCrepe, NSD, MMBench, ...)
python -m data.decontaminate build-index \
    --eval-dirs /path/to/eval1 /path/to/eval2 \
    --output decontamination_index.json

# Per training corpus: filter
ls /path/to/train_images/*.jpg > train_list.txt
python -m data.decontaminate filter \
    --train-list train_list.txt \
    --index decontamination_index.json \
    --output excluded.tsv

# In code:
from data import ConversationDataset, load_excluded_set
dataset = ConversationDataset(
    json_path="train.json",
    image_root="/path/to/train_images",
    image_transform=build_train_transform(224),
    excluded_image_paths=load_excluded_set("excluded.tsv"),
)
```
