# Training and evaluation

These commands require separately obtained models and datasets. For the download-free CPU checks, see the [README](../README.md). Run all commands from the repository root.

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
accelerate launch scripts/train.py --help
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
