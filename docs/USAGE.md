# Training and evaluation

These commands require separately obtained models and datasets. For the download-free CPU checks, see the [README](../README.md). Run all commands from the repository root.

## Training (single GPU)

The available configurations use three language-model sizes:

| Config | Tokenizer/model |
|---|---|
| `configs/main_dev.yaml` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `configs/main.yaml` (default) | `Qwen/Qwen2.5-1.5B-Instruct` |
| `configs/main_pub.yaml` | `Qwen/Qwen2.5-7B-Instruct` |

The `pub` filename comes from the original plan; no trained 7B results are reported.
See the [size-transition notes](tier_transition_checklist.md) for the planned checks
when changing model size. Real-data training still needs validation.

For the default configuration, replace the data paths and provide the exclusion
list produced by the decontamination step below:

```bash
conda activate vlm_custom
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

Control overrides take the full key from [controls.yaml](../configs/controls.yaml),
for example `--control C3_llm_no_recurrence`. The command does not accept the shorthand
`--control C3`. Read the [known control gaps](RESEARCH_STATUS.md) before interpreting
any comparison.

## Preparing evaluation data

```bash
# Winoground: gated on HF — accept terms at
# https://huggingface.co/datasets/facebook/winoground and supply a token
python scripts/prepare_eval_data.py winoground \
    --output-dir data/eval/winoground

# SugarCrepe: requires MS-COCO val2017 downloaded separately
# (~1 GB from http://images.cocodataset.org/zips/val2017.zip)
python scripts/prepare_eval_data.py sugarcrepe \
    --coco-val-dir /path/to/coco/val2017 \
    --output-dir data/eval/sugarcrepe
```

Each emits a JSON manifest in the shape consumed by
`evaluation.benchmarks.{Winoground, BinaryDiscrimination}`.

## Filtering near-duplicate images

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

```

The same exclusion list can be passed to the dataset directly:

```python
from pathlib import Path
from data import ConversationDataset, build_train_transform, load_excluded_set
dataset = ConversationDataset(
    json_path="train.json",
    image_root="/path/to/train_images",
    image_transform=build_train_transform(224),
    excluded_image_paths=load_excluded_set(Path("excluded.tsv")),
)
```
