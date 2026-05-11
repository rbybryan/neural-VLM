"""Training entry point.

Usage (single GPU):
    python scripts/train.py \\
        --config configs/main.yaml \\
        --data-json /path/to/train.json \\
        --image-root /path/to/images \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --decontamination-tsv excluded.tsv \\
        --output-dir outputs/main_K2 \\
        --max-steps 50000 --batch-size 4 --learning-rate 1e-4

Usage (multi-GPU via Accelerate):
    accelerate launch scripts/train.py --config ... [same flags]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (                                                  # noqa: E402
    ConversationDataset,
    VLMCollator,
    build_train_transform,
    load_excluded_set,
)
from scripts.build_main import (                                    # noqa: E402
    apply_control_overrides,
    build_main_model,
    load_config,
)
from training import Trainer, TrainerConfig                         # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--control", default=None,
                        help="apply C1-C9 override from configs/controls.yaml")
    parser.add_argument("--data-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--tokenizer", required=True,
                        help="HF tokenizer model id or local path")
    parser.add_argument("--decontamination-tsv", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--td-lr-mult", type=float, default=7.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mixed-precision", default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--no-grad-checkpoint", action="store_true")

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)

    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.control:
        cfg = apply_control_overrides(cfg, args.control)

    model = build_main_model(cfg, pretrained=not args.no_pretrained)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: total {n_total:,} | trainable {n_train:,} ({100 * n_train / n_total:.2f}%)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    excluded = load_excluded_set(Path(args.decontamination_tsv)) if args.decontamination_tsv else set()
    transform = build_train_transform(image_size=args.image_size)
    dataset = ConversationDataset(
        json_path=args.data_json,
        image_root=args.image_root,
        image_transform=transform,
        excluded_image_paths=excluded,
    )
    print(f"dataset: {len(dataset)} items (excluded {len(excluded)})")

    collator = VLMCollator(tokenizer=tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    tcfg = TrainerConfig(
        learning_rate=args.learning_rate,
        td_lr_mult=args.td_lr_mult,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        use_grad_checkpoint=not args.no_grad_checkpoint,
    )
    trainer = Trainer(model, loader, tcfg)
    trainer.run()


if __name__ == "__main__":
    main()
