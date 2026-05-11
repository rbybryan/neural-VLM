"""End-to-end smoke test for the data pipeline.

Creates a temp dir with a few synthetic images and a LLaVA-style JSON, runs
them through ConversationDataset -> VLMCollator -> the smoke model, and
verifies shapes + a finite loss. Also exercises the perceptual-hash
decontamination path.

Uses a stub tokenizer (no HF download). The stub mimics the small subset of
the HF tokenizer interface that VLMCollator depends on:
  - apply_chat_template(messages, tokenize=False, add_generation_prompt=...)
  - __call__(text, add_special_tokens=False) returning .input_ids
  - pad_token_id
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (                                          # noqa: E402
    ConversationDataset,
    VLMCollator,
    build_eval_index,
    build_train_transform,
    find_decontamination_matches,
)
from scripts.build_main import build_smoke_model            # noqa: E402


@dataclass
class _StubEnc:
    input_ids: list[int]


class StubTokenizer:
    """Minimal HF-compatible tokenizer for smoke tests.

    Hash-based char-to-id mapping into a small vocab. Not a real tokenizer;
    its only job is to produce reasonable token IDs and an HF-shaped chat
    template so VLMCollator can be exercised end-to-end on CPU without any
    model downloads.
    """

    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def apply_chat_template(self, messages, tokenize: bool = False,
                            add_generation_prompt: bool = False) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>{m['content']}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = False) -> _StubEnc:
        ids = [(hash(c) & 0x7FFFFFFF) % self.vocab_size for c in text[:512]]
        if not ids:
            ids = [self.pad_token_id]
        return _StubEnc(input_ids=ids)


def _write_structured_image(path: Path, seed: int, size: int = 224):
    """Make a deterministic structured image PHash can distinguish from
    other seeds (random RGB noise has wide PHash diversity)."""
    rng = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size, size, 3, generator=rng) * 255).to(torch.uint8).numpy()
    Image.fromarray(arr, mode="RGB").save(path, quality=90)


def _build_synthetic_corpus(root: Path, n: int = 4) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        rel = f"img_{i}.jpg"
        _write_structured_image(image_dir / rel, seed=1000 + i)
        items.append({
            "image": rel,
            "conversations": [
                {"from": "human", "value": "<image>\nWhat does this image show?"},
                {"from": "gpt", "value": f"Synthetic noise pattern number {i}."},
            ],
        })
    json_path = root / "train.json"
    with open(json_path, "w") as f:
        json.dump(items, f)
    return json_path, image_dir


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_json, image_root = _build_synthetic_corpus(tmp_path, n=4)
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        # Eval image with the same seed as train img_0 -> exact PHash match.
        _write_structured_image(eval_dir / "duplicate_of_train_img_0.jpg", seed=1000)
        # An unrelated eval image with a different seed.
        _write_structured_image(eval_dir / "unrelated.jpg", seed=99999)

        # 1. Decontamination pipeline
        eval_index = build_eval_index([eval_dir])
        train_paths = sorted(image_root.iterdir())
        matches = find_decontamination_matches(train_paths, eval_index, max_distance=4)
        print(f"decontamination: indexed {len(eval_index)} eval | flagged {len(matches)} train")
        for tp, ep, d in matches:
            print(f"  flagged: {Path(tp).name}  ~  {Path(ep).name}  (dist {d})")
        excluded = {tp for tp, _, _ in matches}

        # 2. Dataset + collator
        transform = build_train_transform(image_size=224)
        dataset = ConversationDataset(
            json_path=train_json,
            image_root=image_root,
            image_transform=transform,
            excluded_image_paths=excluded,
        )
        print(f"dataset: {len(dataset)} items (after decontamination)")

        tokenizer = StubTokenizer(vocab_size=2048)
        collator = VLMCollator(tokenizer=tokenizer, max_length=128)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

        batch = next(iter(loader))
        print("batch shapes:")
        for k, v in batch.items():
            print(f"  {k:<14} {tuple(v.shape)}  dtype={v.dtype}")

        n_supervised = (batch["labels"] != -100).sum().item()
        n_total = batch["labels"].numel()
        print(f"supervision: {n_supervised}/{n_total} text tokens get loss "
              f"(prompt + padding masked)")
        assert n_supervised > 0, "no labels survived masking"

        # 3. End-to-end through the smoke model
        torch.manual_seed(0)
        model = build_smoke_model().eval()
        with torch.no_grad():
            out = model(
                image=batch["image"],
                text_ids=batch["input_ids"],
                text_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        print(f"model: logits {tuple(out['logits'].shape)} | loss {out['loss'].item():.4f}")
        assert torch.isfinite(out["loss"]), "non-finite loss"

        # 4. Backward
        model.train()
        out = model(
            image=batch["image"],
            text_ids=batch["input_ids"],
            text_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out["loss"].backward()
        n_nonzero = sum(1 for p in model.parameters()
                        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"backward: {n_nonzero}/{n_train} trainable params have nonzero grad")

        print("test_data_pipeline: PASSED")


if __name__ == "__main__":
    main()
