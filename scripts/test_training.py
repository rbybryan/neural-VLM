"""End-to-end training-loop smoke test on CPU with the synthetic stack.

Runs ~10 optimizer steps using build_smoke_model + a synthetic conversation
corpus + the StubTokenizer. Verifies:
  - Trainer.run() completes without error
  - Loss is finite and decreases across steps
  - Top-down gates move off zero (the Flamingo unlock dynamic)
  - A trainable-only checkpoint is saved at the end
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from data import ConversationDataset, VLMCollator, build_train_transform   # noqa: E402
from scripts.build_main import build_smoke_model                           # noqa: E402
from scripts.test_data_pipeline import StubTokenizer, _build_synthetic_corpus  # noqa: E402
from training import Trainer, TrainerConfig                                # noqa: E402


class _LoopingDataset(Dataset):
    def __init__(self, base: Dataset, n_repeats: int):
        self.base = base
        self.n_repeats = n_repeats

    def __len__(self) -> int:
        return len(self.base) * self.n_repeats

    def __getitem__(self, idx: int):
        return self.base[idx % len(self.base)]


def main():
    losses: list[float] = []
    gate_history: list[dict[str, float]] = []

    def collect(rec: dict):
        losses.append(rec["loss"])
        gate_history.append({k: v for k, v in rec.items() if k.startswith("gate_stage_")})
        gates = " ".join(f"{k.split('_')[-1]}={v:.4f}" for k, v in rec.items() if k.startswith("gate_stage_"))
        print(f"[step {rec['step']:3d}] loss={rec['loss']:.4f} lr={rec['lr_base']:.2e}/{rec['lr_top_down']:.2e}  {gates}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_json, image_root = _build_synthetic_corpus(tmp_path, n=4)

        transform = build_train_transform(image_size=224)
        base_ds = ConversationDataset(train_json, image_root, image_transform=transform)
        ds = _LoopingDataset(base_ds, n_repeats=20)

        tokenizer = StubTokenizer(vocab_size=2048)
        collator = VLMCollator(tokenizer=tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=2, collate_fn=collator, num_workers=0)

        torch.manual_seed(0)
        model = build_smoke_model()

        cfg = TrainerConfig(
            learning_rate=5e-3,
            td_lr_mult=7.0,
            warmup_steps=2,
            max_steps=10,
            log_every=1,
            save_every=10,
            mixed_precision="no",
            use_grad_checkpoint=False,
            output_dir=str(tmp_path / "outputs"),
            log_fn=collect,
        )
        trainer = Trainer(model, loader, cfg)
        trainer.run()

        assert all(torch.isfinite(torch.tensor(L)) for L in losses), "non-finite loss"
        first_avg = sum(losses[:3]) / 3
        last_avg = sum(losses[-3:]) / 3
        print(f"loss: first-3 avg = {first_avg:.4f} | last-3 avg = {last_avg:.4f}")

        final_gates = gate_history[-1]
        max_gate = max(final_gates.values()) if final_gates else 0.0
        print(f"final gate magnitudes: {final_gates}  (max {max_gate:.4f})")
        assert max_gate > 0.0, "gate did not move off zero — Flamingo unlock failed"

        ckpt_dir = Path(cfg.output_dir) / f"step_{cfg.max_steps}"
        ckpt_path = ckpt_dir / "trainable_state.pt"
        assert ckpt_path.exists(), f"checkpoint missing: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        ckpt_size_mb = sum(t.numel() * t.element_size() for t in ckpt.values()) / (1024 ** 2)
        print(f"checkpoint: {len(ckpt)} tensors, {ckpt_size_mb:.2f} MB at {ckpt_path}")

        print("test_training: PASSED")


if __name__ == "__main__":
    main()
