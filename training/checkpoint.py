"""Selective checkpointing: save only trainable params + trainer state.

A vanilla state_dict on a Stage-2 model is ~14 GB because the frozen Qwen2.5
LM weights round-trip on every checkpoint. Trainable-only checkpoints are
~50-100 MB (LoRA adapters + connector + top-down adapters + memory tokens +
ConvNeXt stages 3-4) and load on top of the frozen backbones at resume.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_trainable_state(model, path: Path | str) -> dict[str, int]:
    """Save only requires_grad=True params. Returns {n_tensors, n_bytes}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    full = model.state_dict()
    state = {n: full[n].detach().cpu() for n in trainable_names if n in full}
    torch.save(state, path)
    n_bytes = sum(t.numel() * t.element_size() for t in state.values())
    return {"n_tensors": len(state), "n_bytes": n_bytes}


def load_trainable_state(model, path: Path | str, strict: bool = False):
    """Load trainable params on top of the (already-loaded) frozen backbones."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    return model.load_state_dict(state, strict=strict)


def save_trainer_state(step: int, optimizer, scheduler, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_trainer_state(path: Path | str, optimizer, scheduler) -> int:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    return int(blob["step"])
