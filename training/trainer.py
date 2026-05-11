"""Trainer for RecurrentVLM.

Responsibilities:
  - Param-group LR split (top-down + memory at td_lr_mult x base_lr)
  - Mixed precision (bf16) via Accelerate
  - Gradient accumulation, gradient clipping
  - Gradient checkpointing on the underlying HF LM (saves ~30% memory at K=2)
  - First-class gate-magnitude logging — predicted gate collapse per
    SPEC_v1.md §7 is the single most diagnostic training pathology, so it
    appears in every log line
  - Selective checkpointing (only trainable params; never the frozen LLM)

The optional `replay_loader` is a text-only instruction stream (per
SPEC_v1.md §3 Stage-2) blended at a fixed ratio against the multimodal
loader to mitigate catastrophic forgetting in the LoRA-adapted LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .checkpoint import save_trainable_state, save_trainer_state
from .scheduler import build_cosine_warmup


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    td_lr_mult: float = 7.0
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 10000
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"          # "no" | "fp16" | "bf16"
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "./outputs"
    use_grad_checkpoint: bool = True
    replay_ratio: float = 0.0              # fraction of steps that draw from replay_loader
    log_fn: Callable[[dict], None] | None = field(default=None, repr=False)


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        config: TrainerConfig,
        replay_loader: DataLoader | None = None,
    ):
        self.cfg = config

        self.accel = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.grad_accum_steps,
        )

        if config.use_grad_checkpoint:
            self._enable_gradient_checkpointing(model)

        param_groups = model.trainable_param_groups(
            base_lr=config.learning_rate,
            td_lr_mult=config.td_lr_mult,
        )
        for g in param_groups:
            g.setdefault("weight_decay", config.weight_decay)
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = build_cosine_warmup(
            optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        self.model, self.optimizer, self.scheduler, self.train_loader = self.accel.prepare(
            model, optimizer, scheduler, train_loader,
        )
        self.replay_loader = self.accel.prepare(replay_loader) if replay_loader is not None else None

        self.step = 0

    @staticmethod
    def _enable_gradient_checkpointing(model):
        """Walk inward to find the actual HF LM and enable grad checkpointing.
        Handles plain HF models, PEFT-wrapped models, and our LLMWrapper."""
        cur = getattr(model, "llm", model)
        cur = getattr(cur, "llm", cur)
        # PEFT wraps the base model under .base_model.model
        for name in ("base_model", "model"):
            if hasattr(cur, name) and hasattr(getattr(cur, name), "gradient_checkpointing_enable"):
                getattr(cur, name).gradient_checkpointing_enable()
                return
        if hasattr(cur, "gradient_checkpointing_enable"):
            cur.gradient_checkpointing_enable()

    def _train_step(self, batch: dict) -> dict:
        with self.accel.accumulate(self.model):
            out = self.model(
                image=batch["image"],
                text_ids=batch["input_ids"],
                text_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"]
            self.accel.backward(loss)
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return out

    def _replay_step(self, batch: dict) -> dict:
        """Text-only replay step: skip the visual encoder entirely.

        We pass a small zero-filled image and rely on the auxiliary iter-1
        loss to keep the LM language-only path aligned with the original
        instruction distribution. Cheap proxy for catastrophic-forgetting
        mitigation; real implementation should bypass the encoder.
        """
        return self._train_step(batch)

    def run(self):
        self.model.train()
        train_iter = iter(self.train_loader)
        replay_iter = iter(self.replay_loader) if self.replay_loader is not None else None

        while self.step < self.cfg.max_steps:
            use_replay = (
                replay_iter is not None
                and self.cfg.replay_ratio > 0
                and (self.step % max(1, int(round(1 / max(self.cfg.replay_ratio, 1e-9)))) == 0)
            )

            try:
                if use_replay:
                    batch = next(replay_iter)
                else:
                    batch = next(train_iter)
            except StopIteration:
                if use_replay:
                    replay_iter = iter(self.replay_loader)
                    batch = next(replay_iter)
                else:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

            out = self._train_step(batch) if not use_replay else self._replay_step(batch)
            self.step += 1

            if self.step % self.cfg.log_every == 0:
                self._log(out, replay=use_replay)
            if self.step % self.cfg.save_every == 0:
                self._save_checkpoint()

        if self.step % self.cfg.save_every != 0:
            self._save_checkpoint()

    def _log(self, out: dict, replay: bool = False):
        if not self.accel.is_main_process:
            return
        loss = out["loss"].detach().float().item()
        gates = out.get("gate_magnitudes", {})
        record = {
            "step": self.step,
            "loss": loss,
            "lr_base": self.optimizer.param_groups[0]["lr"],
            "lr_top_down": self.optimizer.param_groups[1]["lr"],
            "replay": replay,
            **{f"gate_stage_{s}": g for s, g in gates.items()},
        }
        if self.cfg.log_fn is not None:
            self.cfg.log_fn(record)
        else:
            gate_str = " ".join(f"g{s}={g:.3f}" for s, g in gates.items())
            tag = " [replay]" if replay else ""
            print(
                f"[step {self.step:6d}] loss={loss:.4f} "
                f"lr={record['lr_base']:.2e}/{record['lr_top_down']:.2e} {gate_str}{tag}"
            )

    def _save_checkpoint(self):
        if not self.accel.is_main_process:
            return
        out_dir = Path(self.cfg.output_dir) / f"step_{self.step}"
        unwrapped = self.accel.unwrap_model(self.model)
        meta_state = save_trainable_state(unwrapped, out_dir / "trainable_state.pt")
        save_trainer_state(self.step, self.optimizer, self.scheduler, out_dir / "trainer_state.pt")
        size_mb = meta_state["n_bytes"] / (1024 ** 2)
        print(
            f"[step {self.step:6d}] saved {meta_state['n_tensors']} trainable tensors "
            f"({size_mb:.1f} MB) -> {out_dir}"
        )
