"""Cosine learning-rate schedule with linear warmup."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.0):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return LambdaLR(optimizer, lr_lambda)
