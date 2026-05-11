"""ConvNeXt-V2 encoder with stage-level access for top-down injection.

Exposes:
  forward(image) -> list[stage_features]    full forward (stem + 4 stages)
  forward_from(stage_idx, modulated_x) -> list[stage_features at stage_idx..last]
    re-runs the encoder *from* a modulated stage. Used by RecurrentVLM when
    injection happens at a non-final stage: the modulation at stage k must
    propagate through stages k+1..last before the LLM sees it.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class ConvNeXtV2Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnextv2_base",
        pretrained: bool = True,
        freeze_stages: tuple[int, ...] = (),
    ):
        super().__init__()
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.stem = backbone.stem
        self.stages = backbone.stages

        info = timm.create_model(model_name, pretrained=False, features_only=True)
        self.channels: list[int] = list(info.feature_info.channels())
        del info

        self.num_stages = len(self.stages)
        self._freeze(freeze_stages)

    def _freeze(self, stages: tuple[int, ...]):
        if 0 in stages:
            for p in self.stem.parameters():
                p.requires_grad = False
        for s in stages:
            if 0 <= s < self.num_stages:
                for p in self.stages[s].parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def forward_from(self, stage_idx: int, x_at_stage: torch.Tensor) -> list[torch.Tensor]:
        """Re-run from stage stage_idx (modulated input) onwards.
        Returns features at stages [stage_idx, stage_idx+1, ..., last]."""
        feats = [x_at_stage]
        cur = x_at_stage
        for stage in self.stages[stage_idx + 1:]:
            cur = stage(cur)
            feats.append(cur)
        return feats
