"""Modulation-magnitude analyzer (SPEC_v1.md §5.B.6).

Gates in the top-down adapters are *global* scalars — they don't vary
per image. What does vary per image is the magnitude of the modulation
produced by `tanh(gate) * out_proj(attn(...))`, because the attention
output is image-conditional.

This probe computes that per-image modulation magnitude and provides
hooks for correlating it with image-difficulty proxies (e.g. the
feedforward CNN's softmax entropy, or human RT).

The diagnostic value: in SPEC §7's risk table, "gates collapse to zero"
is the failure mode. Conditionally-on-image-difficulty modulation
patterns are how we know the gates are doing *meaningful* work — if a
trained model's modulation magnitude is constant across images, the
gate may be functioning as a constant bias rather than as a selective
top-down signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import spearmanr


@dataclass
class ModulationStats:
    per_image: dict[str, dict[str, float]]
    summary: dict[str, dict[str, float]]


class ModulationAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, prompt: str) -> list[int]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt_text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def per_image_modulation(
        self,
        image: torch.Tensor,
        prompt: str,
    ) -> dict[str, float]:
        """Returns L2 modulation magnitude at each injection stage."""
        param = next(self.model.parameters())
        device, dtype = param.device, param.dtype

        img = image.to(device=device, dtype=dtype).unsqueeze(0)
        input_ids = self._tokenize(prompt)
        ids = torch.tensor([input_ids], device=device)
        mask = torch.ones_like(ids)

        orig_K = self.model.cfg.num_iterations
        orig_use = self.model.cfg.use_recurrence
        self.model.cfg.num_iterations = 2
        self.model.cfg.use_recurrence = True
        try:
            out = self.model(image=img, text_ids=ids, text_mask=mask)
        finally:
            self.model.cfg.num_iterations = orig_K
            self.model.cfg.use_recurrence = orig_use

        per_iter = out["per_iter_features"]
        result: dict[str, float] = {}
        if len(per_iter) < 2:
            for s in self.model.cfg.injection_stages:
                result[f"stage_{s}"] = 0.0
            return result

        for s in self.model.cfg.injection_stages:
            base = per_iter[0][s].float()
            mod = (per_iter[1][s].float() - base)
            # L2 norm of the modulation tensor, normalized by sqrt(numel)
            # so values are comparable across stages of different sizes.
            n = mod.numel()
            result[f"stage_{s}"] = (mod.pow(2).sum().sqrt() / (n ** 0.5)).item()
        return result

    def analyze(
        self,
        samples: list[dict],
    ) -> ModulationStats:
        """samples: each is {image: Tensor, prompt: str, id: str,
        difficulty: float (optional)}.

        Returns per-image modulation magnitudes plus a summary block. If
        any sample has 'difficulty', also computes the Spearman
        correlation between modulation magnitude and difficulty — the
        diagnostic predicted by the SPEC for confirming the gate is
        doing image-conditional work.
        """
        per_image: dict[str, dict[str, float]] = {}
        stages = sorted(self.model.cfg.injection_stages)
        difficulties: list[float] = []
        per_stage_mods: dict[str, list[float]] = {f"stage_{s}": [] for s in stages}

        for sample in samples:
            mods = self.per_image_modulation(sample["image"], sample["prompt"])
            per_image[sample["id"]] = mods
            if "difficulty" in sample:
                difficulties.append(sample["difficulty"])
                for s in stages:
                    per_stage_mods[f"stage_{s}"].append(mods[f"stage_{s}"])

        summary: dict[str, dict[str, float]] = {}
        for s in stages:
            key = f"stage_{s}"
            vals = [per_image[i][key] for i in per_image]
            summary[key] = {
                "mean": sum(vals) / max(len(vals), 1),
                "min":  min(vals) if vals else 0.0,
                "max":  max(vals) if vals else 0.0,
            }
            if (len(difficulties) >= 3
                    and len(set(difficulties)) > 1
                    and len(set(per_stage_mods[key])) > 1):
                r = spearmanr(per_stage_mods[key], difficulties).correlation
                summary[key]["spearman_r_with_difficulty"] = float(r)
                summary[key]["n_for_correlation"] = len(difficulties)

        return ModulationStats(per_image=per_image, summary=summary)
