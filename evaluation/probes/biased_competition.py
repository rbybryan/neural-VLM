"""Biased-competition test (SPEC_v1.md §5.B.6).

The "biased competition" framing (Desimone & Duncan 1995) claims that
top-down attention selectively *amplifies* features matching the
attentional template and *suppresses* features matching distractors.

Our test:
  - Take an image with a known target region and distractor region
    (mask coordinates in stage-3/4 feature-map space).
  - Run the model under prompt P_T cuing the target, then under prompt
    P_D cuing the distractor.
  - For each (prompt, region) combination, measure the L2 norm of the
    top-down modulation (iter-2 features minus iter-1 features) inside
    that region.

  Denote:
    T_T = modulation magnitude in target region when cued for target
    T_D = modulation magnitude in distractor region when cued for target
    D_T = modulation magnitude in target region when cued for distractor
    D_D = modulation magnitude in distractor region when cued for distractor

Biased competition predicts:
    T_T > T_D  AND  D_D > D_T
That is, the modulation is *spatially selective for the cued region.

Operational score:
    biased_competition_score = (T_T - T_D) + (D_D - D_T)
Score > 0 supports the framing; ~0 or negative means the gate is not
selecting target features (per SPEC §1 framing discipline, "biased
competition" should be dropped from the paper if the score is null).

For the score to be meaningful, the model must be trained (gates > 0).
At init this probe correctly returns ~0 across the board.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BiasedCompetitionResult:
    T_T: float
    T_D: float
    D_T: float
    D_D: float
    biased_competition_score: float
    supports_biased_competition: bool


class BiasedCompetitionProbe:
    def __init__(self, model, tokenizer, stage_idx: int = 3):
        """stage_idx is which encoder stage to read modulation from. Default
        is the deepest injected stage (3 in our ConvNeXt-V2-Base setup)."""
        self.model = model
        self.tokenizer = tokenizer
        self.stage_idx = stage_idx

    def _tokenize(self, prompt: str) -> list[int]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt_text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def _modulation_in_regions(
        self,
        image: torch.Tensor,
        prompt: str,
        target_region: torch.Tensor,
        distractor_region: torch.Tensor,
    ) -> tuple[float, float]:
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
        if len(per_iter) < 2:
            return 0.0, 0.0

        baseline = per_iter[0][self.stage_idx]
        modulated = per_iter[1][self.stage_idx]
        mod = modulated - baseline                          # (1, C, H', W')
        mod_norm = mod[0].float().pow(2).sum(dim=0).sqrt()   # (H', W')

        if target_region.shape != mod_norm.shape:
            raise ValueError(
                f"target_region shape {target_region.shape} != "
                f"feature-map shape {mod_norm.shape}"
            )
        t_mag = mod_norm[target_region].mean().item()
        d_mag = mod_norm[distractor_region].mean().item()
        return t_mag, d_mag

    def measure(
        self,
        image: torch.Tensor,
        target_prompt: str,
        distractor_prompt: str,
        target_region: torch.Tensor,
        distractor_region: torch.Tensor,
    ) -> BiasedCompetitionResult:
        T_T, T_D = self._modulation_in_regions(
            image, target_prompt, target_region, distractor_region,
        )
        D_T, D_D = self._modulation_in_regions(
            image, distractor_prompt, target_region, distractor_region,
        )
        score = (T_T - T_D) + (D_D - D_T)
        return BiasedCompetitionResult(
            T_T=T_T, T_D=T_D, D_T=D_T, D_D=D_D,
            biased_competition_score=score,
            supports_biased_competition=score > 0,
        )

    def measure_many(
        self,
        trials: list[dict],
    ) -> list[BiasedCompetitionResult]:
        """trials: each is a dict with keys image, target_prompt,
        distractor_prompt, target_region, distractor_region."""
        return [
            self.measure(
                t["image"],
                t["target_prompt"],
                t["distractor_prompt"],
                t["target_region"],
                t["distractor_region"],
            )
            for t in trials
        ]
