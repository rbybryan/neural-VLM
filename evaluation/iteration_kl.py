"""Iteration KL metric for the H5 RT-specificity hypothesis.

Per PREREGISTRATION.md §2 H5, the "iterations-to-converge" metric is
pre-specified as

    KL(softmax(logits_iter1) || softmax(logits_iter2))

evaluated at the answer-token position. Larger KL means iteration 2
substantially changes the model's predictive distribution, which we
interpret as "this image required more processing." That per-image KL
is what gets partial-correlated against human RT (analysis/rt_correlation
.py).

Implementation: we force K=2, run a single forward, and extract iter-1
and iter-2 logits at the continuation-token positions from the model's
per_iter_logits output. KL is computed in fp32 regardless of the model's
working dtype (KL is numerically sensitive to softmax precision).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .scoring import ScoringQuery


class IterationDivergence:
    """Compute KL(iter1 || iter2) per query."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def kl_for_query(self, query: ScoringQuery) -> dict[str, float]:
        """Returns {kl_first_token, kl_mean, kl_max} over continuation positions.

        kl_first_token  — KL at the first continuation token. Most directly
                          maps to "is the model's first decision changed by
                          iteration 2?"
        kl_mean         — average over all continuation positions. Smoother
                          but mixes uninformative late positions.
        kl_max          — peak divergence anywhere in the continuation.
        """
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + query.continuation
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        n_prompt = len(prompt_ids)
        n_cont = len(full_ids) - n_prompt
        if n_cont <= 0:
            return {"kl_first_token": 0.0, "kl_mean": 0.0, "kl_max": 0.0}

        param = next(self.model.parameters())
        device, dtype = param.device, param.dtype

        image = query.image.to(device=device, dtype=dtype).unsqueeze(0)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        orig_K = self.model.cfg.num_iterations
        orig_use = self.model.cfg.use_recurrence
        self.model.cfg.num_iterations = 2
        self.model.cfg.use_recurrence = True
        try:
            out = self.model(image=image, text_ids=input_ids, text_mask=attention_mask)
        finally:
            self.model.cfg.num_iterations = orig_K
            self.model.cfg.use_recurrence = orig_use

        per_iter = out["per_iter_logits"]
        if len(per_iter) < 2:
            return {"kl_first_token": 0.0, "kl_mean": 0.0, "kl_max": 0.0}

        n_vis = per_iter[0].shape[1] - len(full_ids)
        start = n_vis + n_prompt - 1
        end = start + n_cont

        l1 = per_iter[0][0, start:end].float()           # (n_cont, V)
        l2 = per_iter[1][0, start:end].float()
        log_p1 = F.log_softmax(l1, dim=-1)
        log_p2 = F.log_softmax(l2, dim=-1)
        p1 = log_p1.exp()
        kl_per_pos = (p1 * (log_p1 - log_p2)).sum(dim=-1).clamp(min=0.0)

        return {
            "kl_first_token": float(kl_per_pos[0].item()),
            "kl_mean": float(kl_per_pos.mean().item()),
            "kl_max": float(kl_per_pos.max().item()),
        }

    def kl_many(self, queries, progress: bool = False) -> dict[str, dict[str, float]]:
        """Per-query KL metrics, keyed by query_id."""
        if progress:
            try:
                from tqdm import tqdm
                queries = tqdm(queries, desc="iter-KL")
            except ImportError:
                pass
        return {q.query_id: self.kl_for_query(q) for q in queries}

    @staticmethod
    def per_image_metric(
        kl_results: dict[str, dict[str, float]],
        which: str = "kl_first_token",
        group_reducer: str = "mean",
    ) -> dict[str, float]:
        """Aggregate per-query KL into per-group (typically per-image) values.

        For Winoground items there are 4 queries per group; SugarCrepe has
        2 per group. We reduce across queries within a group to get one
        "iterations-to-converge" value per item — that's what the
        analysis/rt_correlation.py partial-correlation expects.
        """
        if group_reducer not in {"mean", "max", "first"}:
            raise ValueError(f"unknown reducer {group_reducer}")
        groups: dict[str, list[float]] = {}
        for query_id, kl_dict in kl_results.items():
            group_id = query_id.split("__", 1)[0]
            groups.setdefault(group_id, []).append(kl_dict[which])
        out: dict[str, float] = {}
        for gid, vals in groups.items():
            if group_reducer == "mean":
                out[gid] = sum(vals) / len(vals)
            elif group_reducer == "max":
                out[gid] = max(vals)
            elif group_reducer == "first":
                out[gid] = vals[0]
        return out
