"""Log-likelihood scorer for VLM evaluation.

For multiple-choice / paired-discrimination benchmarks (Winoground,
SugarCrepe, ARO, MMMU), the standard evaluation pattern is:

  for each (image, prompt, continuation) triple:
      score = sum_i log P(continuation_i | image, prompt, continuation_<i)
  pick the candidate with the highest score.

This avoids the brittleness of free-form generation + answer parsing and
is what every modern VLM eval harness (lmms-eval, OpenCompass, etc.) uses
for discrimination tasks. It's also deterministic — critical for the
specificity-contrast experiments where we need bit-reproducible per-query
scores across the main model and its controls C1-C9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


@dataclass
class ScoringQuery:
    """A single (image, prompt, continuation) triple to be scored.

    query_id is unique within a benchmark and is the key under which the
    score is stored. group_id identifies the logical "item" that a query
    belongs to (e.g., a Winoground item with 4 underlying queries).
    """
    image: torch.Tensor                                  # (3, H, W) pre-transformed
    prompt: str
    continuation: str
    query_id: str
    group_id: str = ""
    metadata: dict = field(default_factory=dict)


class Scorer:
    def __init__(self, model, tokenizer, length_normalize: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.length_normalize = length_normalize

    @torch.no_grad()
    def score(self, query: ScoringQuery) -> float:
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
            return 0.0

        param = next(self.model.parameters())
        device, dtype = param.device, param.dtype

        image = query.image.to(device=device, dtype=dtype).unsqueeze(0)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        out = self.model(image=image, text_ids=input_ids, text_mask=attention_mask)
        logits = out["logits"]                            # (1, N_vis + len(full_ids), V)

        n_vis = logits.shape[1] - len(full_ids)
        # logits at position (n_vis + n_prompt - 1) predict the first
        # continuation token. There are n_cont such positions.
        start = n_vis + n_prompt - 1
        relevant = logits[0, start: start + n_cont]       # (n_cont, V)
        cont_ids = torch.tensor(full_ids[n_prompt:], dtype=torch.long, device=device)

        log_probs = F.log_softmax(relevant.float(), dim=-1)
        token_lp = log_probs.gather(1, cont_ids.unsqueeze(-1)).squeeze(-1)
        total = token_lp.sum().item()

        if self.length_normalize:
            total = total / max(n_cont, 1)
        return total

    def score_many(self, queries, progress: bool = False) -> dict[str, float]:
        if progress:
            try:
                from tqdm import tqdm
                queries = tqdm(queries, desc="scoring")
            except ImportError:
                pass
        return {q.query_id: self.score(q) for q in queries}
