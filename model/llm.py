"""LLM wrapper for RecurrentVLM.

Takes (vis_tokens, input_ids, attention_mask) and returns a dict with logits
and per-layer hidden states. Visual tokens are spliced into the text token
embedding stream via inputs_embeds (LLaVA-style).

We wrap the *language model* component only; the parent VLM's own visual
encoder is unused. This keeps Qwen2.5's text-domain knowledge but lets our
ConvNeXt-V2 + top-down loop be the visual front-end.
"""

from __future__ import annotations

import torch
from torch import nn


class LLMWrapper(nn.Module):
    """Thin adapter around a HF causal LM exposing the interface RecurrentVLM expects."""

    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm
        self.hidden_size = int(llm.config.hidden_size)

    def forward(
        self,
        vis_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> dict:
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # The visual tower may run in fp32 (CPU / safe default) while the
        # LM was loaded in bf16/fp16. Cast vis_tokens to the LM's working
        # dtype before splicing so the LoRA-wrapped Linear layers accept
        # the mixed sequence.
        vis_tokens = vis_tokens.to(text_embeds.dtype)
        inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        vis_mask = torch.ones(
            vis_tokens.shape[:2], device=vis_tokens.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return {
            "logits": out.logits,
            "hidden_states": out.hidden_states,
        }


def apply_lora(
    llm: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )
    return get_peft_model(llm, config)
