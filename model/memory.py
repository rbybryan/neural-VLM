"""Persistent context-token bank.

A small set of learnable tokens that participate in every top-down injection.
Framed in the spec as 'persistent context tokens' rather than 'PFC working
memory' unless the §5.B.6 capacity / interference probes are run and pass.

read_write performs a residual cross-attention update of the tokens against
the current LLM context, so the tokens accumulate task-relevant state across
recurrent iterations without growing unbounded.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PersistentMemoryBank(nn.Module):
    def __init__(
        self,
        num_tokens: int = 32,
        dim: int = 4096,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim

        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.update_gate = nn.Parameter(torch.zeros(1))

    def initial(self, batch_size: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def read_write(self, tokens: torch.Tensor, llm_ctx: torch.Tensor) -> torch.Tensor:
        q = self.norm(tokens)
        # LM hidden states may arrive in bf16 (when the LM is loaded in bf16
        # for GPU training) while the memory bank's own params are fp32 (or
        # vice versa). Cast incoming context to match the memory's dtype so
        # the MultiheadAttention kernel doesn't reject the mixed inputs.
        llm_ctx = llm_ctx.to(q.dtype)
        update, _ = self.attn(q, llm_ctx, llm_ctx, need_weights=False)
        return tokens + torch.tanh(self.update_gate) * update

    def lesion(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_tokens, self.dim, device=device)
