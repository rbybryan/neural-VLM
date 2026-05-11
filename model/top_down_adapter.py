"""Top-down cross-attention adapter with zero-init gated output.

Queries come from CNN feature maps at a given stage; keys/values come from a
fused stream of LLM hidden states + persistent context tokens. The output is
gated by a learnable scalar initialized at zero, so iteration 1 reproduces the
baseline VLM exactly. The gate becomes informative only after training pulls
it away from zero.

Shape conventions
-----------------
- vis_feat:  (B, C, H, W)  CNN feature map at injection stage
- ctx:       (B, T, D_ctx) LLM hidden states + memory tokens, concatenated
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class TopDownAdapter(nn.Module):
    def __init__(
        self,
        vis_channels: int,
        ctx_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Conv2d(vis_channels, inner, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.v_proj = nn.Linear(ctx_dim, inner, bias=False)
        self.out_proj = nn.Conv2d(inner, vis_channels, kernel_size=1, bias=False)

        # Only the GATE is zero-initialized (Flamingo trick): tanh(0) = 0 so
        # modulation is zero at init -> K=2 forward equals K=1. But out_proj
        # is normally initialized so its output is non-zero, which gives the
        # gate a non-zero gradient and lets the recurrent pathway escape from
        # gate=0. If out_proj were *also* zero-initialized, gate gradient
        # would be sech^2(0) * 0 * dL/dy = 0 forever, and the adapter would
        # be dead on arrival.
        self.gate = nn.Parameter(torch.zeros(1))

        self.norm = nn.GroupNorm(num_groups=32, num_channels=vis_channels)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, vis_feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = vis_feat.shape
        T = ctx.shape[1]

        x = self.norm(vis_feat)
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = self.k_proj(ctx).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(ctx).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.out_proj(out)

        return vis_feat + torch.tanh(self.gate) * out

    def gate_magnitude(self) -> float:
        return float(torch.tanh(self.gate).abs().item())
