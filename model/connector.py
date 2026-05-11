"""MLP connector projecting CNN visual tokens into LLM hidden-state space.

Input shape:  (B, N, vis_channels)   N = H * W of the deepest CNN stage
Output shape: (B, N, llm_dim)
"""

from __future__ import annotations

from torch import nn


class MLPConnector(nn.Module):
    def __init__(self, vis_channels: int, llm_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = llm_dim * hidden_mult
        self.proj = nn.Sequential(
            nn.Linear(vis_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.proj(x)
