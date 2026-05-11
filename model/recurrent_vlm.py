"""Recurrent VLM orchestrator.

Wires together:
  - a CNN visual encoder with stage-level access (forward + forward_from)
  - a connector projecting CNN tokens into LLM hidden-state space
  - top-down adapters at configured stages
  - a persistent memory bank
  - an LLM wrapper that consumes connector-projected visual tokens
  - a K-iteration forward loop where the LLM's mid-layer hidden states
    re-enter the encoder via the adapters on iterations 2..K

Modulation propagation: when injection happens at a non-final encoder stage,
the modulated features are propagated through subsequent stages via the
encoder's forward_from() so the LLM sees the downstream effects of the
modulation, not just the modulated stage in isolation.

The class is deliberately backbone-agnostic. Concrete encoders/LLMs are
injected by composition so the same module can serve the main model and
control models C1-C9 via RecurrentVLMConfig toggles.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .memory import PersistentMemoryBank
from .top_down_adapter import TopDownAdapter


@dataclass
class RecurrentVLMConfig:
    injection_stages: tuple[int, ...] = (2, 3)
    llm_injection_layer: int = 14
    num_iterations: int = 2
    aux_iter1_loss_weight: float = 0.3
    memory_tokens: int = 32

    use_llm_kv: bool = True              # C2: False routes connector-projected lateral CNN features as KV
    use_recurrence: bool = True          # C3: False forces K=1
    randomize_llm_kv: bool = False       # C4: replace LLM KV with frozen random projection
    lesion_memory: bool = False          # C7: zero out memory tokens
    scramble_iterations: bool = False    # C6: swap iter-1 / iter-2 features (eval-only)


class RecurrentVLM(nn.Module):
    def __init__(
        self,
        config: RecurrentVLMConfig,
        encoder: nn.Module,                  # ConvNeXtV2Encoder-like (forward, forward_from)
        encoder_stage_channels: list[int],
        llm: nn.Module,                      # LLMWrapper
        connector: nn.Module,                # MLPConnector
        llm_hidden_dim: int,
        random_kv_proj: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = config
        self.encoder = encoder
        self.llm = llm
        self.connector = connector

        self.adapters = nn.ModuleDict({
            str(s): TopDownAdapter(
                vis_channels=encoder_stage_channels[s],
                ctx_dim=llm_hidden_dim,
            )
            for s in config.injection_stages
        })

        self.memory = PersistentMemoryBank(
            num_tokens=config.memory_tokens,
            dim=llm_hidden_dim,
        )

        self.random_kv_proj = random_kv_proj  # frozen at init for C4

    def _build_ctx(
        self,
        llm_hidden: torch.Tensor,
        memory_tokens: torch.Tensor,
        cnn_lateral_proj: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.use_llm_kv:
            base = cnn_lateral_proj                                # C2 (already in llm_dim via connector)
        elif self.cfg.randomize_llm_kv and self.random_kv_proj is not None:
            with torch.no_grad():
                base = self.random_kv_proj(llm_hidden)             # C4
        else:
            base = llm_hidden
        return torch.cat([base, memory_tokens], dim=1)

    def _project_last_stage(self, stage_features: list[torch.Tensor]) -> torch.Tensor:
        last = stage_features[-1]
        B, C, H, W = last.shape
        vis_tokens = last.flatten(2).transpose(1, 2)               # (B, H*W, C)
        return self.connector(vis_tokens)                          # (B, H*W, llm_dim)

    def _inject_and_propagate(
        self,
        stage_features: list[torch.Tensor],
        ctx: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Inject at each configured stage, shallow-to-deep, propagating
        modulations through subsequent stages via encoder.forward_from()."""
        feats = list(stage_features)
        last_idx = len(feats) - 1
        for s in sorted(self.cfg.injection_stages):
            feats[s] = self.adapters[str(s)](feats[s], ctx)
            if s < last_idx:
                downstream = self.encoder.forward_from(s, feats[s])
                feats = feats[: s + 1] + downstream[1:]
        return feats

    def forward(
        self,
        image: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        B = image.shape[0]
        device = image.device

        if self.cfg.lesion_memory:
            memory_tokens = self.memory.lesion(B, device)
        else:
            memory_tokens = self.memory.initial(B)

        K = 1 if not self.cfg.use_recurrence else self.cfg.num_iterations

        stage_features = self.encoder(image)
        per_iter_logits: list[torch.Tensor] = []
        per_iter_features: list[list[torch.Tensor]] = [stage_features]

        for k in range(K):
            vis_tokens_proj = self._project_last_stage(stage_features)

            llm_out = self.llm(
                vis_tokens=vis_tokens_proj,
                input_ids=text_ids,
                attention_mask=text_mask,
                output_hidden_states=True,
            )
            per_iter_logits.append(llm_out["logits"])

            if k == K - 1:
                break

            mid_hidden = llm_out["hidden_states"][self.cfg.llm_injection_layer]
            if not self.cfg.lesion_memory:
                memory_tokens = self.memory.read_write(memory_tokens, mid_hidden)

            ctx = self._build_ctx(mid_hidden, memory_tokens, vis_tokens_proj)
            stage_features = self._inject_and_propagate(stage_features, ctx)
            per_iter_features.append(stage_features)

        if self.cfg.scramble_iterations and len(per_iter_features) >= 2:
            per_iter_features[0], per_iter_features[-1] = (
                per_iter_features[-1],
                per_iter_features[0],
            )

        out = {
            "logits": per_iter_logits[-1],
            "per_iter_logits": per_iter_logits,
            "per_iter_features": per_iter_features,
            "gate_magnitudes": {s: a.gate_magnitude() for s, a in self.adapters.items()},
            # Final memory token bank after K-1 read/write cycles. Exposed
            # so probes (evaluation/probes/token_decoder.py) can verify the
            # persistent tokens carry image-relevant information.
            "final_memory_tokens": memory_tokens,
        }

        if labels is not None:
            out["loss"] = self._loss(per_iter_logits, labels)
        return out

    def _loss(self, per_iter_logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy with standard next-token shift.

        labels can be either:
          (B, T_text)         text-only — visual prefix is auto-padded with -100
          (B, N_vis + T_text) full-length — pass through unchanged

        Visual-position padding plus the standard HF causal-LM shift means
        the loss supervises predictions of text tokens only, conditioned on
        visual context + earlier text.
        """
        ce = nn.CrossEntropyLoss(ignore_index=-100)
        final = per_iter_logits[-1]
        B, full_len, V = final.shape
        T_label = labels.size(1)

        if T_label < full_len:
            n_vis = full_len - T_label
            pad = torch.full((B, n_vis), -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([pad, labels], dim=1)
        elif T_label == full_len:
            full_labels = labels
        else:
            raise ValueError(f"labels length {T_label} > logits length {full_len}")

        shift_logits = final[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()
        loss = ce(shift_logits.view(-1, V), shift_labels.view(-1))

        if len(per_iter_logits) > 1 and self.cfg.aux_iter1_loss_weight > 0:
            first = per_iter_logits[0]
            shift_first = first[:, :-1, :].contiguous()
            aux = ce(shift_first.view(-1, V), shift_labels.view(-1))
            loss = loss + self.cfg.aux_iter1_loss_weight * aux
        return loss

    def trainable_param_groups(self, base_lr: float, td_lr_mult: float = 7.0):
        td_modules = list(self.adapters.parameters()) + list(self.memory.parameters())
        td_set = {id(p) for p in td_modules}
        td_params, base_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (td_params if id(p) in td_set else base_params).append(p)
        return [
            {"params": base_params, "lr": base_lr},
            {"params": td_params, "lr": base_lr * td_lr_mult},
        ]
