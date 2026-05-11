"""Construct the main RecurrentVLM and any of the C1-C9 control variants.

Usage
-----
    # Real model (requires GPU + ~14 GB free for weights download/load)
    from scripts.build_main import build_main_model
    cfg = load_config("configs/main.yaml")
    model = build_main_model(cfg, pretrained=True)

    # CPU-friendly smoke test with tiny synthetic backbones
    python scripts/build_main.py --smoke

The smoke path uses convnextv2_atto (random init) and a 4-layer Qwen2 LM
constructed from config (no weights download). It exercises the full
forward through every iteration and verifies shape contracts plus the
zero-init = vanilla-VLM property.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import (                                 # noqa: E402
    ConvNeXtV2Encoder,
    LLMWrapper,
    MLPConnector,
    RecurrentVLM,
    RecurrentVLMConfig,
    apply_lora,
)


# ---------------------------------------------------------------- real model

def _build_qwen_lm(model_id: str, dtype: torch.dtype | None = None) -> nn.Module:
    """Load Qwen2.5 base LM (text-only). The visual encoder of the parent
    Qwen2.5-VL checkpoint, if any, is intentionally not used — our ConvNeXt-V2
    plus top-down loop replaces it.

    `dtype=None` (default) keeps the model in fp32 — safe on CPU and for
    verification. For GPU training pass torch.bfloat16."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # HF defaults to the checkpoint's saved dtype (bf16 for Qwen2.5) when
    # no torch_dtype is given, which causes mixed-dtype kernels with our
    # fp32 visual tower on CPU. Force fp32 unless the caller asks otherwise.
    if dtype is None:
        dtype = torch.float32
    return AutoModelForCausalLM.from_pretrained(
        model_id, config=cfg, torch_dtype=dtype, trust_remote_code=True,
    )


def build_main_model(
    cfg: dict[str, Any],
    pretrained: bool = True,
    dtype: torch.dtype | None = None,
) -> RecurrentVLM:
    """Build the full RecurrentVLM with real backbones.

    Args:
        cfg: parsed config dict (see configs/main.yaml).
        pretrained: load ConvNeXt + Qwen2.5 weights from hubs (True) or
            randomly initialize from architecture configs (False).
        dtype: target precision for the whole model. None = fp32 (safe on
            CPU). For GPU training pass torch.bfloat16. The LLM is loaded
            directly in this dtype (avoids the fp32 -> bf16 peak-memory
            blip during from_pretrained), and the visual tower is cast at
            the end of construction so all components agree.
    """
    enc_cfg = cfg["encoder"]
    encoder = ConvNeXtV2Encoder(
        model_name=enc_cfg["arch"],
        pretrained=pretrained and enc_cfg.get("pretrained") is not False,
        freeze_stages=tuple(enc_cfg.get("freeze_stages", ())),
    )
    stage_channels = encoder.channels

    llm_cfg = cfg["llm"]
    llm_id = llm_cfg["arch"]
    base_llm = _build_qwen_lm(llm_id, dtype=dtype) if pretrained else _build_qwen_lm_from_config(llm_id)

    if "lora" in llm_cfg:
        lc = llm_cfg["lora"]
        base_llm = apply_lora(
            base_llm,
            rank=lc.get("rank", 16),
            alpha=lc.get("alpha", 32),
            dropout=lc.get("dropout", 0.05),
            target_modules=tuple(lc.get("target_modules", (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ))),
        )
    llm_wrapper = LLMWrapper(base_llm)
    llm_dim = llm_wrapper.hidden_size

    connector = MLPConnector(stage_channels[-1], llm_dim, hidden_mult=2)

    rec_cfg = cfg["recurrence"]
    rvlm_cfg = RecurrentVLMConfig(
        injection_stages=tuple(enc_cfg.get("injection_stages", (2, 3))),
        llm_injection_layer=int(llm_cfg["injection_layer"]),
        num_iterations=int(rec_cfg["num_iterations"]),
        aux_iter1_loss_weight=float(rec_cfg.get("aux_iter1_loss_weight", 0.3)),
        memory_tokens=int(cfg.get("memory", {}).get("num_tokens", 32)),
        use_llm_kv=bool(cfg.get("use_llm_kv", True)),
        use_recurrence=bool(cfg.get("use_recurrence", True)),
        randomize_llm_kv=bool(cfg.get("randomize_llm_kv", False)),
        lesion_memory=bool(cfg.get("lesion_memory", False)),
        scramble_iterations=bool(cfg.get("scramble_iterations", False)),
    )

    random_kv_proj = None
    if rvlm_cfg.randomize_llm_kv:
        random_kv_proj = nn.Linear(llm_dim, llm_dim, bias=False)
        for p in random_kv_proj.parameters():
            p.requires_grad = False

    model = RecurrentVLM(
        config=rvlm_cfg,
        encoder=encoder,
        encoder_stage_channels=stage_channels,
        llm=llm_wrapper,
        connector=connector,
        llm_hidden_dim=llm_dim,
        random_kv_proj=random_kv_proj,
    )

    if dtype is not None:
        # Cast the visual tower + new modules to the LLM's dtype so the
        # forward pass doesn't hit mixed-dtype kernels. The LLM is already
        # in dtype from _build_qwen_lm; this is a no-op for it.
        model.encoder.to(dtype)
        model.connector.to(dtype)
        model.adapters.to(dtype)
        model.memory.to(dtype)
        if model.random_kv_proj is not None:
            model.random_kv_proj.to(dtype)

    return model


def load_config(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def apply_control_overrides(base_cfg: dict, control_id: str) -> dict:
    controls = OmegaConf.load(ROOT / "configs" / "controls.yaml")
    if control_id not in controls:
        raise KeyError(f"Unknown control id {control_id!r}; known: {list(controls.keys())}")
    delta = OmegaConf.to_container(controls[control_id], resolve=True)
    delta.pop("notes", None)
    delta.pop("eval_only", None)
    merged = OmegaConf.merge(OmegaConf.create(base_cfg), OmegaConf.create(delta))
    return OmegaConf.to_container(merged, resolve=True)


# ------------------------------------------------------------ synthetic smoke

def _build_synthetic_qwen_lm(hidden_size: int = 256, num_layers: int = 4) -> nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=2048,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    return Qwen2ForCausalLM(cfg)


def build_smoke_model() -> RecurrentVLM:
    encoder = ConvNeXtV2Encoder(
        model_name="convnextv2_atto",
        pretrained=False,
        freeze_stages=(0, 1),
    )
    stage_channels = encoder.channels

    llm = _build_synthetic_qwen_lm(hidden_size=256, num_layers=4)
    llm_wrapper = LLMWrapper(llm)
    connector = MLPConnector(stage_channels[-1], llm_wrapper.hidden_size, hidden_mult=2)

    cfg = RecurrentVLMConfig(
        injection_stages=(2, 3),
        llm_injection_layer=2,            # mid-layer of 4-layer synthetic LM
        num_iterations=2,
        aux_iter1_loss_weight=0.3,
        memory_tokens=8,
    )
    return RecurrentVLM(
        config=cfg,
        encoder=encoder,
        encoder_stage_channels=stage_channels,
        llm=llm_wrapper,
        connector=connector,
        llm_hidden_dim=llm_wrapper.hidden_size,
    )


def smoke_test() -> None:
    torch.manual_seed(0)
    model = build_smoke_model().eval()

    B = 2
    image = torch.randn(B, 3, 224, 224)
    text_ids = torch.randint(0, 2048, (B, 16))
    text_mask = torch.ones_like(text_ids)
    labels = torch.full((B, 16 + 49), -100, dtype=torch.long)
    labels[:, -8:] = torch.randint(0, 2048, (B, 8))

    print(f"smoke_test: encoder stage channels = {model.encoder.channels}")
    print(f"smoke_test: llm_dim = {model.llm.hidden_size}")

    # Zero-init property: at init, K=2 forward should equal K=1 forward exactly
    # (because all top-down adapter gates are zero -> modulation = 0).
    with torch.no_grad():
        out_K2 = model(image, text_ids, text_mask, labels=labels)

        # Force K=1 by toggling use_recurrence on the same model
        original_K = model.cfg.use_recurrence
        model.cfg.use_recurrence = False
        out_K1 = model(image, text_ids, text_mask, labels=labels)
        model.cfg.use_recurrence = original_K

    logits_K2 = out_K2["logits"]
    logits_K1 = out_K1["logits"]
    print(f"smoke_test: logits shape K=1 {tuple(logits_K1.shape)} | K=2 {tuple(logits_K2.shape)}")
    print(f"smoke_test: per_iter_logits count K=2 = {len(out_K2['per_iter_logits'])}")
    print(f"smoke_test: per_iter_features iterations K=2 = {len(out_K2['per_iter_features'])}")
    print(f"smoke_test: gate magnitudes = {out_K2['gate_magnitudes']}")

    diff = (logits_K2 - logits_K1).abs().max().item()
    print(f"smoke_test: max |logits_K2 - logits_K1| at init = {diff:.2e}")
    assert diff < 1e-4, f"zero-init K=2 should match K=1, got max diff {diff}"

    # Loss should be finite and gradients should flow
    out = model(image, text_ids, text_mask, labels=labels)
    print(f"smoke_test: loss = {out['loss'].item():.4f}")
    out["loss"].backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"smoke_test: nonzero-grad params {n_grad}/{n_total} (some adapters may be zero-grad initially)")

    # Param-group split
    groups = model.trainable_param_groups(base_lr=1e-4, td_lr_mult=7.0)
    n_base = sum(p.numel() for p in groups[0]["params"])
    n_td = sum(p.numel() for p in groups[1]["params"])
    print(f"smoke_test: param groups | base {n_base:,} ({groups[0]['lr']:.0e}) | top-down {n_td:,} ({groups[1]['lr']:.0e})")

    print("smoke_test: PASSED")


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run synthetic smoke test (CPU-friendly)")
    p.add_argument("--config", default=None, help="path to a YAML config")
    p.add_argument("--control", default=None, help="optional control id from configs/controls.yaml")
    p.add_argument("--no-pretrained", action="store_true", help="skip pretrained weight downloads")
    args = p.parse_args()

    if args.smoke:
        smoke_test()
        return

    if args.config is None:
        print("error: --config required (or use --smoke)", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(args.config)
    if args.control:
        cfg = apply_control_overrides(cfg, args.control)
        print(f"applied control overrides: {args.control}")

    model = build_main_model(cfg, pretrained=not args.no_pretrained)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"built {cfg.get('name', 'model')}: total {n_total:,} | trainable {n_train:,} ({100 * n_train / n_total:.2f}%)")


if __name__ == "__main__":
    main()
