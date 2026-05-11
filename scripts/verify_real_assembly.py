"""Real-weight assembly verification.

Pulls the configured LLM and ConvNeXt-V2-Base from the relevant hubs and
assembles the full RecurrentVLM with real backbones and LoRA wrapping.
Runs a forward + backward + optimizer step to verify:

  1. timm + HF weights load cleanly at the target dtype
  2. the model assembles end-to-end at the chosen tier's dimensions
  3. LoRA wrap targets the right modules and reports a sensible
     trainable-parameter fraction
  4. zero-init guarantee: K=2 logits exactly equal K=1 logits at init
  5. Flamingo unlock: gate gradient non-zero on step 0 -> gate moves on
     step 1 -> out_proj gradient unlocks at step 1

Usage
-----
    # CPU verification with the dev tier (default)
    python scripts/verify_real_assembly.py

    # GPU verification with bf16 on the 1.5B tier
    python scripts/verify_real_assembly.py \\
        --config configs/main.yaml --dtype bf16 --device cuda

    # Full 7B publication-tier check (only on a >=80 GB GPU)
    python scripts/verify_real_assembly.py \\
        --config configs/main_pub.yaml --dtype bf16 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_main import build_main_model, load_config       # noqa: E402


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def fmt_n(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "main_dev.yaml"))
    p.add_argument("--dtype", default="fp32", choices=list(DTYPE_MAP.keys()))
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    p.add_argument("--seq-len", type=int, default=8, help="text tokens for the synthetic batch")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    if dtype == torch.bfloat16 and device.type == "cpu":
        print("warning: bf16 on CPU has limited kernel support; falling back to fp32 might be safer")

    cfg = load_config(args.config)
    print(f"config: {cfg['name']}  (from {args.config})")
    print(f"  encoder: {cfg['encoder']['arch']} (pretrained={cfg['encoder']['pretrained']})")
    print(f"  llm:     {cfg['llm']['arch']}  injection_layer={cfg['llm']['injection_layer']}")
    print(f"  K:       {cfg['recurrence']['num_iterations']}")
    print(f"  device:  {device}    dtype: {args.dtype}")
    if device.type == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print()

    print("=" * 70)
    print("1. Building model with real weights (downloads on first run)")
    print("=" * 70)
    t0 = time.time()
    model = build_main_model(cfg, pretrained=True, dtype=dtype)
    model = model.to(device)
    print(f"  built and moved to {device} in {time.time() - t0:.1f}s")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params:     {fmt_n(n_total):>10}  ({n_total:,})")
    print(f"  trainable params: {fmt_n(n_train):>10}  ({n_train:,})  "
          f"= {100 * n_train / n_total:.2f}% of total")

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"  GPU memory: {mem_alloc:.2f} GB allocated / {mem_reserved:.2f} GB reserved")

    # Dtype consistency check
    print("  dtype consistency:")
    components = {
        "encoder.stages.3": next(model.encoder.stages[3].parameters()),
        "connector":        next(model.connector.parameters()),
        "adapters['3']":    next(model.adapters['3'].parameters()),
        "memory.tokens":    model.memory.tokens,
        "llm (first param)": next(model.llm.parameters()),
    }
    for k, p in components.items():
        print(f"    {k:<22} dtype={p.dtype}  device={p.device}")
    print()

    # Trainable breakdown
    print("  trainable parameter breakdown:")
    groups = {
        "encoder.stages.2": 0, "encoder.stages.3": 0,
        "connector": 0, "adapters": 0, "memory": 0,
        "llm (LoRA)": 0, "other": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if "encoder.stages.2." in name:
            groups["encoder.stages.2"] += n
        elif "encoder.stages.3." in name:
            groups["encoder.stages.3"] += n
        elif name.startswith("connector"):
            groups["connector"] += n
        elif name.startswith("adapters"):
            groups["adapters"] += n
        elif name.startswith("memory"):
            groups["memory"] += n
        elif "lora_" in name.lower():
            groups["llm (LoRA)"] += n
        else:
            groups["other"] += n
    for k, v in groups.items():
        if v > 0:
            print(f"    {k:<22} {fmt_n(v):>10}  ({v:,})")
    print()

    print("=" * 70)
    print("2. Forward pass: zero-init equivalence (K=2 must equal K=1 at init)")
    print("=" * 70)
    model.eval()
    torch.manual_seed(0)
    B = 1
    image = torch.randn(B, 3, 224, 224, device=device, dtype=dtype)
    text_ids = torch.randint(0, 1024, (B, args.seq_len), device=device)
    text_mask = torch.ones_like(text_ids)

    t0 = time.time()
    with torch.no_grad():
        out_K2 = model(image=image, text_ids=text_ids, text_mask=text_mask)
        original = model.cfg.use_recurrence
        model.cfg.use_recurrence = False
        out_K1 = model(image=image, text_ids=text_ids, text_mask=text_mask)
        model.cfg.use_recurrence = original
    t_fwd = time.time() - t0

    logits_K2 = out_K2["logits"]
    logits_K1 = out_K1["logits"]
    diff = (logits_K2.float() - logits_K1.float()).abs().max().item()
    n_vis = logits_K2.shape[1] - text_ids.shape[1]

    print(f"  logits K=1:       {tuple(logits_K1.shape)}  dtype={logits_K1.dtype}")
    print(f"  logits K=2:       {tuple(logits_K2.shape)}  ({n_vis} visual + {text_ids.shape[1]} text)")
    print(f"  per_iter_logits:  {len(out_K2['per_iter_logits'])} iterations")
    print(f"  gate magnitudes:  {out_K2['gate_magnitudes']}")
    print(f"  max |K=2 - K=1| at init: {diff:.2e}")
    print(f"  forward time ({device.type}, both passes): {t_fwd:.1f}s")
    # bf16 has limited precision so allow a looser tolerance
    tol = 1e-3 if dtype == torch.float32 else 5e-3
    assert diff < tol, f"zero-init violated: max diff {diff} (tol {tol})"
    print(f"  PASS: zero-init equivalence holds on real weights (tol {tol:.0e})")
    print()

    print("=" * 70)
    print("3. Backward pass + optimizer step (Flamingo unlock)")
    print("=" * 70)
    model.train()
    labels = torch.full((B, args.seq_len), -100, dtype=torch.long, device=device)
    labels[:, -args.seq_len // 2:] = torch.randint(0, 1024, (B, args.seq_len // 2), device=device)

    t0 = time.time()
    out = model(image=image, text_ids=text_ids, text_mask=text_mask, labels=labels)
    loss = out["loss"]
    print(f"  step-0 loss: {loss.item():.4f}")
    loss.backward()
    t_bwd = time.time() - t0
    print(f"  backward time ({device.type}): {t_bwd:.1f}s")

    gate_grads = {s: a.gate.grad.abs().max().item() for s, a in model.adapters.items()}
    print(f"  gate gradients (must be > 0 for Flamingo unlock): {gate_grads}")
    assert all(g > 0 for g in gate_grads.values()), "gate gradient is zero — Flamingo unlock broken"

    out_proj_grads_step0 = {
        s: (a.out_proj.weight.grad.abs().max().item() if a.out_proj.weight.grad is not None else 0.0)
        for s, a in model.adapters.items()
    }
    print(f"  out_proj gradients at step 0 (expected 0): {out_proj_grads_step0}")

    groups = model.trainable_param_groups(base_lr=1e-3, td_lr_mult=7.0)
    optim = torch.optim.AdamW(groups)
    optim.step()
    optim.zero_grad()
    gates_after = {s: a.gate.detach().abs().max().item() for s, a in model.adapters.items()}
    print(f"  gate values after one optimizer step (must be > 0): {gates_after}")
    assert all(g > 0 for g in gates_after.values()), "gate did not move after step"

    out2 = model(image=image, text_ids=text_ids, text_mask=text_mask, labels=labels)
    out2["loss"].backward()
    out_proj_grads_step1 = {
        s: a.out_proj.weight.grad.abs().max().item() for s, a in model.adapters.items()
    }
    print(f"  out_proj gradients at step 1: {out_proj_grads_step1}")
    assert all(g > 0 for g in out_proj_grads_step1.values()), \
        "out_proj did not unlock after gate moved"
    print("  PASS: Flamingo unlock dynamic verified on real weights")
    print()

    if device.type == "cuda":
        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"GPU peak memory across forward + backward: {mem_peak:.2f} GB")
        print()

    print("=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)
    print(f"  {cfg['name']} assembles cleanly at {args.dtype} on {device}.")
    print("  Zero-init equivalence holds. Flamingo unlock works.")


if __name__ == "__main__":
    main()
