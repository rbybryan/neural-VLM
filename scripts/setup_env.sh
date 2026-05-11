#!/usr/bin/env bash
# One-shot environment setup for VLM-custom.
#
# Creates conda env `vlm_custom` (Python 3.11), installs PyTorch 2.5.1 with
# CUDA 12.1, then the rest of the stack from requirements.txt. Re-run is safe:
# pip will skip already-satisfied packages.

set -euo pipefail

ENV_NAME=vlm_custom
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi

PYBIN="$(conda run -n "$ENV_NAME" which python)"

"$PYBIN" -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

"$PYBIN" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"

"$PYBIN" - <<'PY'
import importlib, sys
mods = ["torch", "torchvision", "transformers", "timm", "peft",
        "accelerate", "datasets", "numpy", "scipy", "sklearn",
        "nibabel", "h5py", "mne", "yaml"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ok  {m:<14} {v}")
    except Exception as e:
        print(f"  ERR {m:<14} {e}")
        sys.exit(1)
print("setup_env.sh: all imports succeeded.")
PY
