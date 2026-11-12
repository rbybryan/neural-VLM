"""Run all synthetic smoke checks, without downloading weights or datasets."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ["scripts/build_main.py", "--smoke"],
    ["scripts/test_data_pipeline.py"],
    ["scripts/test_training.py"],
    ["scripts/test_eval.py"],
    ["scripts/test_specificity_suite.py"],
    ["scripts/test_probes.py"],
    ["scripts/test_loaders.py"],
]


def main() -> None:
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
                "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "true"})
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    start = time.monotonic()
    for check in CHECKS:
        print(f"\n>>> {' '.join(check)}", flush=True)
        subprocess.run([sys.executable, *check], cwd=ROOT, env=env, check=True)
    print(f"\nAll {len(CHECKS)} CPU checks passed in {time.monotonic() - start:.1f}s.")
    print("Synthetic checks establish implementation behavior, not trained-model quality.")


if __name__ == "__main__":
    main()
