"""End-to-end eval smoke test on CPU with the synthetic stack.

Builds tiny synthetic Winoground-format and SugarCrepe-format benchmarks,
runs k_sweep at K=1 and K=2 with build_smoke_model, and verifies:
  - the Scorer produces finite log-likelihoods
  - Winoground aggregate produces text/image/group scores in [0, 1]
  - BinaryDiscrimination aggregate produces accuracy in [0, 1]
  - K=1 and K=2 produce identical scores at init (gates are zero -> K=2
    forward is functionally equivalent to K=1, which the prior smoke tests
    already verified at the logits level)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data import build_eval_transform                                      # noqa: E402
from evaluation import BinaryDiscrimination, Scorer, Winoground, k_sweep   # noqa: E402
from scripts.build_main import build_smoke_model                           # noqa: E402
from scripts.test_data_pipeline import StubTokenizer, _write_structured_image  # noqa: E402


def _make_synthetic_winoground(root: Path, n: int = 3) -> tuple[Path, Path]:
    image_dir = root / "wino_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        rel0, rel1 = f"wino_{i}_a.jpg", f"wino_{i}_b.jpg"
        _write_structured_image(image_dir / rel0, seed=10_000 + i * 2)
        _write_structured_image(image_dir / rel1, seed=10_000 + i * 2 + 1)
        items.append({
            "id": f"wino_{i}",
            "image_0": rel0,
            "image_1": rel1,
            "caption_0": f"A picture of pattern A-{i}",
            "caption_1": f"A picture of pattern B-{i}",
        })
    json_path = root / "winoground.json"
    with open(json_path, "w") as f:
        json.dump(items, f)
    return json_path, image_dir


def _make_synthetic_sugarcrepe(root: Path, n: int = 4) -> tuple[Path, Path]:
    image_dir = root / "sc_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    items = []
    categories = ["replace", "swap", "add"]
    for i in range(n):
        rel = f"sc_{i}.jpg"
        _write_structured_image(image_dir / rel, seed=20_000 + i)
        items.append({
            "id": f"sc_{i}",
            "image": rel,
            "positive": f"This is a picture of object {i}.",
            "negative": f"This is a picture of nothing relevant to {i}.",
            "category": categories[i % len(categories)],
        })
    json_path = root / "sugarcrepe.json"
    with open(json_path, "w") as f:
        json.dump(items, f)
    return json_path, image_dir


def _check_range(name: str, val: float, lo: float = 0.0, hi: float = 1.0):
    assert lo <= val <= hi, f"{name}={val} out of [{lo}, {hi}]"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wino_json, wino_root = _make_synthetic_winoground(tmp_path, n=3)
        sc_json, sc_root = _make_synthetic_sugarcrepe(tmp_path, n=4)

        transform = build_eval_transform(image_size=224)
        wino = Winoground(items=wino_json, image_root=wino_root, image_transform=transform)
        sc = BinaryDiscrimination(items=sc_json, image_root=sc_root,
                                   image_transform=transform, name="sugarcrepe_smoke")

        torch.manual_seed(0)
        model = build_smoke_model().eval()
        tokenizer = StubTokenizer(vocab_size=2048)
        scorer = Scorer(model=model, tokenizer=tokenizer, length_normalize=False)

        # ---- Winoground ----
        print("Running Winoground (3 items x 4 queries = 12 forwards x 2 K values)")
        wino_results = k_sweep(
            model, scorer, wino,
            K_values=(1, 2),
            log_fn=lambda K, m: print(f"  K={K}: text={m['text_score']:.3f} "
                                       f"image={m['image_score']:.3f} "
                                       f"group={m['group_score']:.3f} (n={m['n_items']})"),
        )

        for K in (1, 2):
            m = wino_results["per_K"][K]
            for name in ("text_score", "image_score", "group_score"):
                _check_range(f"wino K={K} {name}", m[name])
            assert m["n_items"] == 3

        # At init, all gates are 0 so K=2 must yield identical scores to K=1
        s1 = wino_results["scores_per_K"][1]
        s2 = wino_results["scores_per_K"][2]
        assert set(s1.keys()) == set(s2.keys())
        max_diff = max(abs(s1[q] - s2[q]) for q in s1)
        print(f"  max |K=2 - K=1| score at init = {max_diff:.2e}")
        assert max_diff < 1e-3, f"K=1 vs K=2 score diff at init = {max_diff}"

        # ---- BinaryDiscrimination ----
        print("Running BinaryDiscrimination (4 items x 2 queries = 8 forwards x 2 K values)")
        sc_results = k_sweep(
            model, scorer, sc,
            K_values=(1, 2),
            log_fn=lambda K, m: print(f"  K={K}: acc={m['accuracy']:.3f} (n={m['n_items']})  "
                                       f"per-cat: " + ", ".join(
                                           f"{k.split('/')[1]}={v:.2f}"
                                           for k, v in m.items()
                                           if k.startswith("accuracy/"))),
        )

        for K in (1, 2):
            m = sc_results["per_K"][K]
            _check_range(f"sc K={K} accuracy", m["accuracy"])
            assert m["n_items"] == 4

        s1, s2 = sc_results["scores_per_K"][1], sc_results["scores_per_K"][2]
        max_diff = max(abs(s1[q] - s2[q]) for q in s1)
        assert max_diff < 1e-3, f"K=1 vs K=2 score diff at init = {max_diff}"

        print("test_eval: PASSED")


if __name__ == "__main__":
    main()
