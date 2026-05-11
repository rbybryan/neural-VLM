"""Smoke test for the HF row transformers.

Tests the pure-function part of each loader (the row -> manifest
transform) without requiring HF access. Uses synthetic HF-shaped rows
that mimic the schemas at facebook/winoground and HuggingFaceM4/SugarCrepe.

Verifies:
  - winoground_rows_to_local materializes PIL images and emits items in
    the shape our `Winoground` benchmark consumes
  - sugarcrepe_rows_to_local emits items in the BinaryDiscrimination shape,
    respects the subset filter, and drops rows whose COCO image is missing
  - The emitted JSON loads cleanly into the existing Benchmark classes
    and the eval Scorer runs through them without errors
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
from PIL import Image

from data import build_eval_transform                                          # noqa: E402
from evaluation import BinaryDiscrimination, Scorer, Winoground               # noqa: E402
from evaluation.loaders import sugarcrepe_rows_to_local, winoground_rows_to_local  # noqa: E402
from scripts.build_main import build_smoke_model                              # noqa: E402
from scripts.test_data_pipeline import StubTokenizer                          # noqa: E402


def _pil_noise(seed: int, size: int = 64) -> Image.Image:
    rng = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size, size, 3, generator=rng) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def _wino_rows(n: int):
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "image_0": _pil_noise(seed=500 + i * 2),
            "image_1": _pil_noise(seed=500 + i * 2 + 1),
            "caption_0": f"caption A {i}",
            "caption_1": f"caption B {i}",
            "tag": "object",
            "collapsed_tag": "object",
        })
    return rows


def _sc_rows(coco_dir: Path, n: int, missing_ratio: float = 0.0):
    rows = []
    n_missing = int(n * missing_ratio)
    for i in range(n):
        filename = f"{i:012d}.jpg"
        if i >= n - n_missing:
            # don't materialize -> simulate missing COCO image
            pass
        else:
            _pil_noise(seed=700 + i).save(coco_dir / filename)
        category = "replace_obj" if i % 2 == 0 else "swap_att"
        rows.append({
            "image": filename,
            "caption": f"positive caption {i}",
            "negative_caption": f"negative caption {i}",
            "type": category,
        })
    return rows


def main():
    transform = build_eval_transform(image_size=224)
    tokenizer = StubTokenizer(vocab_size=2048)
    torch.manual_seed(0)
    model = build_smoke_model().eval()
    scorer = Scorer(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ----- Winoground -----
        wino_out = tmp_path / "wino"
        wino_rows = _wino_rows(n=3)
        json_path, image_root = winoground_rows_to_local(wino_rows, output_dir=wino_out)
        print(f"Winoground: {json_path}")

        with open(json_path) as f:
            items = json.load(f)
        assert len(items) == 3
        for it in items:
            for key in ("id", "image_0", "image_1", "caption_0", "caption_1"):
                assert key in it, f"missing {key} in {it}"
            assert (image_root / it["image_0"]).exists()
            assert (image_root / it["image_1"]).exists()
            assert "metadata" in it and it["metadata"]["tag"] == "object"

        wino_bench = Winoground(items=json_path, image_root=image_root, image_transform=transform)
        n_queries = 0
        for q in wino_bench.queries():
            scorer.score(q)
            n_queries += 1
            if n_queries >= 4:
                break
        assert n_queries == 4
        print(f"  Winoground benchmark: 3 items, {n_queries} sampled queries scored OK")

        # ----- SugarCrepe -----
        sc_out = tmp_path / "sc"
        coco_dir = tmp_path / "coco_val2017"
        coco_dir.mkdir()
        sc_rows = _sc_rows(coco_dir, n=10, missing_ratio=0.2)   # 2 missing out of 10
        sc_json = sugarcrepe_rows_to_local(
            sc_rows, coco_val_dir=coco_dir, output_dir=sc_out,
        )
        print(f"SugarCrepe: {sc_json}")
        with open(sc_json) as f:
            items = json.load(f)
        assert len(items) == 8, f"expected 8 (10 - 2 missing), got {len(items)}"
        cats = {it["category"] for it in items}
        assert cats == {"replace_obj", "swap_att"}

        # Subset filter test
        sc_filtered = sugarcrepe_rows_to_local(
            sc_rows, coco_val_dir=coco_dir,
            output_dir=tmp_path / "sc_filt",
            subsets=["replace_obj"],
        )
        with open(sc_filtered) as f:
            filt = json.load(f)
        assert all(it["category"] == "replace_obj" for it in filt)
        print(f"  subset filter: {len(filt)} items in 'replace_obj' only")

        sc_bench = BinaryDiscrimination(
            items=sc_json, image_root=coco_dir, image_transform=transform,
            name="sugarcrepe_smoke",
        )
        n_queries = 0
        for q in sc_bench.queries():
            scorer.score(q)
            n_queries += 1
            if n_queries >= 4:
                break
        assert n_queries == 4
        print(f"  SugarCrepe benchmark: {len(items)} items, {n_queries} sampled queries scored OK")

        print("test_loaders: PASSED")


if __name__ == "__main__":
    main()
