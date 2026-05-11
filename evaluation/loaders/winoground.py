"""Winoground HF loader.

Source: facebook/winoground (gated on HuggingFace; requires accepting terms
at https://huggingface.co/datasets/facebook/winoground and an HF token).

The HF dataset row schema is:
  id: int
  image_0: PIL.Image
  image_1: PIL.Image
  caption_0: str
  caption_1: str
  tag, secondary_tag, num_main_preds, collapsed_tag: metadata

We materialize the embedded PIL images to disk and emit a JSON manifest in
the shape that `evaluation.benchmarks.Winoground` consumes:

    [{"id": str, "image_0": "path/to.png", "image_1": "path/to.png",
      "caption_0": str, "caption_1": str, "metadata": {...}}, ...]

We always re-save images (Winoground is only 400 items / 800 images and
fits in ~30 MB at typical resolution). For decontamination, the saved
images become the input to `data.decontaminate build-index`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image

METADATA_FIELDS = ("tag", "secondary_tag", "num_main_preds", "collapsed_tag")


def winoground_rows_to_local(
    rows: Iterable[dict],
    output_dir: str | Path,
    image_format: str = "png",
) -> tuple[Path, Path]:
    """Transform Winoground-format rows into our local JSON + image format.

    Each row must have id, image_0 (PIL), image_1 (PIL), caption_0, caption_1.
    Returns (items_json_path, image_root).
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for row in rows:
        item_id = str(row["id"])
        rel_0 = f"wino_{item_id}_0.{image_format}"
        rel_1 = f"wino_{item_id}_1.{image_format}"
        img_0 = row["image_0"]
        img_1 = row["image_1"]
        if not isinstance(img_0, Image.Image):
            raise TypeError(f"image_0 for id={item_id} is {type(img_0)}; expected PIL.Image")
        img_0.convert("RGB").save(image_dir / rel_0)
        img_1.convert("RGB").save(image_dir / rel_1)

        item = {
            "id": item_id,
            "image_0": rel_0,
            "image_1": rel_1,
            "caption_0": row["caption_0"],
            "caption_1": row["caption_1"],
        }
        meta = {k: row[k] for k in METADATA_FIELDS if k in row}
        if meta:
            item["metadata"] = meta
        items.append(item)

    json_path = output_dir / "winoground.json"
    with open(json_path, "w") as f:
        json.dump(items, f, indent=2)
    return json_path, image_dir


def load_winoground(
    output_dir: str | Path,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    split: str = "test",
) -> tuple[Path, Path]:
    """Pull facebook/winoground from HF and save to local format.

    Requires the user to have accepted terms on the HF dataset page and
    supplied a token (via `hf_token=` or HF_TOKEN env var).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "facebook/winoground",
        token=hf_token,
        cache_dir=cache_dir,
        split=split,
    )
    return winoground_rows_to_local(ds, output_dir)
