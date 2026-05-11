"""SugarCrepe HF loader.

SugarCrepe stores captions on HuggingFace but its images come from MS-COCO
val2017 (~1 GB). The standard distribution covers 7 subsets:
  replace_obj, replace_att, replace_rel,
  swap_obj, swap_att,
  add_obj, add_att

Each row has:
  image: COCO val2017 filename (e.g. "000000262284.jpg")
  caption: positive caption
  negative_caption: perturbed caption (the model should reject)
  type: subset name (used as our `category` field)

Usage requires the user to have COCO val2017 downloaded separately:
    http://images.cocodataset.org/zips/val2017.zip

Our manifest uses the COCO val2017 directory as `image_root` and stores
just the filename in each item, so the existing BinaryDiscrimination
benchmark works without modification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SUBSETS = (
    "replace_obj", "replace_att", "replace_rel",
    "swap_obj", "swap_att",
    "add_obj", "add_att",
)


def sugarcrepe_rows_to_local(
    rows: Iterable[dict],
    coco_val_dir: str | Path,
    output_dir: str | Path,
    subsets: Sequence[str] | None = None,
    require_images_exist: bool = True,
) -> Path:
    """Transform SugarCrepe-format rows into our JSON manifest.

    Each row must have: image (COCO filename), caption (positive),
    negative_caption, type (subset name).

    Returns the path to the emitted JSON. `image_root` (passed to
    BinaryDiscrimination) should be set to coco_val_dir at use time.
    """
    coco_val_dir = Path(coco_val_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_filter = set(subsets) if subsets is not None else None

    items: list[dict] = []
    missing_images = 0
    for i, row in enumerate(rows):
        category = row.get("type") or row.get("category") or ""
        if subset_filter is not None and category not in subset_filter:
            continue

        filename = row["image"]
        if require_images_exist and not (coco_val_dir / filename).exists():
            missing_images += 1
            continue

        items.append({
            "id": f"{category or 'sc'}_{i:06d}",
            "image": filename,
            "positive": row["caption"],
            "negative": row["negative_caption"],
            "category": category,
        })

    json_path = output_dir / "sugarcrepe.json"
    with open(json_path, "w") as f:
        json.dump(items, f, indent=2)

    if missing_images:
        print(
            f"warning: {missing_images} rows skipped because their COCO image "
            f"was not present in {coco_val_dir}"
        )
    return json_path


def load_sugarcrepe(
    output_dir: str | Path,
    coco_val_dir: str | Path,
    hf_repo: str = "HuggingFaceM4/SugarCrepe",
    hf_token: str | None = None,
    cache_dir: str | None = None,
    subsets: Sequence[str] | None = DEFAULT_SUBSETS,
) -> Path:
    """Pull SugarCrepe rows from HF (`hf_repo`) and emit our manifest.

    The HF repo's exact split layout varies between mirrors (HuggingFaceM4
    vs xinli vs the official jsons). We read every available split and
    concatenate; the per-row `type` field tells us which subset each row
    belongs to.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_repo, token=hf_token, cache_dir=cache_dir)
    # Different mirrors expose splits differently; concatenate everything.
    rows: list[dict] = []
    for split_name in ds.keys() if hasattr(ds, "keys") else [None]:
        rows.extend(ds[split_name] if split_name else ds)
    return sugarcrepe_rows_to_local(
        rows, coco_val_dir=coco_val_dir, output_dir=output_dir, subsets=subsets,
    )
