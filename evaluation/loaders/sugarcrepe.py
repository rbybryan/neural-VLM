"""SugarCrepe HF loader.

HuggingFaceM4 hosts SugarCrepe as one repo per subset:
  HuggingFaceM4/SugarCrepe_{replace_obj,replace_att,replace_rel,swap_obj,swap_att,add_obj,add_att}

Each row has:
  image:            PIL Image (the COCO image, embedded in the HF repo)
  tested_labels:    [positive_caption, negative_caption]  (index 0 is always the correct one)
  true_label:       internal SugarCrepe annotation ID (not a COCO image ID)
  type:             (injected by us) — subset name

We save the PIL images to output_dir/images/ and build a manifest that
BinaryDiscrimination can consume.  coco_val_dir is accepted as an optional
argument for compatibility with test stubs that provide pre-saved filenames
instead of PIL images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

SUGARCREPE_SUBSETS = (
    "replace_obj", "replace_att", "replace_rel",
    "swap_obj", "swap_att",
    "add_obj", "add_att",
)
DEFAULT_SUBSETS = SUGARCREPE_SUBSETS

# HF hosts one repo per subset: HuggingFaceM4/SugarCrepe_{subset}
_HF_REPO_TEMPLATE = "HuggingFaceM4/SugarCrepe_{subset}"


def sugarcrepe_rows_to_local(
    rows: Iterable[dict],
    output_dir: str | Path,
    coco_val_dir: str | Path | None = None,
    subsets: Sequence[str] | None = None,
    require_images_exist: bool = True,
) -> Path:
    """Transform SugarCrepe-format rows into our JSON manifest.

    Supports two row layouts:

    Layout A — HF per-subset repos (HuggingFaceM4/SugarCrepe_*):
        row["image"]          PIL Image
        row["tested_labels"]  [positive_caption, negative_caption]
        row["type"]           subset name (injected before calling this fn)

    Layout B — legacy COCO-filename rows (used by smoke tests):
        row["image"]          COCO val2017 filename string
        row["caption"]        positive caption
        row["negative_caption"] negative caption
        row["type"]           subset name

    Returns the path to the emitted JSON.
    image_root for BinaryDiscrimination is output_dir/images (layout A)
    or coco_val_dir (layout B).
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_filter = set(subsets) if subsets is not None else None

    items: list[dict] = []
    missing_images = 0
    for i, row in enumerate(rows):
        category = row.get("type") or row.get("category") or ""
        if subset_filter is not None and category not in subset_filter:
            continue

        raw_image = row["image"]

        # --- Layout A: PIL image embedded in HF row ---
        if not isinstance(raw_image, (str, Path)):
            positive = row["tested_labels"][0]
            negative = row["tested_labels"][1]
            filename = f"{category}_{i:06d}.jpg"
            images_dir.mkdir(parents=True, exist_ok=True)
            img_path = images_dir / filename
            if not img_path.exists():
                raw_image.convert("RGB").save(img_path, format="JPEG", quality=95)
            items.append({
                "id": f"{category}_{i:06d}",
                "image": filename,
                "positive": positive,
                "negative": negative,
                "category": category,
            })

        # --- Layout B: filename string (smoke tests / legacy) ---
        else:
            filename = str(raw_image)
            if coco_val_dir is not None:
                img_full = Path(coco_val_dir) / filename
                if require_images_exist and not img_full.exists():
                    missing_images += 1
                    continue
            positive = row.get("caption") or row.get("positive") or ""
            negative = row.get("negative_caption") or row.get("negative") or ""
            items.append({
                "id": f"{category or 'sc'}_{i:06d}",
                "image": filename,
                "positive": positive,
                "negative": negative,
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
    coco_val_dir: str | Path | None = None,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    subsets: Sequence[str] | None = DEFAULT_SUBSETS,
) -> Path:
    """Pull SugarCrepe rows from HF and emit our manifest.

    HuggingFaceM4 hosts one repo per subset: HuggingFaceM4/SugarCrepe_{subset}.
    hf_repo overrides the per-subset template (for custom mirrors).
    coco_val_dir is unused for the standard HF repos (images are embedded).
    """
    from datasets import load_dataset

    target_subsets = list(subsets) if subsets is not None else list(SUGARCREPE_SUBSETS)
    rows: list[dict] = []
    for subset in target_subsets:
        repo = hf_repo or _HF_REPO_TEMPLATE.format(subset=subset)
        ds = load_dataset(repo, token=hf_token, cache_dir=cache_dir)
        n_before = len(rows)
        for split_name in ds.keys():
            for row in ds[split_name]:
                row = dict(row)
                row.setdefault("type", subset)
                rows.append(row)
        print(f"  {subset}: {len(rows) - n_before} rows")

    return sugarcrepe_rows_to_local(
        rows, output_dir=output_dir, coco_val_dir=coco_val_dir, subsets=subsets,
    )
