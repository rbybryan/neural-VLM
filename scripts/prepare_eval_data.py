"""CLI to pull benchmark data from HuggingFace and emit our manifest format.

Usage
-----
    # Winoground (needs HF_TOKEN after accepting terms)
    python scripts/prepare_eval_data.py winoground \\
        --output-dir data/eval/winoground

    # SugarCrepe (needs MS-COCO val2017 downloaded separately)
    python scripts/prepare_eval_data.py sugarcrepe \\
        --coco-val-dir /path/to/coco/val2017 \\
        --output-dir data/eval/sugarcrepe \\
        [--subsets replace_obj swap_att]   # default = all 7
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.loaders import (                                            # noqa: E402
    SUGARCREPE_SUBSETS,
    load_sugarcrepe,
    load_winoground,
)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_w = sub.add_parser("winoground")
    p_w.add_argument("--output-dir", required=True, type=Path)
    p_w.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p_w.add_argument("--cache-dir", default=None)
    p_w.add_argument("--split", default="test")

    p_s = sub.add_parser("sugarcrepe")
    p_s.add_argument("--output-dir", required=True, type=Path)
    p_s.add_argument("--coco-val-dir", required=True, type=Path)
    p_s.add_argument("--hf-repo", default="HuggingFaceM4/SugarCrepe")
    p_s.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p_s.add_argument("--cache-dir", default=None)
    p_s.add_argument("--subsets", nargs="+", default=list(SUGARCREPE_SUBSETS))

    args = parser.parse_args()

    if args.cmd == "winoground":
        if not args.hf_token:
            print("warning: no HF token supplied; Winoground is gated and the load may fail")
        json_path, image_dir = load_winoground(
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            split=args.split,
        )
        print(f"wrote {json_path}  (images in {image_dir})")

    elif args.cmd == "sugarcrepe":
        if not args.coco_val_dir.exists():
            print(f"error: --coco-val-dir does not exist: {args.coco_val_dir}", file=sys.stderr)
            sys.exit(2)
        json_path = load_sugarcrepe(
            output_dir=args.output_dir,
            coco_val_dir=args.coco_val_dir,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            subsets=args.subsets,
        )
        print(f"wrote {json_path}  (image_root: {args.coco_val_dir})")


if __name__ == "__main__":
    main()
