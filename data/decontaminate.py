"""Perceptual-hash decontamination.

Per PREREGISTRATION.md §4: training data is filtered against eval images at
PHash Hamming distance <= 4 bits. Flagged training images are excluded
before any benchmark number is reported.

Two-step usage so the eval index can be built once and reused:

  # Step 1: index every eval image (Winoground, SugarCrepe, NSD, MMBench, ...)
  python -m data.decontaminate build-index \\
      --eval-dirs /path/to/winoground /path/to/sugarcrepe /path/to/nsd \\
      --output decontamination_index.json

  # Step 2: filter a training image list against that index
  python -m data.decontaminate filter \\
      --train-list train_images.txt \\
      --index decontamination_index.json \\
      --output excluded_training_images.tsv

The TSV has columns: train_path, matched_eval_path, hamming_distance.
The training-time dataset constructor accepts a set of excluded paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imagededup.methods import PHash


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _iter_image_files(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        yield from root.rglob(ext)


def build_eval_index(eval_dirs: list[Path]) -> dict[str, str]:
    """{image_path -> phash_hex} over every image in any of the eval dirs."""
    phasher = PHash()
    index: dict[str, str] = {}
    for d in eval_dirs:
        for p in _iter_image_files(d):
            try:
                h = phasher.encode_image(image_file=str(p))
            except Exception:
                continue
            if h is not None:
                index[str(p)] = h
    return index


def find_decontamination_matches(
    train_image_paths: list[Path],
    eval_index: dict[str, str],
    max_distance: int = 4,
) -> list[tuple[str, str, int]]:
    """For each train path, return the *first* eval match within max_distance,
    or skip. Returns list of (train_path, matched_eval_path, distance)."""
    phasher = PHash()
    eval_items = list(eval_index.items())
    matches: list[tuple[str, str, int]] = []
    for train_path in train_image_paths:
        try:
            train_hash = phasher.encode_image(image_file=str(train_path))
        except Exception:
            continue
        if train_hash is None:
            continue
        for eval_path, eval_hash in eval_items:
            d = _hamming_hex(train_hash, eval_hash)
            if d <= max_distance:
                matches.append((str(train_path), eval_path, d))
                break
    return matches


def load_excluded_set(matches_tsv: Path) -> set[str]:
    """Convenience loader for the dataset constructor's `excluded_image_paths`."""
    excluded: set[str] = set()
    with open(matches_tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                excluded.add(parts[0])
    return excluded


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("build-index")
    p_idx.add_argument("--eval-dirs", nargs="+", required=True, type=Path)
    p_idx.add_argument("--output", required=True, type=Path)

    p_filt = sub.add_parser("filter")
    p_filt.add_argument("--train-list", required=True, type=Path,
                        help="text file with one image path per line")
    p_filt.add_argument("--index", required=True, type=Path,
                        help="phash index JSON from build-index")
    p_filt.add_argument("--max-distance", type=int, default=4)
    p_filt.add_argument("--output", required=True, type=Path,
                        help="TSV: train_path\\tmatched_eval_path\\tdistance (excluded)")

    args = parser.parse_args()

    if args.cmd == "build-index":
        index = build_eval_index(args.eval_dirs)
        with open(args.output, "w") as f:
            json.dump(index, f, indent=2)
        print(f"indexed {len(index)} eval images -> {args.output}")

    elif args.cmd == "filter":
        with open(args.index) as f:
            eval_index = json.load(f)
        with open(args.train_list) as f:
            train_paths = [Path(line.strip()) for line in f if line.strip()]
        matches = find_decontamination_matches(train_paths, eval_index, max_distance=args.max_distance)
        with open(args.output, "w") as f:
            for train_path, eval_path, d in matches:
                f.write(f"{train_path}\t{eval_path}\t{d}\n")
        print(f"checked {len(train_paths)} train images, "
              f"{len(matches)} matched eval at <= {args.max_distance} bits "
              f"-> {args.output}")


if __name__ == "__main__":
    main()
