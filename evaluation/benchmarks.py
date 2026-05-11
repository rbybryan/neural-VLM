"""Benchmark adapters for evaluation.

Each adapter knows:
  - how to enumerate (image, prompt, continuation) triples from a
    benchmark's items
  - how to aggregate per-query log-likelihood scores into headline metrics
    matching the benchmark's standard reporting convention

v1 covers Winoground (image-text matching with text/image/group scores)
and BinaryDiscrimination (the shape used by SugarCrepe, ARO-relation,
ARO-attribution, ARO-order). MMMU/MMBench/MMVet need answer-parsing on
top of free-form generation and are deferred.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from math import inf
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from .scoring import ScoringQuery


def _load_image(path: Path, transform: Callable | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img) if transform is not None else img


class Benchmark:
    name: str = "benchmark"

    def queries(self) -> Iterator[ScoringQuery]:
        raise NotImplementedError

    def aggregate(self, scores: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError


class BinaryDiscrimination(Benchmark):
    """SugarCrepe / ARO-style: each item has one image and a (positive,
    negative) caption pair. The model must score positive > negative.

    Items format (list of dict):
      {"id": str, "image": "relative/path.jpg",
       "positive": str, "negative": str, "category": str (optional)}

    The default prompt is empty ("describe the image" intent encoded via
    the chat template's user turn convention — keeps the continuation
    log-likelihoods comparable across candidates). Override via `prompt`.
    """

    name = "binary_discrim"

    def __init__(
        self,
        items: list[dict] | str | Path,
        image_root: str | Path,
        image_transform: Callable | None = None,
        prompt: str = "Describe this image:",
        name: str | None = None,
    ):
        if isinstance(items, (str, Path)):
            with open(items) as f:
                items = json.load(f)
        self.items = items
        self.image_root = Path(image_root)
        self.image_transform = image_transform
        self.prompt = prompt
        if name is not None:
            self.name = name

    def queries(self) -> Iterator[ScoringQuery]:
        for item in self.items:
            image = _load_image(self.image_root / item["image"], self.image_transform)
            for label in ("pos", "neg"):
                key = "positive" if label == "pos" else "negative"
                yield ScoringQuery(
                    image=image,
                    prompt=self.prompt,
                    continuation=item[key],
                    query_id=f"{item['id']}__{label}",
                    group_id=str(item["id"]),
                    metadata={"label": label, "category": item.get("category", "")},
                )

    def aggregate(self, scores: dict[str, float]) -> dict[str, float]:
        groups: dict[str, dict[str, float]] = {}
        categories: dict[str, str] = {}
        for qid, s in scores.items():
            gid, label = qid.rsplit("__", 1)
            groups.setdefault(gid, {})[label] = s
        for item in self.items:
            categories[str(item["id"])] = item.get("category", "")

        total = 0
        correct = 0
        by_category: dict[str, list[bool]] = {}
        for gid, g in groups.items():
            if "pos" not in g or "neg" not in g:
                continue
            total += 1
            ok = g["pos"] > g["neg"]
            correct += int(ok)
            cat = categories.get(gid, "")
            if cat:
                by_category.setdefault(cat, []).append(ok)

        out = {
            "accuracy": correct / max(total, 1),
            "n_items": total,
        }
        for cat, vals in sorted(by_category.items()):
            out[f"accuracy/{cat}"] = sum(vals) / max(len(vals), 1)
            out[f"n_items/{cat}"] = len(vals)
        return out


class Winoground(Benchmark):
    """Winoground item-format: each item has 2 images and 2 captions; the
    correct pairings are (image_0, caption_0) and (image_1, caption_1).

    Items format (list of dict):
      {"id": str, "image_0": "path", "image_1": "path",
       "caption_0": str, "caption_1": str}

    Per Thrush et al. 2022:
      text_score(i)  = 1 if for *both* images, the model prefers the
                       correct caption over the other.
      image_score(i) = 1 if for *both* captions, the model prefers the
                       correct image over the other.
      group_score(i) = 1 if both text_score(i) and image_score(i).

    Headline metric for the paper is group_score.
    """

    name = "winoground"

    def __init__(
        self,
        items: list[dict] | str | Path,
        image_root: str | Path,
        image_transform: Callable | None = None,
        prompt: str = "Describe this image:",
    ):
        if isinstance(items, (str, Path)):
            with open(items) as f:
                items = json.load(f)
        self.items = items
        self.image_root = Path(image_root)
        self.image_transform = image_transform
        self.prompt = prompt

    def queries(self) -> Iterator[ScoringQuery]:
        for item in self.items:
            imgs = [
                _load_image(self.image_root / item["image_0"], self.image_transform),
                _load_image(self.image_root / item["image_1"], self.image_transform),
            ]
            caps = [item["caption_0"], item["caption_1"]]
            for j in (0, 1):
                for k in (0, 1):
                    yield ScoringQuery(
                        image=imgs[j],
                        prompt=self.prompt,
                        continuation=caps[k],
                        query_id=f"{item['id']}__im{j}_c{k}",
                        group_id=str(item["id"]),
                    )

    def aggregate(self, scores: dict[str, float]) -> dict[str, float]:
        groups: dict[str, list[list[float | None]]] = {}
        for qid, s in scores.items():
            gid, suffix = qid.rsplit("__im", 1)
            j_str, k_str = suffix.split("_c")
            j, k = int(j_str), int(k_str)
            groups.setdefault(gid, [[None, None], [None, None]])
            groups[gid][j][k] = s

        n = 0
        text_correct = image_correct = group_correct = 0
        for gid, s in groups.items():
            if any(s[j][k] is None for j in (0, 1) for k in (0, 1)):
                continue
            n += 1
            text_ok = (s[0][0] > s[0][1]) and (s[1][1] > s[1][0])
            image_ok = (s[0][0] > s[1][0]) and (s[1][1] > s[0][1])
            text_correct += int(text_ok)
            image_correct += int(image_ok)
            group_correct += int(text_ok and image_ok)

        return {
            "text_score": text_correct / max(n, 1),
            "image_score": image_correct / max(n, 1),
            "group_score": group_correct / max(n, 1),
            "n_items": n,
        }
