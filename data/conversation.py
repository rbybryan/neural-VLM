"""Unified conversation schema and JSON-backed dataset.

A conversation is a list of {role, content} dicts plus a single image path.
This is the LLaVA-style schema used by ShareGPT4V, LLaVA-1.5-mix, and
LLaVA-OneVision-SI. Other formats (Cambrian-7M, etc.) can be adapted to
this schema with a per-dataset converter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Conversation:
    image_path: str
    messages: list[dict[str, str]]   # [{role: 'user'|'assistant'|'system', content: str}]

    @staticmethod
    def from_llava(item: dict, image_root: Path) -> "Conversation":
        """Convert LLaVA's {image, conversations} dict to our schema.

        LLaVA uses {from: 'human'|'gpt'|'system', value: str}; we map to
        user/assistant/system. The first user turn typically contains an
        '<image>' placeholder; we strip it because our model handles vision
        out-of-band (visual tokens are prepended in LLMWrapper).
        """
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = []
        for m in item["conversations"]:
            role = role_map[m["from"]]
            content = m["value"].replace("<image>", "").strip()
            messages.append({"role": role, "content": content})
        return Conversation(
            image_path=str(image_root / item["image"]),
            messages=messages,
        )


class ConversationDataset(Dataset):
    """JSON-backed conversation dataset.

    Expects a JSON file with a list of LLaVA-format items:
      [{"image": "path.jpg", "conversations": [{"from": "human", "value": ...}, ...]}, ...]

    Items whose image path is in `excluded_image_paths` (the decontamination
    output) are filtered out at dataset construction time.
    """

    def __init__(
        self,
        json_path: str | Path,
        image_root: str | Path,
        image_transform: Callable | None = None,
        excluded_image_paths: set[str] | None = None,
    ):
        self.json_path = Path(json_path)
        self.image_root = Path(image_root)
        self.image_transform = image_transform
        excluded = excluded_image_paths or set()

        with open(self.json_path) as f:
            items = json.load(f)

        if excluded:
            items = [
                i for i in items
                if str(self.image_root / i["image"]) not in excluded
            ]

        self.items = items
        self.n_excluded = len(excluded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        conv = Conversation.from_llava(item, self.image_root)
        image = Image.open(conv.image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        return {"image": image, "messages": conv.messages}
