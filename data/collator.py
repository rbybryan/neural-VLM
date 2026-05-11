"""Collator that turns a batch of {image, messages} into model inputs.

Responsibilities:
  - Apply the tokenizer's chat template to render messages as text
  - Tokenize twice (prompt-only and prompt+response) to compute the
    supervision mask: only assistant response tokens are supervised
  - Pad text token sequences in the batch
  - Stack images

Output keys:
  image          : (B, 3, H, W)
  input_ids      : (B, T)
  attention_mask : (B, T)
  labels         : (B, T)  with -100 on prompt / padding tokens

The model's _loss handles the visual-prefix padding (-100 for the N_vis
visual tokens that get prepended in LLMWrapper) and the next-token shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class VLMCollator:
    tokenizer: object
    max_length: int = 1024

    def __call__(self, batch: list[dict]) -> dict:
        images = torch.stack([b["image"] for b in batch])

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        for b in batch:
            messages = b["messages"]
            assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            if not assistant_idxs:
                raise ValueError("conversation has no assistant turn")
            last_a = assistant_idxs[-1]

            prompt_text = self.tokenizer.apply_chat_template(
                messages[:last_a], tokenize=False, add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                messages[: last_a + 1], tokenize=False, add_generation_prompt=False,
            )

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            labels = list(full_ids)
            mask_len = min(len(prompt_ids), len(labels))
            for i in range(mask_len):
                labels[i] = -100

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
