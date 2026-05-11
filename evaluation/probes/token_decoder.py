"""Persistent-token decoding probe (SPEC_v1.md §5.B.6).

After K=2 forward, the persistent memory bank holds 32 "context tokens"
that have been read/written across iterations. If the gate-magnitude /
biased-competition story is real, those tokens should carry
image-relevant information (otherwise the persistent-token framing
collapses to a static learned prior).

Test: train a linear probe on (extracted_tokens -> image_label) pairs,
report train/test accuracy vs chance.

  - High test accuracy: tokens carry decodable image content.
  - At-chance accuracy: tokens are vestigial — the spec's "persistent
    context tokens" framing is decorative rather than load-bearing.

For SPEC's optional working-memory framing to be earned, this probe
plus the capacity-limit test would both need positive results. v1
ships only this part.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class PersistentTokenDecoder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, prompt: str) -> list[int]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt_text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def extract_features(
        self,
        images: list[torch.Tensor],
        prompt: str = "Describe this image:",
    ) -> torch.Tensor:
        """Returns (N, 32 * D) flattened persistent-token vectors per image."""
        param = next(self.model.parameters())
        device, dtype = param.device, param.dtype

        input_ids = self._tokenize(prompt)
        ids = torch.tensor([input_ids], device=device)
        mask = torch.ones_like(ids)

        orig_K = self.model.cfg.num_iterations
        orig_use = self.model.cfg.use_recurrence
        self.model.cfg.num_iterations = 2
        self.model.cfg.use_recurrence = True

        feats = []
        try:
            for img in images:
                img_b = img.to(device=device, dtype=dtype).unsqueeze(0)
                out = self.model(image=img_b, text_ids=ids, text_mask=mask)
                tokens = out["final_memory_tokens"][0]  # (32, dim)
                feats.append(tokens.flatten().float().cpu())
        finally:
            self.model.cfg.num_iterations = orig_K
            self.model.cfg.use_recurrence = orig_use

        return torch.stack(feats)

    def probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_frac: float = 0.5,
        seed: int = 0,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> dict:
        """Train a logistic regression on `features -> labels`, report
        train/test accuracy and chance baseline.

        features: (N, D); labels: (N,) int.
        """
        X = features.numpy()
        y = labels.numpy()
        n_classes = int(y.max() + 1)
        N = X.shape[0]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        n_train = max(1, int(N * train_frac))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        chance = 1.0 / max(n_classes, 1)
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            return {
                "train_accuracy": 0.0,
                "test_accuracy": 0.0,
                "above_chance": False,
                "chance": chance,
                "n_classes": n_classes,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "note": "insufficient class diversity in train set; probe skipped",
            }

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X[train_idx], y[train_idx])
        train_acc = float(clf.score(X[train_idx], y[train_idx]))
        test_acc = float(clf.score(X[test_idx], y[test_idx]))

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "above_chance": test_acc > chance + 1e-6,
            "chance": chance,
            "n_classes": n_classes,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
