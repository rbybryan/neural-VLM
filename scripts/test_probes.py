"""Smoke test for the three §5.B.6 mechanism probes.

Uses the synthetic smoke model. At init, all gates are zero -> modulation
is identically zero -> the probes correctly return null results
(biased_competition score ~ 0, modulation magnitudes ~ 0, persistent-
token decoding at chance because the tokens carry no image content).

This is the right invariant to test: the probes report "no mechanism
observed" *when no mechanism is present*. After Stage-2 training, when
gates have unlocked, the same probes will report positive findings; the
spec then earns the right to use "biased competition" / "persistent
context tokens" framing in the paper.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from data import build_eval_transform                                          # noqa: E402
from evaluation import (                                                        # noqa: E402
    BiasedCompetitionProbe,
    ModulationAnalyzer,
    PersistentTokenDecoder,
)
from scripts.build_main import build_smoke_model                              # noqa: E402
from scripts.test_data_pipeline import StubTokenizer                          # noqa: E402


def _composite_image(seed: int, image_size: int = 224):
    """Make a composite image with two halves of different random patterns.
    The 'left' half will be the target region; the 'right' half is the
    distractor."""
    rng = torch.Generator().manual_seed(seed)
    arr = (torch.rand(image_size, image_size, 3, generator=rng) * 255).to(torch.uint8)
    # Make halves visibly different in their pattern statistics
    arr[:, image_size // 2:, :] = (arr[:, image_size // 2:, :] // 2)  # darker right
    return Image.fromarray(arr.numpy(), mode="RGB")


def _half_masks(feature_h: int, feature_w: int):
    """Boolean masks for left half / right half of a feature map."""
    left = torch.zeros(feature_h, feature_w, dtype=torch.bool)
    right = torch.zeros(feature_h, feature_w, dtype=torch.bool)
    mid = feature_w // 2
    left[:, :mid] = True
    right[:, mid:] = True
    return left, right


def main():
    torch.manual_seed(0)
    model = build_smoke_model().eval()
    tokenizer = StubTokenizer(vocab_size=2048)
    transform = build_eval_transform(image_size=224)

    # convnextv2_atto at 224 -> stage-3 feature map is 7x7
    # we'll probe stage 3 (the deepest injected stage)
    stage_idx = 3
    H, W = 7, 7
    left_mask, right_mask = _half_masks(H, W)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_pil = [_composite_image(seed=100 + i) for i in range(4)]
        for i, img in enumerate(images_pil):
            img.save(tmp_path / f"comp_{i}.jpg")
        images = [transform(img) for img in images_pil]

        # ----- biased-competition probe -----
        print("BiasedCompetitionProbe (stage 3, 4 trials)")
        bc_probe = BiasedCompetitionProbe(model=model, tokenizer=tokenizer, stage_idx=stage_idx)
        trials = [
            {
                "image": img,
                "target_prompt": "Look at the left side.",
                "distractor_prompt": "Look at the right side.",
                "target_region": left_mask,
                "distractor_region": right_mask,
            }
            for img in images
        ]
        results = bc_probe.measure_many(trials)
        for i, r in enumerate(results):
            print(f"  trial {i}: T_T={r.T_T:.3e} T_D={r.T_D:.3e} "
                  f"D_T={r.D_T:.3e} D_D={r.D_D:.3e}  "
                  f"score={r.biased_competition_score:.3e}")
        max_abs = max(abs(r.biased_competition_score) for r in results)
        print(f"  max |bc_score| = {max_abs:.3e}")
        assert max_abs < 1e-3, (
            f"At init (gates=0), modulation is identically zero and all four "
            f"magnitudes should be 0. Got max |score| = {max_abs}"
        )
        print("  PASS: probe correctly reports null at init")
        print()

        # ----- modulation analyzer -----
        print("ModulationAnalyzer (4 samples with synthetic difficulty)")
        mod_analyzer = ModulationAnalyzer(model=model, tokenizer=tokenizer)
        samples = [
            {
                "image": img,
                "prompt": "Describe this image:",
                "id": f"img_{i}",
                "difficulty": float(i),  # synthetic monotonic difficulty
            }
            for i, img in enumerate(images)
        ]
        stats = mod_analyzer.analyze(samples)
        print(f"  per-image keys: {list(stats.per_image.keys())}")
        for stage_key, summary in stats.summary.items():
            print(f"  {stage_key}: {summary}")
            assert summary["mean"] < 1e-4, \
                f"At init, modulation must be ~0; got {stage_key} mean={summary['mean']}"
        print("  PASS: modulation magnitudes ~0 at init")
        print()

        # ----- persistent-token decoder -----
        print("PersistentTokenDecoder (4 images x 2 fake classes)")
        decoder = PersistentTokenDecoder(model=model, tokenizer=tokenizer)
        features = decoder.extract_features(images)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        print(f"  extracted features: {tuple(features.shape)}")
        print(f"  token feature dim = 32 tokens x {features.shape[1] // 32} hidden = {features.shape[1]}")

        # At init: tokens are the same constant `self.memory.tokens` for every
        # image (update_gate=0 means read_write is a no-op). So features are
        # identical across images and the probe cannot exceed chance.
        std_across_images = features.std(dim=0).mean().item()
        print(f"  feature std across 4 images = {std_across_images:.3e}")
        assert std_across_images < 1e-5, (
            f"At init, persistent tokens are image-invariant (update_gate=0), "
            f"so feature std across images must be ~0. Got {std_across_images}"
        )

        probe_result = decoder.probe(features, labels, train_frac=0.5, seed=0)
        print(f"  probe: {probe_result}")
        # Test accuracy is at-chance because all features are identical.
        # Logistic regression on degenerate features may give 0.5 or 0.0
        # depending on the constant predictor it falls into; either way
        # it's not "above chance".
        assert not probe_result["above_chance"], (
            "At init with image-invariant tokens, the probe should not "
            "exceed chance; if it does, something is leaking image info "
            "outside the memory bank"
        )
        print("  PASS: probe at-chance with image-invariant tokens (expected)")
        print()

        print("test_probes: PASSED")


if __name__ == "__main__":
    main()
