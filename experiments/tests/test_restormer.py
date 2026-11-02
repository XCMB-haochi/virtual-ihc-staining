from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_restormer import Restormer, restormer_small
from ihc_torch import (
    build_model_from_checkpoint,
    checkpoint_model_specification,
    model_specification,
)


def tiny_restormer(out_channels: int = 1) -> Restormer:
    return Restormer(
        out_channels=out_channels,
        dim=8,
        num_blocks=(1, 1, 1, 1),
        refinement_blocks=1,
        heads=(1, 2, 4, 8),
        ffn_expansion=2.0,
    )


class RestormerTests(unittest.TestCase):
    def test_output_shape_range_and_non_divisible_crop(self) -> None:
        model = tiny_restormer(out_channels=2)
        result = model(torch.rand(1, 1, 17, 29)).detach()

        self.assertEqual(tuple(result.shape), (1, 2, 17, 29))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_backward_reaches_attention_and_output(self) -> None:
        model = tiny_restormer()
        result = model(torch.rand(2, 1, 16, 16))
        result.square().mean().backward()

        for parameter in (
            model.encoder_level1[0].attention.qkv.weight,
            model.output.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_rejects_invalid_input_and_conditioning(self) -> None:
        model = tiny_restormer()
        with self.assertRaisesRegex(ValueError, "does not support conditioning"):
            model(torch.rand(1, 1, 16, 16), torch.zeros(1, 1))
        with self.assertRaisesRegex(ValueError, "one input channel"):
            model(torch.rand(1, 2, 16, 16))

    def test_zeroed_network_has_no_global_input_residual(self) -> None:
        model = tiny_restormer()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        dark = model(torch.zeros(1, 1, 16, 16))
        bright = model(torch.ones(1, 1, 16, 16))

        expected = torch.full_like(dark, 0.5)
        self.assertTrue(torch.equal(dark, expected))
        self.assertTrue(torch.equal(bright, expected))

    def test_default_model_metadata_and_head_bias(self) -> None:
        model = restormer_small(out_channels=1)
        specification = model_specification("restormer-small")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 6_707_543)
        self.assertEqual(specification["dim"], 32)
        self.assertEqual(specification["num_blocks"], [2, 3, 3, 4])
        self.assertEqual(specification["heads"], [1, 2, 4, 8])
        torch.testing.assert_close(
            model.output.bias,
            torch.full_like(model.output.bias, -2.0),
        )

    def test_checkpoint_factory_roundtrip_and_strict_specification(self) -> None:
        model = restormer_small(out_channels=1)
        specification = model_specification("restormer-small")
        checkpoint = {
            "model": model.state_dict(),
            "markers": ("CD68",),
            "base_channels": 32,
            "block_type": "residual",
            "conditioning": {"version": 1, "mode": "none"},
            "model_spec": specification,
            "args": {
                "model_arch": "restormer-small",
                "model_spec": specification,
                "condition_mode": "none",
            },
        }

        loaded = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(loaded, Restormer)
        torch.testing.assert_close(loaded.output.weight, model.output.weight)

        invalid = copy.deepcopy(checkpoint)
        invalid["model_spec"]["dim"] = 48
        with self.assertRaisesRegex(ValueError, "Unsupported Restormer-small"):
            checkpoint_model_specification(invalid)


if __name__ == "__main__":
    unittest.main()
