from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_nafnet import NAFNet, nafnet_width32
from ihc_torch import (
    build_model_from_checkpoint,
    checkpoint_model_specification,
    model_specification,
)


def small_nafnet(out_channels: int = 1) -> NAFNet:
    return NAFNet(
        out_channels=out_channels,
        width=4,
        enc_blk_nums=(1, 1, 1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1, 1, 1),
    )


class NAFNetTests(unittest.TestCase):
    def test_output_shape_range_and_configurable_channels(self) -> None:
        model = small_nafnet(out_channels=3)
        result = model(torch.rand(2, 1, 16, 32)).detach()

        self.assertEqual(tuple(result.shape), (2, 3, 16, 32))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_backward_reaches_trainable_parameters(self) -> None:
        model = small_nafnet()
        result = model(torch.rand(2, 1, 16, 16))
        result.square().mean().backward()

        self.assertIsNotNone(model.ending.weight.grad)
        self.assertTrue(torch.isfinite(model.ending.weight.grad).all())
        self.assertGreater(float(model.ending.weight.grad.abs().sum()), 0.0)
        first_beta = model.encoders[0][0].beta
        self.assertIsNotNone(first_beta.grad)
        self.assertTrue(torch.isfinite(first_beta.grad).all())

    def test_non_divisible_input_is_padded_and_cropped(self) -> None:
        model = small_nafnet(out_channels=2)
        source = torch.rand(1, 1, 17, 29)

        result = model(source)

        self.assertEqual(tuple(result.shape), (1, 2, 17, 29))
        self.assertEqual(model.padder_size, 16)

    def test_rejects_conditioning_tensor(self) -> None:
        model = small_nafnet()
        with self.assertRaisesRegex(ValueError, "does not support conditioning"):
            model(torch.rand(1, 1, 16, 16), torch.zeros(1, 1))

    def test_zeroed_network_has_no_global_input_residual(self) -> None:
        model = small_nafnet()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        dark = model(torch.zeros(1, 1, 16, 16))
        bright = model(torch.ones(1, 1, 16, 16))

        expected = torch.full_like(dark, 0.5)
        self.assertTrue(torch.equal(dark, expected))
        self.assertTrue(torch.equal(bright, expected))

    def test_default_width32_parameter_count(self) -> None:
        model = nafnet_width32(out_channels=1)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 17_110_753)
        torch.testing.assert_close(model.ending.bias, torch.full_like(model.ending.bias, -2.0))

    def test_checkpoint_factory_roundtrip_and_metadata_validation(self) -> None:
        model = nafnet_width32(out_channels=1)
        specification = model_specification("nafnet-width32")
        checkpoint = {
            "model": model.state_dict(),
            "markers": ("CD68",),
            "base_channels": 32,
            "block_type": "residual",
            "conditioning": {"version": 1, "mode": "none"},
            "model_spec": specification,
            "args": {
                "model_arch": "nafnet-width32",
                "model_spec": specification,
                "condition_mode": "none",
            },
        }

        loaded = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(loaded, NAFNet)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        torch.testing.assert_close(loaded.ending.weight, model.ending.weight)

        checkpoint["args"] = {**checkpoint["args"], "model_arch": "unet"}
        with self.assertRaisesRegex(ValueError, "architecture disagrees"):
            checkpoint_model_specification(checkpoint)


if __name__ == "__main__":
    unittest.main()
