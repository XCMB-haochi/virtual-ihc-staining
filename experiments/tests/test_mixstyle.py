from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_unet
from ihc_torch import MixStyle, UNet


class MixStyleTests(unittest.TestCase):
    def test_eval_probability_zero_and_singleton_are_exact_identity(self) -> None:
        inputs = torch.rand(4, 3, 8, 8)
        module = MixStyle(probability=1.0, alpha=0.1).eval()
        self.assertTrue(torch.equal(module(inputs), inputs))

        module.train()
        module.configure(0.0, 0.1)
        self.assertTrue(torch.equal(module(inputs), inputs))

        module.configure(1.0, 0.1)
        singleton = inputs[:1]
        self.assertTrue(torch.equal(module(singleton), singleton))

    def test_training_mixes_statistics_and_preserves_gradient(self) -> None:
        torch.manual_seed(42)
        inputs = torch.stack(
            tuple(torch.rand(2, 8, 8) * (index + 1) + index for index in range(4))
        ).requires_grad_()
        module = MixStyle(probability=1.0, alpha=0.1).train()

        mixed = module(inputs)

        self.assertEqual(mixed.shape, inputs.shape)
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertFalse(torch.equal(mixed, inputs))
        mixed.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)

    def test_unet_configuration_adds_no_state_and_is_eval_invariant(self) -> None:
        torch.manual_seed(7)
        model = UNet(out_channels=1, base_channels=4).eval()
        source = torch.rand(2, 1, 32, 32)
        state_keys = tuple(model.state_dict())
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        with torch.no_grad():
            baseline = model(source)

        model.configure_mixstyle(probability=0.5, alpha=0.1)
        with torch.no_grad():
            configured = model(source)

        self.assertTrue(torch.equal(configured, baseline))
        self.assertEqual(tuple(model.state_dict()), state_keys)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            parameter_count,
        )

    def test_configuration_and_training_options_are_validated(self) -> None:
        for probability in (-0.1, 1.1, float("nan")):
            with self.subTest(probability=probability), self.assertRaisesRegex(
                ValueError, "probability"
            ):
                train_unet.validate_mixstyle_options(probability, 0.1, "unet")
        for alpha in (0.0, -0.1, float("inf")):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "alpha"):
                train_unet.validate_mixstyle_options(0.5, alpha, "unet")
        with self.assertRaisesRegex(ValueError, "alpha"):
            MixStyle(probability=0.5, alpha=float("inf"))
        with self.assertRaisesRegex(ValueError, "only by U-Net"):
            train_unet.validate_mixstyle_options(0.5, 0.1, "restormer-small")


if __name__ == "__main__":
    unittest.main()
