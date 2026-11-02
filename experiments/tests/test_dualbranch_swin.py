from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_unet
from ihc_dualbranch import DualBranchSwinSmall
from ihc_torch import (
    UNet,
    build_model,
    build_model_from_checkpoint,
    checkpoint_model_specification,
    model_specification,
)


class DualBranchSwinSmallTests(unittest.TestCase):
    def test_output_contract_for_regular_and_odd_shapes(self) -> None:
        model = DualBranchSwinSmall(out_channels=2).eval()

        with torch.no_grad():
            regular = model(torch.rand(1, 1, 32, 48))
            odd = model(torch.rand(1, 1, 33, 49))

        self.assertEqual(tuple(regular.shape), (1, 2, 32, 48))
        self.assertEqual(tuple(odd.shape), (1, 2, 33, 49))
        self.assertGreaterEqual(float(odd.min()), 0.0)
        self.assertLessEqual(float(odd.max()), 1.0)

    def test_rejects_invalid_input_and_conditioning(self) -> None:
        model = DualBranchSwinSmall()
        with self.assertRaisesRegex(ValueError, "does not support conditioning"):
            model(torch.rand(1, 1, 16, 16), torch.zeros(1, 1))
        with self.assertRaisesRegex(ValueError, "expects input shape"):
            model(torch.rand(1, 3, 16, 16))
        with self.assertRaisesRegex(ValueError, ">= 16"):
            model(torch.rand(1, 1, 15, 16))

    def test_zero_gates_preserve_seed_matched_base32_unet(self) -> None:
        torch.manual_seed(1234)
        control = UNet(out_channels=2, base_channels=32, block_type="residual").eval()
        torch.manual_seed(1234)
        candidate = DualBranchSwinSmall(out_channels=2).eval()

        candidate_state = candidate.state_dict()
        for key, value in control.state_dict().items():
            torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)
        for fusion in candidate.fusions:
            self.assertEqual(float(fusion.gate.detach()), 0.0)

        source = torch.rand(1, 1, 32, 48)
        with torch.no_grad():
            expected = control(source)
            actual = candidate(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_reaches_local_global_and_fusion_parameters(self) -> None:
        model = DualBranchSwinSmall()
        with torch.no_grad():
            for fusion in model.fusions:
                fusion.gate.fill_(1.0)

        model(torch.rand(1, 1, 32, 32)).mean().backward()

        checked = (
            model.encoders[0].block[0].weight,
            model.patch_embed.weight,
            model.swin_stages[0].blocks[0].attn.qkv.weight,
            model.fusions[0].project.weight,
            model.fusions[0].gate,
            model.head.weight,
        )
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_fixed_parameter_count_and_model_specification(self) -> None:
        model = build_model("dualbranch-swin-s", out_channels=1)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertIsInstance(model, DualBranchSwinSmall)
        self.assertEqual(parameter_count, 10_410_134)
        specification = model_specification("dualbranch-swin-s")
        self.assertEqual(specification["local_base_channels"], 32)
        self.assertEqual(specification["swin_dims"], [48, 96, 192])
        self.assertEqual(specification["swin_depths"], [2, 2, 4])
        self.assertEqual(specification["swin_heads"], [3, 6, 12])
        self.assertEqual(specification["fusion_mode"], "zero-gated-add")

        with self.assertRaisesRegex(ValueError, "fixed local base_channels=32"):
            model_specification("dualbranch-swin-s", base_channels=48)
        with self.assertRaisesRegex(ValueError, "fixed residual"):
            model_specification("dualbranch-swin-s", block_type="dense")

    def test_training_optimizer_covers_all_parameters_in_one_group(self) -> None:
        model = build_model("dualbranch-swin-s", out_channels=1)
        optimizer, report = train_unet.build_optimizer(
            model,
            "dualbranch-swin-s",
            learning_rate=3e-4,
            encoder_learning_rate=None,
            weight_decay=1e-4,
        )

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(report[0]["name"], "all")
        self.assertEqual(report[0]["parameters"], 10_410_134)
        grouped = optimizer.param_groups[0]["params"]
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )

    def test_checkpoint_factory_roundtrip_and_strict_metadata(self) -> None:
        source = DualBranchSwinSmall(out_channels=1)
        specification = model_specification("dualbranch-swin-s")
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "base_channels": 32,
            "block_type": "residual",
            "conditioning": {"version": 1, "mode": "none"},
            "model_spec": specification,
            "args": {
                "model_arch": "dualbranch-swin-s",
                "model_spec": specification,
                "condition_mode": "none",
            },
        }

        loaded = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(loaded, DualBranchSwinSmall)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key in (
            "encoders.0.block.0.weight",
            "patch_embed.weight",
            "fusions.2.gate",
            "head.bias",
        ):
            torch.testing.assert_close(
                loaded.state_dict()[key], source.state_dict()[key]
            )

        invalid_specification = {
            **specification,
            "swin_depths": [2, 2, 2],
        }
        checkpoint["model_spec"] = invalid_specification
        checkpoint["args"] = {
            **checkpoint["args"],
            "model_spec": invalid_specification,
        }
        with self.assertRaisesRegex(ValueError, "Unsupported DualBranch-Swin-S"):
            checkpoint_model_specification(checkpoint)


if __name__ == "__main__":
    unittest.main()
