from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from ihc_feature_guidance import FrozenFCMAEPerceptualLoss
import train_unet


class ToyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 2, kernel_size=1)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[[[1.0]]], [[[-1.0]]]]))
            self.projection.bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, inputs: torch.Tensor) -> tuple[list[torch.Tensor], None]:
        feature = self.projection(inputs)
        return [feature, feature, feature, feature], None


class FeatureGuidanceTests(unittest.TestCase):
    def test_loss_backpropagates_only_to_prediction(self) -> None:
        teacher = ToyTeacher()
        loss_fn = FrozenFCMAEPerceptualLoss(
            teacher, stage=4, crop_size=16, batch_samples=2
        )
        prediction = torch.full((3, 1, 20, 20), 0.25, requires_grad=True)
        target = torch.full((3, 1, 20, 20), 0.75, requires_grad=True)

        loss = loss_fn(prediction, target)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))

    def test_option_validation_is_strict(self) -> None:
        valid = SimpleNamespace(
            feature_guidance_weight=0.05,
            feature_guidance_checkpoint="teacher.pt",
            feature_guidance_checkpoint_sha256="a" * 64,
            feature_guidance_marker="CD68",
            feature_guidance_stage=4,
            feature_guidance_crop_size=256,
            feature_guidance_batch_samples=4,
            loss_profile="hybrid",
            model_arch="unext2-fcmae-tiny",
        )
        train_unet.validate_feature_guidance_options(valid, ("CD68", "CD45RO"))
        for changed, message in (
            ({"feature_guidance_weight": -0.1}, "finite and non-negative"),
            ({"feature_guidance_stage": 5}, "stage"),
            ({"feature_guidance_crop_size": 255}, "divisible by 16"),
            ({"feature_guidance_batch_samples": 0}, "positive"),
            ({"loss_profile": "exact-score"}, "hybrid"),
            ({"feature_guidance_checkpoint_sha256": "bad"}, "SHA-256"),
        ):
            options = SimpleNamespace(**vars(valid))
            for key, value in changed.items():
                setattr(options, key, value)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, message):
                train_unet.validate_feature_guidance_options(
                    options, ("CD68", "CD45RO")
                )


if __name__ == "__main__":
    unittest.main()
