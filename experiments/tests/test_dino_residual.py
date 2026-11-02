from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from ihc_dino_residual import UNeXt2DinoResidualVirtualStainer
from ihc_spatial import SPATIAL_THUMBNAIL_CONDITION_DIMENSION
from cv_protocol import load_protocol
from train_unet import build_optimizer


class DinoResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_initialized_residual_shape_and_updates(self):
        model = UNeXt2DinoResidualVirtualStainer(4, "spatial-thumbnail-film")
        x = torch.rand(1, 1, 64, 64)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        output = model(x, condition)
        self.assertEqual(output.shape, (1, 4, 64, 64))
        final = model.dinov2_residual.refinement[-1]
        self.assertTrue(torch.equal(final.weight.detach(), torch.zeros_like(final.weight)))
        output.mean().backward()
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.0)
        optimizer, groups = build_optimizer(model, "unext2-fcmae-tiny-dinov2-residual", 3e-4, None, 1e-4)
        rates = {group["name"]: group["initial_learning_rate"] for group in groups}
        self.assertEqual(rates["encoder"], 3e-4)
        self.assertAlmostEqual(rates["dinov2_encoder"], 3e-5)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})

    def test_protocol_uses_effective_batch_sixteen(self):
        protocol = load_protocol(ROOT / "configs/round113_cv2_dino_residual.json")
        recipe = protocol["recipes"]["dinov2residual"]
        self.assertEqual(recipe["batch_size"] * recipe["gradient_accumulation_steps"], 16)
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertNotIn("encoder_learning_rate_multiplier", recipe)


if __name__ == "__main__":
    unittest.main()
