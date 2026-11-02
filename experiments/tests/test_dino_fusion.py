from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ihc_dino_fusion import (
    UNeXt2DinoBottleneckFusionVirtualStainer,
    _align_dino_to_bottleneck,
)
from ihc_spatial import SPATIAL_THUMBNAIL_CONDITION_DIMENSION
from ihc_torch import build_model_from_checkpoint, checkpoint_model_specification, model_specification
from ihc_unext2 import UNeXt2VirtualStainer
from cv_protocol import load_protocol
from train_unet import build_optimizer
import train_unet


class DinoFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_alignment_preserves_image_coordinates_with_symmetric_and_odd_padding(self):
        for height, width in ((384, 512), (65, 81)):
            with self.subTest(shape=(height, width)):
                pad_h, pad_w = (-height) % 14, (-width) % 14
                ys = (torch.arange((height + pad_h) // 14) + 0.5) * 14 - pad_h // 2
                xs = (torch.arange((width + pad_w) // 14) + 0.5) * 14 - pad_w // 2
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                source = torch.stack((xx, yy))[None]
                th, tw = height // 16, width // 16
                result = _align_dino_to_bottleneck(source, (height, width), (th, tw))
                expected_y = (torch.arange(th) + 0.5) * height / th
                expected_x = (torch.arange(tw) + 0.5) * width / tw
                ey, ex = torch.meshgrid(expected_y, expected_x, indexing="ij")
                torch.testing.assert_close(result, torch.stack((ex, ey))[None], atol=1e-4, rtol=1e-6)

    def test_zero_start_matches_base_and_gradients_reach_encoder_after_projection_update(self):
        model = UNeXt2DinoBottleneckFusionVirtualStainer(4, "spatial-thumbnail-film")
        source = torch.rand(1, 1, 64, 80)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        output = model(source, condition)
        with torch.no_grad():
            self.assertTrue(torch.equal(output, UNeXt2VirtualStainer.forward(model, source, condition)))
        output.mean().backward()
        projection = model.dinov2_fusion.projection.weight
        self.assertGreater(float(projection.grad.abs().sum()), 0.0)
        with torch.no_grad():
            projection.add_(projection.grad, alpha=-0.01)
        model.zero_grad(set_to_none=True)
        model(source, condition).mean().backward()
        grad = model.dinov2_fusion.encoder.patch_embed.proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)
        optimizer, groups = build_optimizer(model, "unext2-fcmae-tiny-dinov2-fusion", 3e-4, None, 1e-4)
        rates = {group["name"]: group["initial_learning_rate"] for group in groups}
        self.assertEqual(rates["encoder"], 3e-4)
        self.assertEqual(rates["decoder_head"], 3e-4)
        self.assertAlmostEqual(rates["dinov2_encoder"], 3e-5)
        ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for parameter in model.parameters()})

    def test_checkpoint_restore_requires_no_public_weight_file(self):
        with mock.patch("ihc_dino_fusion.initialize_dinov2_encoder", side_effect=AssertionError("Public weights forbidden")):
            model = UNeXt2DinoBottleneckFusionVirtualStainer().eval()
            checkpoint = {"model_spec": model_specification("unext2-fcmae-tiny-dinov2-fusion"),
                          "markers": ["CD68"], "model": model.state_dict()}
            restored = build_model_from_checkpoint(checkpoint, torch.device("cpu")).eval()
            source = torch.rand(1, 1, 64, 80)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(source), restored(source)))
            checkpoint["model_spec"]["fusion_encoder_layers"] = [2, 5, 8, 11]
            with self.assertRaisesRegex(ValueError, "DINOv2 fusion checkpoint"):
                checkpoint_model_specification(checkpoint)

    def test_protocol_and_cli_accept_fresh_fcmae_conditioned_training(self):
        protocol = load_protocol(ROOT / "configs/round114_cv2_dino_bottleneck_fusion.json")
        recipe = protocol["recipes"]["dinov2fusion"]
        self.assertEqual(recipe["batch_size"] * recipe["gradient_accumulation_steps"], 16)
        self.assertEqual(recipe["train_stages"], [1, 2])
        argv = ["train_unet.py", "dataset", "--model-arch", recipe["model_arch"],
                "--fcmae-encoder-init-checkpoint", "encoder.pt", "--condition-mode", recipe["condition_mode"],
                "--output-dir", "unused"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(train_unet, "prepare_output_dir"), \
             mock.patch.object(train_unet, "resolve_data_root", side_effect=RuntimeError("dataset boundary reached")):
            with self.assertRaisesRegex(RuntimeError, "dataset boundary reached"):
                train_unet.main()


if __name__ == "__main__":
    unittest.main()
