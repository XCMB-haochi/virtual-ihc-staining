from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from ihc_dinov2 import DinoV2SmallUNet, DINOV2_REPOSITORY, DINOV2_WEIGHT_PATH, initialize_dinov2_encoder
from ihc_spatial import SPATIAL_THUMBNAIL_CONDITION_DIMENSION
from ihc_torch import checkpoint_model_specification, model_specification
from cv_protocol import load_protocol, validate_protocol
from run_cv_training import build_train_command
from train_unet import build_optimizer
import train_unet


class DinoProtocolTests(unittest.TestCase):
    def test_cli_accepts_dino_conditioning_and_public_initialization(self):
        argv = ["train_unet.py", "dataset", "--model-arch", "dinov2-small-unet",
                "--encoder-init", "dinov2", "--condition-mode", "spatial-thumbnail-film",
                "--output-dir", "unused", "--encoder-learning-rate", "3e-5"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(train_unet, "prepare_output_dir"), \
             mock.patch.object(train_unet, "resolve_data_root", side_effect=RuntimeError("dataset boundary reached")):
            with self.assertRaisesRegex(RuntimeError, "dataset boundary reached"):
                train_unet.main()

    def test_initialization_is_only_applied_to_fresh_stage1(self):
        p = load_protocol(ROOT / "configs/round112_cv2_dinov2.json")
        for stage in (1, 2):
            command = build_train_command("python", Path("dataset"), Path("runs/example"),
                p["folds"][0]["val_rois"], "CD68", p["recipes"]["dino"], p["training"],
                stage, Path("runs/init.pt") if stage == 2 else None, "cpu", fold_name="fold0")
            self.assertEqual("--encoder-init" in command, stage == 1)
            self.assertEqual(command[command.index("--encoder-learning-rate")+1],
                             str(p["training"][f"stage{stage}_learning_rate"] * 0.1))
            self.assertNotIn("--fcmae-encoder-init-checkpoint", command)
        invalid = copy.deepcopy(p)
        invalid["recipes"]["dino"]["model_arch"] = "unext2-fcmae-tiny"
        with self.assertRaisesRegex(ValueError, "DINOv2 initialization"):
            validate_protocol(invalid)

    def test_checkpoint_spec_rejects_changed_encoder_revision(self):
        spec = model_specification("dinov2-small-unet")
        self.assertEqual(checkpoint_model_specification({"model_spec": spec}), spec)
        spec["encoder_revision"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "DINOv2 checkpoint"):
            checkpoint_model_specification({"model_spec": spec})


@unittest.skipUnless(DINOV2_REPOSITORY.is_dir(), "Run tools/prepare_dinov2.py for encoder integration tests")
class DinoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_encoder_shapes_gradients_and_offline_reload(self):
        model = DinoV2SmallUNet(4, "spatial-thumbnail-film")
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        x = torch.rand(1, 1, 64, 80)
        output = model(x, condition)
        self.assertEqual(output.shape, (1, 4, 64, 80))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        for parameter in (model.encoder.patch_embed.proj.weight, model.head.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer, groups = build_optimizer(model, "dinov2-small-unet", 3e-4, 3e-5, 1e-4)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            with mock.patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Network forbidden")):
                restored = DinoV2SmallUNet(4, "spatial-thumbnail-film")
                restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(output, restored(x, condition)))

    @unittest.skipUnless(DINOV2_WEIGHT_PATH.is_file(), "Public weights not downloaded")
    def test_public_weights_strict_load_without_changing_decoder(self):
        model = DinoV2SmallUNet()
        head = model.head.weight.detach().clone()
        report = initialize_dinov2_encoder(model, DINOV2_WEIGHT_PATH)
        self.assertGreater(report["loaded_tensors"], 100)
        self.assertTrue(torch.equal(head, model.head.weight))
        state = torch.load(DINOV2_WEIGHT_PATH, weights_only=True, map_location="cpu")
        self.assertTrue(torch.equal(state["patch_embed.proj.weight"], model.encoder.patch_embed.proj.weight))


if __name__ == "__main__":
    unittest.main()
