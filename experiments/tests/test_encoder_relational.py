from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from ihc_feature_guidance import FrozenFCMAERelationalLoss, relational_distance_loss
from ihc_unext2 import UNeXt2VirtualStainer
from ihc_torch import CompetitionLoss
from train_unet import training_batch_losses
from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 1)

    def forward(self, x):
        x = self.conv(x)
        return [x] * 4, None


class EncoderRelationalTests(unittest.TestCase):
    def test_known_geometry_and_teacher_is_constant(self):
        student = torch.tensor([[0.0], [1.0], [3.0]], requires_grad=True)
        teacher = torch.tensor([[0.0], [2.0], [3.0]], requires_grad=True)
        loss = relational_distance_loss(student, teacher)
        self.assertAlmostEqual(float(loss.detach()), 1 / 12, places=6)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)

    def test_translation_rotation_scale_invariance_and_pairing(self):
        torch.manual_seed(12)
        student = torch.randn(8, 5)
        q, _ = torch.linalg.qr(torch.randn(5, 5))
        teacher = 3 * student @ q + 7
        self.assertLess(float(relational_distance_loss(student, teacher)), 1e-10)
        self.assertGreater(float(relational_distance_loss(student, teacher.roll(1, 0))), 1e-4)

    def test_degenerate_batch_is_finite(self):
        for n in (1, 2, 4):
            student = torch.ones(n, 4, requires_grad=True)
            loss = relational_distance_loss(student, torch.zeros(n, 3))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(student.grad).all())

    def test_real_encoder_receives_guidance_without_decoder_or_teacher_gradients(self):
        torch.manual_seed(21)
        model = UNeXt2VirtualStainer(out_channels=1)
        guidance = FrozenFCMAERelationalLoss(Teacher(), stage=4, crop_size=32, batch_samples=3)
        source = torch.rand(3, 1, 64, 64)
        target = torch.rand(3, 1, 32, 32, requires_grad=True)
        criterion = CompetitionLoss()
        plain = training_batch_losses(model, source, target, None, criterion, 0)
        guided = training_batch_losses(model, source, target, None, criterion, 0,
                                       feature_guidance=guidance, feature_guidance_weight=0.05)
        torch.testing.assert_close(plain[1], guided[1], rtol=0, atol=0)
        self.assertGreater(float(guided[3].detach()), 0)
        guided[3].backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in guidance.teacher.parameters()))
        self.assertIsNone(target.grad)
        guidance.train()
        self.assertFalse(guidance.teacher.training)
        restored = UNeXt2VirtualStainer(out_channels=1)
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(source), restored(source), rtol=0, atol=0)

    def test_registered_commands_keep_full_batch_and_fold_teachers(self):
        protocol = load_protocol(ROOT / "configs/round121_cv2_encoder_relational.json")
        recipe = protocol["recipes"]["encoderrelational15"]
        for fold in protocol["folds"]:
            for stage in (1, 2):
                cmd = build_train_command("python", ROOT / "dataset", ROOT / "runs/test", fold["val_rois"],
                    protocol["marker"], recipe, protocol["training"], stage,
                    ROOT / "runs/stage1.pt" if stage == 2 else None, "cpu", fold_name=fold["name"])
                self.assertEqual(cmd[cmd.index("--feature-guidance-mode") + 1], "encoder-relational")
                self.assertEqual(cmd[cmd.index("--feature-guidance-batch-samples") + 1], "16")
                self.assertEqual(cmd[cmd.index("--batch-size") + 1], "16")
                self.assertIn(fold["name"], cmd[cmd.index("--feature-guidance-checkpoint") + 1])
                self.assertNotIn("--primary-gradient-projection", cmd)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
