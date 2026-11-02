import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from ihc_torch import CompetitionLoss, PairedPatchDataset
from train_unet import training_batch_losses


class IdentityModel(torch.nn.Module):
    def forward(self, source, condition=None):
        return source


class DistillationTests(unittest.TestCase):
    def test_dataset_returns_teacher_target(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            name = "ROI000_00_00.jpg"
            for folder, value in ((root / "train/DAPI", 20), (root / "train/CD68", 80), (root / "teacher", 120)):
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((256, 256), value, np.uint8)).save(folder / name)
            source, target, teacher, actual_name = PairedPatchDataset(
                root, [name], markers=("CD68",), distillation_target_dir=root / "teacher"
            )[0]
            self.assertEqual(actual_name, name)
            self.assertEqual(tuple(source.shape), (1, 256, 256))
            self.assertAlmostEqual(float(target.mean()), 80 / 255)
            self.assertAlmostEqual(float(teacher.mean()), 120 / 255)

    def test_loss_is_fixed_supervised_teacher_mixture(self):
        model = IdentityModel()
        source = torch.full((2, 1, 32, 32), 0.4)
        target = torch.full((2, 1, 32, 32), 0.6)
        teacher = torch.full((2, 1, 16, 16), 0.5)
        criterion = CompetitionLoss(ssim_weight=0.7, l1_weight=0.2, mse_weight=0.1)
        expected = 0.8 * criterion(source, target) + 0.2 * criterion(
            source[..., 8:24, 8:24], teacher
        )
        total, supervised, consistency, feature = training_batch_losses(
            model,
            source,
            target,
            None,
            criterion,
            0,
            distillation_target=teacher,
            distillation_criterion=criterion,
            distillation_weight=0.2,
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertTrue(torch.allclose(supervised, expected))
        self.assertEqual(float(consistency), 0.0)
        self.assertEqual(float(feature), 0.0)


if __name__ == "__main__":
    unittest.main()
