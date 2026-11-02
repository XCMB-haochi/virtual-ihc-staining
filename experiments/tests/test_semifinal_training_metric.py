import sys
import unittest
import json
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_unet import validate
from run_round204_cd45_cv import common_clock, training_command


class ZeroModel(torch.nn.Module):
    def forward(self, source, condition=None):
        return torch.zeros_like(source)


class SelectionMetricTests(unittest.TestCase):
    def test_common_clock_uses_weighted_curve_not_median_fold_optimum(self):
        folds = [{"name": f"fold{i}"} for i in range(5)]
        texts = []
        for i, count in enumerate([420, 420, 420, 560, 560]):
            scores = ([1., .9] if i < 3 else [0., .2]) + [0.] * 12
            rows = [{"epoch": e + 1, "validation": {
                "count": count, "ssim_mode": "gaussian11", "markers": {"CD45RO": {
                    "competition_score_mean_image_psnr": score}}}} for e, score in enumerate(scores)]
            texts.append('\n'.join(json.dumps(r) for r in rows))
        with mock.patch.object(Path, 'read_text', side_effect=texts):
            result = common_clock(folds, 1)
        self.assertEqual(result['selected_epoch'], 2)

    def test_stage2_uses_own_fold_at_common_stage1_epoch(self):
        command, _ = training_command({'name': 'fold3', 'val_rois': ['ROI044']}, 2, 11)
        self.assertEqual(command[command.index('--init-checkpoint') + 1], 'runs/round204_fold3_stage1/epochs/epoch_011.pt')
        self.assertEqual(command[command.index('--selection-ssim-mode') + 1], 'gaussian11')
        self.assertEqual(command[command.index('--ssim-mode') + 1], 'uniform7')

    def test_validator_uses_explicit_profile_and_preserves_old_default(self):
        target = torch.from_numpy(np.random.default_rng(2026).integers(0, 100, (1, 1, 32, 32)).astype(np.float32)) / 255
        loader = [(torch.zeros_like(target), target, ['ROI001_00_00.jpg'])]
        for mode in ('uniform7', 'gaussian11'):
            kwargs = {} if mode == 'uniform7' else {'selection_ssim_mode': mode}
            result = validate(ZeroModel(), loader, torch.device('cpu'), ('CD45RO',), 95, **kwargs)
            opts = dict(win_size=7) if mode == 'uniform7' else dict(win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            truth = target.numpy()[0, 0] * 255
            expected = structural_similarity(truth, np.zeros_like(truth), data_range=255, **opts)
            self.assertEqual(result['ssim_mode'], mode)
            self.assertAlmostEqual(result['markers']['CD45RO']['ssim'], expected)


if __name__ == '__main__':
    unittest.main()
