import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_round204_cd45_cv import training_command
from run_round206_gaussian import command_for


class GaussianScreenTests(unittest.TestCase):
    def test_only_loss_mode_and_artifact_paths_change(self):
        fold = {'name': 'fold3', 'val_rois': ['ROI044', 'ROI067', 'ROI074', 'ROI078']}
        for stage in (1, 2):
            baseline, _ = training_command(fold, stage, 14)
            candidate, _ = command_for(fold, stage)
            expected = [part.replace('round204', 'round206') for part in baseline]
            expected[expected.index('--ssim-mode') + 1] = 'gaussian11'
            self.assertEqual(candidate, expected)


if __name__ == '__main__':
    unittest.main()
