from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round110Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(
            PROJECT_ROOT / "configs" / "round110_cv2_unext2_feature_guidance.json"
        )

    def test_feature_guidance_is_fixed_and_fold_local(self) -> None:
        feature = self.protocol["recipes"]["featureguidance15"]["feature_guidance"]
        self.assertEqual(feature["weight"], 0.05)
        self.assertEqual(feature["stage"], 4)
        self.assertEqual(feature["crop_size"], 256)
        self.assertEqual(feature["batch_samples"], 4)
        self.assertEqual(set(feature["teachers"]), {"fold0", "fold2"})

    def test_runner_resolves_fold_teacher_and_keeps_batch16(self) -> None:
        recipe = self.protocol["recipes"]["featureguidance15"]
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            self.protocol["folds"][0]["val_rois"],
            "CD68",
            recipe,
            self.protocol["training"],
            1,
            None,
            "cuda:0",
            fold_name="fold0",
        )
        self.assertEqual(command[command.index("--batch-size") + 1], "16")
        self.assertEqual(
            command[command.index("--feature-guidance-weight") + 1], "0.05"
        )
        checkpoint = command[command.index("--feature-guidance-checkpoint") + 1]
        self.assertIn("round96_cv2_multimarker_fcmae_fold0", checkpoint)

    def test_gate_requires_both_folds(self) -> None:
        gate = self.protocol["distribution_gates"]["feature_guidance"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)


if __name__ == "__main__":
    unittest.main()
