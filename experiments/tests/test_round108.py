from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round108ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(
            PROJECT_ROOT / "configs" / "round108_cv2_unext2_exact_score.json"
        )

    def test_exact_score_recipe_is_single_marker_and_fixed(self) -> None:
        recipe = self.protocol["recipes"]["exactscore"]
        self.assertEqual(recipe["loss_profile"], "exact-score")
        self.assertNotIn("training_markers", recipe)
        self.assertNotIn("marker_loss_weights", recipe)
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(recipe["inference_context_halo"], 128)

    def test_runner_forwards_loss_profile(self) -> None:
        recipe = self.protocol["recipes"]["exactscore"]
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
        )
        index = command.index("--loss-profile")
        self.assertEqual(command[index + 1], "exact-score")
        self.assertNotIn("--marker-loss-weights", command)

    def test_gate_is_distribution_bounded(self) -> None:
        gate = self.protocol["distribution_gates"]["exact_score"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)


if __name__ == "__main__":
    unittest.main()
