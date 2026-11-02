from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round111Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(
            PROJECT_ROOT / "configs" / "round111_cv2_unext2_pyramid_loss.json"
        )

    def test_pyramid_loss_is_single_fixed_candidate(self) -> None:
        recipe = self.protocol["recipes"]["pyramidauxiliary15"]
        self.assertEqual(recipe["pyramid_weight"], 0.1)
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(recipe["train_stages"], [1, 2])

    def test_runner_forwards_pyramid_weight(self) -> None:
        recipe = self.protocol["recipes"]["pyramidauxiliary15"]
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
        self.assertEqual(command[command.index("--pyramid-weight") + 1], "0.1")

    def test_gate_is_matched_two_fold_screen(self) -> None:
        gate = self.protocol["distribution_gates"]["pyramid_loss"]
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)


if __name__ == "__main__":
    unittest.main()
