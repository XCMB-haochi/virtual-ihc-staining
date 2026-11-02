from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round100ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round100_cv2_shallow_dapi_skip.json"
        self.protocol = load_protocol(self.path)

    def test_fixed_shallow_skip_recipe(self) -> None:
        recipe = self.protocol["recipes"]["shallowskip15"]
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny-shallow-dapi-skip")
        self.assertEqual(recipe["training_markers"], ["CD68", "CD45RO", "HLA-DR", "Vimentin"])
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(recipe["train_stages"], [1, 2])

    def test_runner_command_and_gate_are_fixed(self) -> None:
        recipe = self.protocol["recipes"]["shallowskip15"]
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
        self.assertEqual(command[command.index("--model-arch") + 1], recipe["model_arch"])
        gate = self.protocol["distribution_gates"]["shallow_dapi_skip"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)

    def test_launcher_has_no_round99_dependency(self) -> None:
        launcher = (
            PROJECT_ROOT / "tools" / "run_round100_shallow_dapi_skip.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_cv_training.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)
        self.assertIn("unext2-fcmae-tiny-shallow-dapi-skip", launcher)
        self.assertNotIn("ROUND99_", launcher)


if __name__ == "__main__":
    unittest.main()
