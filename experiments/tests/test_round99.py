from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round99ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round99_cv2_dapi_detail_residual.json"
        self.protocol = load_protocol(self.path)

    def test_fixed_detail_recipe(self) -> None:
        recipe = self.protocol["recipes"]["detailresidual15"]
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny-dapi-detail-residual")
        self.assertEqual(recipe["training_markers"], ["CD68", "CD45RO", "HLA-DR", "Vimentin"])
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(recipe["inference_context_halo"], 128)

    def test_runner_emits_new_architecture_and_fcmae_init(self) -> None:
        recipe = self.protocol["recipes"]["detailresidual15"]
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
        self.assertEqual(
            command[command.index("--model-arch") + 1],
            "unext2-fcmae-tiny-dapi-detail-residual",
        )
        self.assertNotIn("--d4-consistency-weight", command)

    def test_gate_and_launcher_are_fixed(self) -> None:
        gate = self.protocol["distribution_gates"]["dapi_detail_residual"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round99_dapi_detail_residual.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("detailresidual15", launcher)
        self.assertIn("run_cv_training.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)
        self.assertNotIn("ROUND98_", launcher)


if __name__ == "__main__":
    unittest.main()
