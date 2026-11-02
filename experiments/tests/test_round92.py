from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from run_cv_training import build_train_command


class Round92ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            PROJECT_ROOT
            / "configs"
            / "round92_cv2_unext2_d4_consistency_context128.json"
        )
        self.protocol = load_protocol(self.path)

    def test_fixed_d4_recipe(self) -> None:
        recipe = self.protocol["recipes"]["d4auxiliary15"]
        self.assertEqual(recipe["d4_consistency_weight"], 0.1)
        self.assertEqual(recipe["batch_size"], 8)
        self.assertEqual(recipe["gradient_accumulation_steps"], 2)
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])

    def test_runner_emits_d4_flag(self) -> None:
        recipe = self.protocol["recipes"]["d4auxiliary15"]
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
            command[command.index("--d4-consistency-weight") + 1],
            "0.1",
        )

    def test_fixed_candidates_and_dependencies(self) -> None:
        self.assertEqual(
            self.protocol["primary_candidate"],
            "round90_d4_replace_context128",
        )
        self.assertEqual(
            self.protocol["gates"]["d4_replacement"]["requires"],
            "d4_mechanism",
        )
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)
        self.assertEqual(self.protocol["inference"]["tta"], "d4")

    def test_launcher_is_fixed_and_trains_only_candidate(self) -> None:
        launcher = (
            PROJECT_ROOT
            / "tools"
            / "run_round92_unext2_d4_consistency_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_cv_training.py", launcher)
        self.assertIn("d4auxiliary15", launcher)
        self.assertIn("--d4-consistency-weight", launcher)
        self.assertIn("evaluate_cv_gate.py", launcher)
        self.assertNotIn("ROUND90_GPU", launcher)


if __name__ == "__main__":
    unittest.main()
