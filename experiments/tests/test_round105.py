from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol, run_directory, sourced_run_directory


class Round105ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round105_cv2_unext2_activity_hard_mix.json"
        self.protocol = load_protocol(self.path)

    def test_activity_recipe_is_fixed_and_label_only(self) -> None:
        recipe = self.protocol["recipes"]["activityauxiliary15"]
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(
            recipe["activity_sampler"],
            {
                "mode": "hard-mix",
                "start_ratio": 0.75,
                "hard_fraction": 0.30,
                "num_bins": 4,
            },
        )
        self.assertNotIn("test", str(recipe["activity_sampler"]).casefold())

    def test_baseline_and_candidate_share_inference_contract(self) -> None:
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)
        self.assertEqual(self.protocol["inference"]["tta"], "d4")
        self.assertEqual(
            self.protocol["candidate_output_source_offsets"],
            {"baseline_d4_context128": [0, -1], "activity_d4_context128": [0, -1]},
        )
        self.assertEqual(
            self.protocol["recipes"]["baselineauxiliary15"]["inference_context_halo"],
            128,
        )
        self.assertEqual(
            self.protocol["recipes"]["activityauxiliary15"]["inference_context_halo"],
            128,
        )

    def test_run_sources_keep_baseline_immutable(self) -> None:
        baseline = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "baselineauxiliary15"
        )
        candidate = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "activityauxiliary15"
        )
        self.assertIn("round55_cv2_unext2_official_auxiliary", str(baseline))
        self.assertEqual(
            candidate,
            run_directory(
                PROJECT_ROOT / "runs",
                self.protocol["name"],
                "fold0",
                "activityauxiliary15",
                2,
            ),
        )

    def test_gate_and_launcher_are_bounded(self) -> None:
        gate = self.protocol["distribution_gates"]["activity_hard_mix"]
        self.assertEqual(gate["minimum_q95_gain"], 0.03)
        self.assertTrue(gate["require_positive_float"])
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round105_activity_hard_mix.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--activity-sampler hard-mix", launcher)
        self.assertIn("--activity-sampler-start-ratio", launcher)
        self.assertIn("run_cv_training.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)


if __name__ == "__main__":
    unittest.main()
