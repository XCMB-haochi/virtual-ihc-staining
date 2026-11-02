from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol, run_directory, sourced_run_directory


class Round103ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round103_cv2_unext2_auxiliary_continuation.json"
        self.protocol = load_protocol(self.path)

    def test_continuation_is_one_fixed_stage_from_round55(self) -> None:
        recipe = self.protocol["recipes"]["auxiliary15continue"]
        self.assertEqual(recipe["train_stages"], [2])
        self.assertEqual(recipe["epochs_per_stage"], 14)
        self.assertEqual(recipe["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(
            recipe["init_source"],
            {
                "protocol_name": "round55_cv2_unext2_official_auxiliary",
                "recipe_name": "officialauxiliary15",
                "stage": 2,
            },
        )

    def test_baseline_and_candidate_have_identical_inference_contract(self) -> None:
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)
        self.assertEqual(self.protocol["inference"]["tta"], "d4")
        self.assertEqual(
            self.protocol["candidate_output_source_offsets"],
            {
                "baseline_d4_context128": [0, -1],
                "continuation_cycle_d4_context128": [0, -1],
            },
        )
        for name in ("baselineauxiliary15", "auxiliary15continue"):
            self.assertEqual(
                self.protocol["recipes"][name]["inference_context_halo"], 128
            )

    def test_candidate_source_is_round103_and_baseline_is_round55(self) -> None:
        baseline = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "baselineauxiliary15"
        )
        candidate = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "auxiliary15continue"
        )
        self.assertIn("round55_cv2_unext2_official_auxiliary", str(baseline))
        self.assertEqual(
            candidate,
            run_directory(
                PROJECT_ROOT / "runs",
                self.protocol["name"],
                "fold0",
                "auxiliary15continue",
                2,
            ),
        )

    def test_gate_is_a_bounded_distribution_check(self) -> None:
        gate = self.protocol["distribution_gates"]["continuation_cycle"]
        self.assertEqual(gate["minimum_q95_gain"], 0.03)
        self.assertTrue(gate["require_positive_float"])
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round103_auxiliary_continuation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--init-checkpoint", launcher)
        self.assertIn("run_cv_training.py", launcher)
        self.assertIn("evaluate_cv_ensemble.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)


if __name__ == "__main__":
    unittest.main()
