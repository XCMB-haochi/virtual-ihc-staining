from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol, run_directory, sourced_run_directory


class Round107ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round107_cv2_unext2_checkpoint_swa.json"
        self.protocol = load_protocol(self.path)

    def test_soup_rule_is_fixed(self) -> None:
        self.assertEqual(
            self.protocol["checkpoint_swa"],
            {
                "source_protocol": "round55_cv2_unext2_official_auxiliary",
                "source_recipe": "officialauxiliary15",
                "stage": 2,
                "epoch_rule": "three_distinct_epochs_centered_on_each_fold_global_mse_best_epoch",
                "weights": [1, 2, 1],
                "buffer_policy": "error",
            },
        )

    def test_baseline_and_soup_share_inference_contract(self) -> None:
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)
        self.assertEqual(self.protocol["inference"]["tta"], "d4")
        self.assertEqual(
            self.protocol["candidate_output_source_offsets"],
            {"baseline_d4_context128": [0, -1], "swa_d4_context128": [0, -1]},
        )
        for name in ("baselineauxiliary15", "swaauxiliary15"):
            self.assertEqual(
                self.protocol["recipes"][name]["inference_context_halo"], 128
            )

    def test_source_paths_are_separate(self) -> None:
        baseline = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "baselineauxiliary15"
        )
        soup = sourced_run_directory(
            PROJECT_ROOT / "runs", self.protocol, "fold0", "swaauxiliary15"
        )
        self.assertIn("round55_cv2_unext2_official_auxiliary", str(baseline))
        self.assertEqual(
            soup,
            run_directory(
                PROJECT_ROOT / "runs",
                self.protocol["name"],
                "fold0",
                "swaauxiliary15",
                2,
            ),
        )
        self.assertNotEqual(baseline, soup)

    def test_gate_and_launcher_are_bounded(self) -> None:
        gate = self.protocol["distribution_gates"]["checkpoint_swa"]
        self.assertEqual(gate["minimum_q95_gain"], 0.01)
        self.assertTrue(gate["require_positive_float"])
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round107_checkpoint_swa.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("build_checkpoint_swa.py", launcher)
        self.assertIn("build_soup", launcher)
        self.assertIn("evaluate_cv_ensemble.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)


if __name__ == "__main__":
    unittest.main()
