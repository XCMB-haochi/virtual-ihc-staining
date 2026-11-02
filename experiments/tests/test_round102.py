from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol, sourced_run_directory


class Round102ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round102_cv2_stride_phase_tta.json"
        self.protocol = load_protocol(self.path)

    def test_phase_geometry_and_candidates_are_fixed(self) -> None:
        self.assertEqual(
            self.protocol["phase_tta"],
            {
                "model_context_halo": 128,
                "outer_margin": 1,
                "phase_offsets": [[0, 0], [0, 1], [1, 0], [1, 1]],
                "alignment": "crop-to-fixed-center-before-average",
            },
        )
        self.assertEqual(self.protocol["inference"]["context_halo"], 129)
        self.assertEqual(
            set(self.protocol["candidates"]),
            {"standard_d4_context128", "stride_phase4_d4_context128"},
        )

    def test_both_variants_reuse_the_same_round55_checkpoint(self) -> None:
        for fold in ("fold0", "fold2"):
            standard = sourced_run_directory(
                Path("runs"), self.protocol, fold, "standardauxiliary15"
            )
            phase = sourced_run_directory(
                Path("runs"), self.protocol, fold, "phaseauxiliary15"
            )
            self.assertEqual(standard, phase)
            self.assertIn("round55_cv2_unext2_official_auxiliary", str(standard))

    def test_gate_and_launcher_are_bounded(self) -> None:
        gate = self.protocol["distribution_gates"]["stride_phase_tta"]
        self.assertEqual(gate["minimum_q95_gain"], 0.02)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        self.assertEqual(gate["minimum_roi_q95_gain"], -0.05)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round102_stride_phase_tta.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--max-images-per-fold 1", launcher)
        self.assertIn("evaluate_stride_phase_tta.py", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)


if __name__ == "__main__":
    unittest.main()
