from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cv_protocol import load_protocol


class Round90ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = PROJECT_ROOT / "configs" / "round90_cv5_auxiliary_context128.json"
        self.protocol = load_protocol(self.path)

    def test_fixed_identity_and_context(self) -> None:
        self.assertEqual(self.protocol["name"], "round90_cv5_auxiliary_context128")
        self.assertEqual(self.protocol["inference"]["context_halo"], 128)
        self.assertEqual(self.protocol["inference"]["tta"], "d4")
        self.assertFalse(self.protocol["inference"]["test_like_diagnostic"])

    def test_single_candidate_preserves_round67_weights(self) -> None:
        candidate = "tri_auxhalf_residual_half_context128"
        self.assertEqual(self.protocol["primary_candidate"], candidate)
        self.assertEqual(
            self.protocol["candidates"][candidate],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0875,
                "officialauxiliary15": 0.175,
                "alldapiofficialfcmae630": 0.0875,
            },
        )
        self.assertEqual(
            self.protocol["candidate_output_source_offsets"][candidate],
            [0, -1],
        )

    def test_all_members_are_inference_only_context128(self) -> None:
        self.assertEqual(len(self.protocol["recipes"]), 6)
        self.assertTrue(
            all(
                recipe["train_stages"] == []
                and recipe["inference_context_halo"] == 128
                for recipe in self.protocol["recipes"].values()
            )
        )

    def test_source_binding_and_gate_are_fixed(self) -> None:
        self.assertEqual(
            self.protocol["context128_source"],
            {
                "protocol_name": "round67_cv5_auxiliary_all_dapi_composition",
                "candidate": "tri_auxhalf_residual_half",
                "oof_sha256": "8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2",
                "gate_sha256": "cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc",
            },
        )
        self.assertEqual(
            self.protocol["context128_gate"],
            {
                "minimum_q95_gain": 0.02,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 18,
                "minimum_roi_q95_gain": -0.05,
            },
        )

    def test_runner_is_no_training_and_binds_source(self) -> None:
        launcher = (
            PROJECT_ROOT
            / "tools"
            / "run_round90_auxiliary_context128_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_cv_training.py", launcher)
        self.assertIn("evaluate_cv_ensemble.py", launcher)
        self.assertIn("evaluate_round90_context128_composition_gate.py", launcher)
        self.assertIn(
            "8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2",
            launcher,
        )
        self.assertIn(
            "cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
