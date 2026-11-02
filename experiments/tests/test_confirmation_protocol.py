from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cv_protocol import load_protocol, validate_protocol
from evaluate_distribution_gate import evaluate_gates


class ConfirmationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "configs/round115_fold1_dino_residual_confirmation.json")

    def test_confirmation_keeps_training_recipe_and_uses_an_extra_fold(self):
        original = load_protocol(ROOT / "configs/round113_cv2_dino_residual.json")
        for field in ("recipes", "training", "inference", "fcmae_pretraining",
                      "candidates", "candidate_output_source_offsets"):
            self.assertEqual(self.protocol[field], original[field], field)
        self.assertFalse(set(self.protocol["expected_rois"]) & set(original["expected_rois"]))

    def test_one_fold_requires_explicit_confirmation_scope(self):
        for change in ({"confirmation_only": False}, {"confirmation_only": "true"},
                       {"roi_scope": "complete"}):
            with self.subTest(change=change):
                protocol = {**self.protocol, **change}
                with self.assertRaises(ValueError):
                    validate_protocol(protocol)

    def test_positive_confirmation_can_never_authorize_submission(self):
        protocol = self.protocol
        roi_names = sorted(protocol["expected_rois"])
        variants = {candidate: {variant: {"competition_score_official": score}
                    for variant in ("float", "jpeg_q95")}
                    for candidate, score in ((protocol["baseline_candidate"], 75.0),
                                              (protocol["primary_candidate"], 75.1))}
        report = {"protocol_name": protocol["name"], "marker": "CD68",
                  "validation_rois": roi_names, "overall": variants,
                  "fold_metrics": {"fold1": variants},
                  "roi_metrics": {candidate: {"jpeg_q95": {roi: values["jpeg_q95"] for roi in roi_names}}
                                  for candidate, values in variants.items()}}
        passed = evaluate_gates(protocol, report)
        self.assertTrue(passed["confirmation"]["passed"])
        self.assertEqual(passed["promotion"], {"allowed": False, "gate": None, "candidate": None})
        failed_report = copy.deepcopy(report)
        failed_report["overall"][protocol["primary_candidate"]]["jpeg_q95"]["competition_score_official"] = 74.9
        failed = evaluate_gates(protocol, failed_report)
        self.assertFalse(failed["confirmation"]["passed"])
        self.assertFalse(failed["promotion"]["allowed"])


if __name__ == "__main__":
    unittest.main()
