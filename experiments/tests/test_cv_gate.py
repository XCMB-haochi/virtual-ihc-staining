from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol
from evaluate_cv_gate import evaluate_gates


PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round38_cv5_aligned_thumbnail_fusion.json"
)


def metrics(score: float) -> dict[str, float]:
    return {
        "competition_score_mean_image_psnr": score,
        "competition_score_global_mse_psnr": score,
    }


def synthetic_report(protocol: dict[str, object]) -> dict[str, object]:
    scores = {
        "round35_shrink30": 70.0,
        "context64control": 70.0,
        "aligneddirect": 70.2,
        "round35_aligned50": 70.13,
    }

    def candidate_metrics() -> dict[str, object]:
        return {
            candidate: {
                "float": metrics(score),
                "jpeg_q95": metrics(score),
            }
            for candidate, score in scores.items()
        }

    test_like_rois = ["ROI004", "ROI006", "ROI007"]
    return {
        "protocol_name": protocol["name"],
        "test_like_train_rois": test_like_rois,
        "overall": candidate_metrics(),
        "test_like": candidate_metrics(),
        "fold_metrics": {
            fold["name"]: candidate_metrics() for fold in protocol["folds"]
        },
        "roi_metrics": {
            candidate: {
                "float": {roi: metrics(score) for roi in test_like_rois},
                "jpeg_q95": {roi: metrics(score) for roi in test_like_rois},
            }
            for candidate, score in scores.items()
        },
    }


class CVGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol(PROTOCOL_PATH)
        self.report = synthetic_report(self.protocol)

    def test_primary_gate_promotes_the_fixed_blend(self) -> None:
        result = evaluate_gates(self.protocol, self.report)
        self.assertTrue(result["gates"]["mechanism"]["passed"])
        self.assertTrue(result["gates"]["primary"]["passed"])
        self.assertEqual(
            result["promotion"],
            {
                "allowed": True,
                "gate": "primary",
                "candidate": "round35_aligned50",
            },
        )

    def test_direct_candidate_is_the_only_preregistered_fallback(self) -> None:
        report = copy.deepcopy(self.report)
        report["overall"]["round35_aligned50"]["jpeg_q95"] = metrics(70.11)
        result = evaluate_gates(self.protocol, report)
        self.assertFalse(result["gates"]["primary"]["passed"])
        self.assertTrue(result["gates"]["direct_diagnostic"]["passed"])
        self.assertEqual(result["promotion"]["gate"], "direct_diagnostic")
        self.assertEqual(result["promotion"]["candidate"], "aligneddirect")

    def test_failed_mechanism_dependency_blocks_the_primary_gate(self) -> None:
        report = copy.deepcopy(self.report)
        for scope in ("overall", "test_like"):
            report[scope]["context64control"] = {
                "float": metrics(70.15),
                "jpeg_q95": metrics(70.15),
            }
        for fold_metrics in report["fold_metrics"].values():
            fold_metrics["context64control"] = {
                "float": metrics(70.15),
                "jpeg_q95": metrics(70.15),
            }
        for roi in report["test_like_train_rois"]:
            report["roi_metrics"]["context64control"]["jpeg_q95"][roi] = metrics(
                70.15
            )

        result = evaluate_gates(self.protocol, report)
        self.assertFalse(result["gates"]["mechanism"]["passed"])
        self.assertFalse(result["gates"]["primary"]["checks"]["dependency"])
        self.assertFalse(result["gates"]["primary"]["passed"])
        self.assertEqual(result["promotion"]["gate"], "direct_diagnostic")

    def test_protocol_mismatch_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["protocol_name"] = "other_protocol"
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            evaluate_gates(self.protocol, report)


if __name__ == "__main__":
    unittest.main()
