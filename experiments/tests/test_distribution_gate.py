from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_distribution_gate import evaluate_gates, official_score


def metric(score: float, *, legacy: bool = False) -> dict[str, float]:
    key = (
        "competition_score_mean_image_psnr" if legacy else "competition_score_official"
    )
    return {key: score}


class DistributionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "name": "round42",
            "marker": "CD68",
            "expected_rois": ["ROI000", "ROI001", "ROI002"],
            "folds": [
                {"name": "fold0", "val_rois": ["ROI000"]},
                {"name": "fold1", "val_rois": ["ROI001", "ROI002"]},
            ],
            "inference": {"jpeg_quality": 95},
            "distribution_gates": {
                "mechanism": {
                    "candidate": "direct",
                    "baseline": "control",
                    "minimum_q95_gain": 0.05,
                    "require_positive_float": True,
                    "minimum_positive_folds": 2,
                    "minimum_positive_rois": 2,
                    "minimum_roi_q95_gain": -0.05,
                },
                "primary": {
                    "candidate": "primary",
                    "baseline": "baseline",
                    "minimum_q95_gain": 0.08,
                    "require_positive_float": True,
                    "minimum_positive_folds": 2,
                    "minimum_positive_rois": 2,
                    "minimum_roi_q95_gain": -0.05,
                    "requires": "mechanism",
                },
            },
            "distribution_promotion_order": ["primary"],
        }
        self.report = self.build_report(
            {
                "control": 70.0,
                "direct": 70.1,
                "baseline": 70.0,
                "primary": 70.12,
            }
        )

    def build_report(self, scores: dict[str, float]) -> dict[str, object]:
        return {
            "protocol_name": "round42",
            "marker": "CD68",
            "validation_rois": ["ROI000", "ROI001", "ROI002"],
            "overall": {
                candidate: {
                    "float": metric(score + 0.01),
                    "jpeg_q95": metric(score),
                }
                for candidate, score in scores.items()
            },
            "fold_metrics": {
                fold: {
                    candidate: {"jpeg_q95": metric(score)}
                    for candidate, score in scores.items()
                }
                for fold in ("fold0", "fold1")
            },
            "roi_metrics": {
                candidate: {
                    "jpeg_q95": {
                        roi: metric(score) for roi in ("ROI000", "ROI001", "ROI002")
                    }
                }
                for candidate, score in scores.items()
            },
        }

    def test_score_accepts_current_and_legacy_metric_keys(self) -> None:
        self.assertEqual(official_score(metric(71.0), "current"), 71.0)
        self.assertEqual(official_score(metric(72.0, legacy=True), "legacy"), 72.0)
        self.assertEqual(
            official_score(
                {
                    "competition_score_official": float("nan"),
                    "competition_score_mean_image_psnr": 73.0,
                },
                "fallback",
            ),
            73.0,
        )

    def test_all_distribution_gates_pass_and_promote_primary(self) -> None:
        result = evaluate_gates(self.protocol, self.report)
        self.assertTrue(result["gates"]["mechanism"]["passed"])
        self.assertTrue(result["gates"]["primary"]["passed"])
        self.assertEqual(result["promotion"]["candidate"], "primary")
        self.assertEqual(
            result["gates"]["primary"]["positive_rois"],
            ["ROI000", "ROI001", "ROI002"],
        )

    def test_failed_dependency_blocks_primary_promotion(self) -> None:
        report = copy.deepcopy(self.report)
        report["overall"]["direct"]["jpeg_q95"] = metric(70.01)
        result = evaluate_gates(self.protocol, report)
        self.assertFalse(result["gates"]["mechanism"]["passed"])
        self.assertFalse(result["gates"]["primary"]["checks"]["dependency"])
        self.assertFalse(result["promotion"]["allowed"])

    def test_missing_roi_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        del report["roi_metrics"]["primary"]["jpeg_q95"]["ROI002"]
        with self.assertRaisesRegex(ValueError, "ROI metrics"):
            evaluate_gates(self.protocol, report)


if __name__ == "__main__":
    unittest.main()
