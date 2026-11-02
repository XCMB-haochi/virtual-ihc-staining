from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_checkpoint_selection_gate import compare_reports, official_score


def metric(score: float, *, legacy: bool = False) -> dict[str, float]:
    key = (
        "competition_score_mean_image_psnr" if legacy else "competition_score_official"
    )
    return {key: score}


class CheckpointSelectionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "name": "round41",
            "inference": {"jpeg_quality": 95},
            "folds": [
                {"name": "fold0", "val_rois": ["ROI000"]},
                {"name": "fold2", "val_rois": ["ROI001", "ROI002"]},
            ],
            "checkpoint_selection_gate": {
                "baseline_protocol_name": "round35",
                "candidate": "ensemble",
                "baseline_checkpoint_filename": "best_global_mse.pt",
                "candidate_checkpoint_filename": "best_mean_image.pt",
                "changed_folds": ["fold2"],
                "minimum_overall_q95_gain": 0.02,
                "require_positive_float": True,
                "minimum_changed_fold_q95_gain": 0.05,
                "minimum_positive_changed_rois": 2,
                "minimum_changed_roi_q95_gain": -0.1,
                "maximum_unchanged_fold_abs_delta": 1e-6,
            },
        }
        self.baseline = self.report(
            "round35",
            "best_global_mse.pt",
            overall_q95=70.0,
            overall_float=70.0,
            fold2_q95=70.0,
            roi1_q95=70.0,
            roi2_q95=70.0,
            legacy=True,
        )
        self.candidate = self.report(
            "round41",
            "best_mean_image.pt",
            overall_q95=70.1,
            overall_float=70.05,
            fold2_q95=70.2,
            roi1_q95=70.2,
            roi2_q95=70.1,
            legacy=False,
        )

    @staticmethod
    def report(
        protocol_name: str,
        checkpoint_filename: str,
        *,
        overall_q95: float,
        overall_float: float,
        fold2_q95: float,
        roi1_q95: float,
        roi2_q95: float,
        legacy: bool,
    ) -> dict[str, object]:
        return {
            "protocol_name": protocol_name,
            "marker": "CD68",
            "validation_images": 3,
            "validation_rois": ["ROI000", "ROI001", "ROI002"],
            "checkpoint_filename": checkpoint_filename,
            "overall": {
                "ensemble": {
                    "float": metric(overall_float, legacy=legacy),
                    "jpeg_q95": metric(overall_q95, legacy=legacy),
                }
            },
            "fold_metrics": {
                "fold0": {
                    "ensemble": {
                        "jpeg_q95": metric(70.0, legacy=legacy),
                    }
                },
                "fold2": {
                    "ensemble": {
                        "jpeg_q95": metric(fold2_q95, legacy=legacy),
                    }
                },
            },
            "roi_metrics": {
                "ensemble": {
                    "jpeg_q95": {
                        "ROI000": metric(70.0, legacy=legacy),
                        "ROI001": metric(roi1_q95, legacy=legacy),
                        "ROI002": metric(roi2_q95, legacy=legacy),
                    }
                }
            },
        }

    def test_official_score_accepts_current_and_legacy_keys(self) -> None:
        self.assertEqual(official_score(metric(71.0)), 71.0)
        self.assertEqual(official_score(metric(72.0, legacy=True)), 72.0)

    def test_gate_passes_stable_official_checkpoint_gain(self) -> None:
        result = compare_reports(self.protocol, self.baseline, self.candidate)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["overall_q95_gain"], 0.1)
        self.assertEqual(result["positive_changed_rois"], ["ROI001", "ROI002"])
        self.assertEqual(result["maximum_unchanged_fold_abs_delta"], 0.0)

    def test_gate_rejects_changed_roi_regression(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["roi_metrics"]["ensemble"]["jpeg_q95"]["ROI001"] = metric(69.0)
        result = compare_reports(self.protocol, self.baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["changed_roi_floor"])
        self.assertFalse(result["checks"]["positive_changed_rois"])


if __name__ == "__main__":
    unittest.main()
