from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_affine_calibration import (
    apply_regression,
    fit_regression,
    validation_roi_moments,
)


class AffineCalibrationTests(unittest.TestCase):
    def test_standardized_regression_recovers_linear_target(self) -> None:
        rows = [
            {
                "source_mean": float(mean),
                "source_std": float(std),
                "target_std": 2.0 * mean - 0.5 * std + 3.0,
            }
            for mean, std in ((1, 2), (2, 5), (4, 3), (7, 8), (9, 4))
        ]
        model = fit_regression(
            rows, ("source_mean", "source_std"), "target_std"
        )
        prediction = apply_regression(
            model, {"source_mean": 6.0, "source_std": 7.0}
        )
        self.assertAlmostEqual(prediction, 11.5, places=10)

    def test_validation_moments_pool_all_pixels_by_roi(self) -> None:
        names = ["ROI001_00_00.jpg", "ROI001_00_01.jpg", "ROI002_00_00.jpg"]
        sources = np.asarray(
            [
                [[0, 2], [4, 6]],
                [[8, 10], [12, 14]],
                [[3, 3], [3, 3]],
            ],
            dtype=np.uint8,
        )
        predictions = sources.astype(np.float32) + 1.0
        moments = validation_roi_moments(names, sources, predictions)

        self.assertAlmostEqual(moments["ROI001"]["source_mean"], 7.0)
        self.assertAlmostEqual(
            moments["ROI001"]["source_std"], np.std(np.arange(0, 16, 2))
        )
        self.assertAlmostEqual(moments["ROI001"]["prediction_mean"], 8.0)
        self.assertEqual(moments["ROI002"]["source_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
