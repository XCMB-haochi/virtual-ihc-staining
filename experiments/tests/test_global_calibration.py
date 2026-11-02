from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_global_calibration_cv import (
    aggregate_metrics,
    apply_affine,
    cross_fitted_parameters,
    fit_affine,
    merge_moments,
    moments_from_arrays,
    validate_protocol,
)
from search_postprocessing import evaluate


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round40_cv5_global_affine_calibration.json"


class GlobalCalibrationTests(unittest.TestCase):
    def test_merged_moments_recover_analytic_affine_mapping(self) -> None:
        prediction = np.linspace(0.0, 200.0, 101, dtype=np.float32)
        target = 1.1 * prediction + 3.0
        moments = merge_moments(
            (
                moments_from_arrays(prediction[:40], target[:40]),
                moments_from_arrays(prediction[40:], target[40:]),
            )
        )
        slope, intercept = fit_affine(moments)
        self.assertAlmostEqual(slope, 1.1, places=6)
        self.assertAlmostEqual(intercept, 3.0, places=5)

    def test_cross_fit_excludes_holdout_moments(self) -> None:
        prediction = np.arange(1.0, 6.0, dtype=np.float32)
        regular = moments_from_arrays(prediction, 2.0 * prediction + 1.0)
        anomalous = moments_from_arrays(prediction, 5.0 * prediction - 7.0)
        parameters = cross_fitted_parameters(
            {"fold0": regular, "fold1": regular, "fold2": anomalous}
        )
        slope, intercept = parameters["fold2"]
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertNotAlmostEqual(parameters["fold0"][0], 2.0)

    def test_apply_affine_clips_without_mutating_input(self) -> None:
        prediction = np.asarray([-5.0, 10.0, 200.0], dtype=np.float32)
        calibrated = apply_affine(prediction, 2.0, 3.0, 0.0, 255.0)
        np.testing.assert_array_equal(prediction, [-5.0, 10.0, 200.0])
        np.testing.assert_array_equal(calibrated, [0.0, 23.0, 255.0])

    def test_weighted_metric_aggregation_matches_direct_evaluation(self) -> None:
        target = np.zeros((3, 8, 8), dtype=np.uint8)
        prediction = np.stack(
            (
                np.full((8, 8), 10.0, dtype=np.float32),
                np.full((8, 8), 30.0, dtype=np.float32),
                np.full((8, 8), 50.0, dtype=np.float32),
            )
        )
        direct = evaluate(prediction, target, 0)
        aggregated = aggregate_metrics(
            (
                (1, evaluate(prediction[:1], target[:1], 0)),
                (2, evaluate(prediction[1:], target[1:], 0)),
            )
        )
        for field in direct:
            self.assertAlmostEqual(aggregated[field], direct[field])

    def test_round40_protocol_is_valid_and_rejects_invalid_gate(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            validate_protocol(protocol)["fit"]["method"],
            "global-affine-ols",
        )
        invalid = copy.deepcopy(protocol)
        invalid["promotion_gate"]["minimum_positive_folds"] = 6
        with self.assertRaisesRegex(ValueError, "must be in"):
            validate_protocol(invalid)


if __name__ == "__main__":
    unittest.main()
