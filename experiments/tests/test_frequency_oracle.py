from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from evaluate_frequency_oracle import (
    MetricAccumulator,
    clipped_alpha,
    dense_local_variance_oracle,
    load_protocol,
    local_highpass,
    per_image_alphas,
    unsharp_moments,
)


class FrequencyOracleTests(unittest.TestCase):
    def test_protocol_parses_registered_round72(self) -> None:
        protocol = load_protocol(PROJECT_ROOT / "configs" / "round72_frequency_oracle.json")
        self.assertEqual(protocol["analysis"]["window_size"], 7)
        self.assertEqual(protocol["analysis"]["dense_variance_threshold"], 58.5225)
        self.assertEqual(sum(protocol["baseline_weights"].values()), 1.0)
        self.assertFalse(protocol.get("test_inference_authorized", False))

    def test_optimal_alpha_recovers_synthetic_unsharp_target(self) -> None:
        prediction = np.zeros((2, 9, 9), dtype=np.float32)
        prediction[:, 2:7, 2:7] = 50.0
        _, highpass = local_highpass(prediction, 3)
        target = prediction + np.float32(0.4) * highpass
        numerator, denominator = unsharp_moments(prediction, target, highpass)
        self.assertAlmostEqual(clipped_alpha(numerator, denominator, 0.0, 1.0), 0.4, places=5)
        alphas = per_image_alphas(prediction, target, highpass, 0.0, 1.0)
        np.testing.assert_allclose(alphas, np.asarray([0.4, 0.4]), atol=1e-5)

    def test_constant_prediction_has_zero_highpass_and_minimum_alpha(self) -> None:
        prediction = np.full((1, 9, 9), 20.0, dtype=np.float32)
        _, highpass = local_highpass(prediction, 3)
        np.testing.assert_allclose(highpass, 0.0, atol=1e-6)
        alpha = per_image_alphas(
            prediction,
            np.full_like(prediction, 30.0),
            highpass,
            0.0,
            1.0,
        )
        np.testing.assert_array_equal(alpha, np.asarray([0.0], dtype=np.float32))

    def test_dense_variance_oracle_is_identity_below_threshold(self) -> None:
        prediction = np.full((1, 9, 9), 20.0, dtype=np.float32)
        prediction[:, 4, 4] = 21.0
        target = prediction.copy()
        candidate = dense_local_variance_oracle(
            prediction,
            target,
            3,
            variance_threshold=58.5225,
            epsilon=1e-6,
            maximum_alpha=1.0,
        )
        np.testing.assert_allclose(candidate, prediction, atol=1e-5)

    def test_metric_accumulator_matches_identity(self) -> None:
        image = np.arange(81, dtype=np.float32).reshape(9, 9)
        accumulator = MetricAccumulator()
        score = accumulator.add(image, image)
        result = accumulator.result()
        self.assertEqual(score, 100.0)
        self.assertEqual(result["competition_score_official"], 100.0)
        self.assertTrue(np.isinf(result["psnr_global_mse"]))

    def test_invalid_protocol_rejects_even_window(self) -> None:
        protocol = json.loads(
            (PROJECT_ROOT / "configs" / "round72_frequency_oracle.json").read_text(
                encoding="utf-8"
            )
        )
        protocol["analysis"]["window_size"] = 6
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "odd integer"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
