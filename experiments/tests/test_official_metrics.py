from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from ihc_torch import competition_score
from search_postprocessing import evaluate


class OfficialMetricTests(unittest.TestCase):
    def test_psnr_is_clipped_per_image_before_score_aggregation(self) -> None:
        target = np.zeros((2, 8, 8), dtype=np.uint8)
        prediction = np.stack(
            (
                np.full((8, 8), 0.25, dtype=np.float32),
                np.full((8, 8), 255.0, dtype=np.float32),
            )
        )

        result = evaluate(prediction, target, jpeg_quality=0)
        image_scores = []
        for index in range(2):
            image_ssim = structural_similarity(
                target[index].astype(np.float32),
                prediction[index],
                data_range=255,
                win_size=7,
            )
            image_psnr = peak_signal_noise_ratio(
                target[index].astype(np.float32),
                prediction[index],
                data_range=255,
            )
            image_scores.append(competition_score(image_ssim, image_psnr))

        expected = float(np.mean(image_scores))
        self.assertAlmostEqual(result["competition_score_official"], expected)
        self.assertAlmostEqual(result["competition_score"], expected)
        self.assertNotAlmostEqual(
            result["competition_score_official"],
            result["competition_score_mean_image_psnr"],
        )


if __name__ == "__main__":
    unittest.main()
