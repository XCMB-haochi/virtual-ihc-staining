import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_semifinal_markers import fold_plan, image_metrics, summarize


class MarkerAuditTests(unittest.TestCase):
    def test_identical_black_image_is_finite_and_perfect(self):
        result = image_metrics(np.zeros((32, 32)), np.zeros((32, 32)))
        self.assertEqual(result["gaussian11_ssim"], 1.)
        self.assertEqual(result["uniform7_score"], 100.)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_gaussian_population_valid_window_against_direct_convolution(self):
        # Prediction is black, so JPEG encoding leaves it unchanged.
        target = np.random.default_rng(2026).integers(0, 100, (32, 32)).astype(float)
        axis = np.arange(-5, 6)
        kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 1.5 ** 2))
        kernel /= kernel.sum()
        mean = convolve(target, kernel)
        variance = convolve(target ** 2, kernel) - mean ** 2
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        expected = ((c1 * c2) / ((mean ** 2 + c1) * (variance + c2)))[5:-5, 5:-5].mean()
        result = image_metrics(np.zeros_like(target), target)
        self.assertAlmostEqual(result["gaussian11_ssim"], expected, places=12)
        self.assertNotAlmostEqual(result["gaussian11_ssim"], result["uniform7_ssim"], places=6)

    def test_pooled_psnr_does_not_replace_mean_image_score(self):
        a = image_metrics(np.zeros((32, 32)), np.ones((32, 32)) * 10)
        b = image_metrics(np.zeros((32, 32)), np.ones((32, 32)) * 100)
        result = summarize([a, b])
        self.assertGreater(result["psnr"], result["psnr_global_mse"])
        self.assertEqual(result["gaussian11_score"], (a["gaussian11_score"] + b["gaussian11_score"]) / 2)

    def test_fold_leakage_and_incomplete_universe_rejected(self):
        manifest = {"images": 2, "folds": [
            {"name": "f0", "val_rois": ["ROI001"], "images": 1},
            {"name": "f1", "val_rois": ["ROI002"], "images": 1}]}
        names = ["ROI001_00_00.jpg", "ROI002_00_00.jpg"]
        self.assertEqual(fold_plan(manifest, names)["ROI002"], "f1")
        with self.assertRaises(ValueError):
            fold_plan(manifest, names[:1])
        manifest["folds"][1]["val_rois"] = ["ROI001"]
        with self.assertRaises(ValueError):
            fold_plan(manifest, names)


if __name__ == "__main__":
    unittest.main()
