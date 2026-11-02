from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from analyze_dapi_knn import downsample_features


class Round106DiagnosticTests(unittest.TestCase):
    def test_feature_shape_is_fixed(self) -> None:
        values = np.zeros((2, 256, 256), dtype=np.uint8)
        self.assertEqual(downsample_features(values).shape, (2, 519))

    def test_feature_changes_with_morphology(self) -> None:
        flat = np.zeros((1, 256, 256), dtype=np.uint8)
        checker = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)[None]
        self.assertFalse(np.array_equal(downsample_features(flat), downsample_features(checker)))

    def test_source_script_has_crossfit_and_fixed_blend(self) -> None:
        script = (PROJECT_ROOT / "tools" / "analyze_dapi_knn.py").read_text(encoding="utf-8")
        self.assertIn("NearestNeighbors", script)
        self.assertIn("args.neighbors", script)
        self.assertIn("args.blend_weight", script)
        self.assertIn("train_indices = [i for i in range(5) if i != holdout]", script)


if __name__ == "__main__":
    unittest.main()
