from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from analyze_cross_modal_features import (
    EXPECTED_PRETRAINING_MARKERS,
    cosine_rows,
    deterministic_mismatch_indices,
    load_protocol,
    ridge_feature_report,
)


class Round109Tests(unittest.TestCase):
    def test_protocol_fixes_two_hash_bound_folds(self) -> None:
        protocol = load_protocol(
            PROJECT_ROOT / "configs" / "round109_cross_modal_feature_diagnostic.json"
        )
        self.assertEqual(len(protocol["folds"]), 2)
        self.assertEqual(protocol["embedding"]["dimension"], 768)
        self.assertEqual(protocol["mapping"]["alpha"], 10.0)
        self.assertEqual(
            EXPECTED_PRETRAINING_MARKERS,
            ["DAPI", "CD68", "CD45RO", "HLA-DR", "Vimentin"],
        )

    def test_ridge_recovers_cross_modal_linear_mapping(self) -> None:
        rng = np.random.default_rng(7)
        train_x = rng.normal(size=(128, 6))
        transform = rng.normal(size=(6, 6))
        train_y = train_x @ transform
        val_x = rng.normal(size=(40, 6))
        val_y = val_x @ transform
        names = [f"ROI000_{index // 8:02d}_{index % 8:02d}.jpg" for index in range(40)]
        report = ridge_feature_report(
            train_x,
            train_y,
            val_x,
            val_y,
            names,
            alpha=1e-6,
            solver="cholesky",
        )
        self.assertGreater(report["standardized_r2"], 0.999)

    def test_mismatch_stays_inside_each_roi_and_has_no_fixed_points(self) -> None:
        names = [
            "ROI000_00_00.jpg",
            "ROI000_00_01.jpg",
            "ROI001_00_00.jpg",
            "ROI001_00_01.jpg",
        ]
        indices = deterministic_mismatch_indices(names)
        self.assertEqual(indices.tolist(), [1, 0, 3, 2])
        self.assertTrue(np.all(indices != np.arange(len(names))))

    def test_cosine_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal two-dimensional"):
            cosine_rows(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_protocol_rejects_wrong_identity(self) -> None:
        source = json.loads(
            (PROJECT_ROOT / "configs" / "round109_cross_modal_feature_diagnostic.json")
            .read_text(encoding="utf-8")
        )
        source["name"] = "wrong"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
