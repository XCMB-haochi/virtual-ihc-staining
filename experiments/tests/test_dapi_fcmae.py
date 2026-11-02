from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from train_dapi_fcmae import (
    parse_pretraining_markers,
    select_pretraining_names,
    select_pretraining_samples,
)


class DapiFCMAEDataScopeTests(unittest.TestCase):
    def test_fold_train_excludes_validation_rois(self) -> None:
        names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg", "ROI002_00_00.jpg"]
        self.assertEqual(
            select_pretraining_names(names, {"ROI001"}, "fold-train"),
            ["ROI000_00_00.jpg", "ROI002_00_00.jpg"],
        )

    def test_all_train_keeps_unlabelled_validation_inputs(self) -> None:
        names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg"]
        self.assertEqual(
            select_pretraining_names(names, {"ROI001"}, "all-train"),
            names,
        )

    def test_unknown_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            select_pretraining_names(["ROI000_00_00.jpg"], set(), "test")

    def test_multimarker_fold_scope_excludes_every_validation_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            train = Path(temporary)
            names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg"]
            for marker in ("DAPI", "CD68", "Vimentin"):
                directory = train / marker
                directory.mkdir()
                for name in names:
                    (directory / name).touch()
            samples = select_pretraining_samples(
                train,
                ["DAPI", "CD68", "Vimentin"],
                {"ROI001"},
                "fold-train",
            )
        self.assertEqual(
            samples,
            [
                ("DAPI", "ROI000_00_00.jpg"),
                ("CD68", "ROI000_00_00.jpg"),
                ("Vimentin", "ROI000_00_00.jpg"),
            ],
        )

    def test_multimarker_filenames_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            train = Path(temporary)
            (train / "DAPI").mkdir()
            (train / "CD68").mkdir()
            (train / "DAPI" / "ROI000_00_00.jpg").touch()
            with self.assertRaisesRegex(ValueError, "filenames must match"):
                select_pretraining_samples(
                    train,
                    ["DAPI", "CD68"],
                    set(),
                    "fold-train",
                )

    def test_pretraining_markers_require_unique_dapi_and_official_channels(
        self,
    ) -> None:
        self.assertEqual(
            parse_pretraining_markers("DAPI,CD68,HLA-DR"),
            ["DAPI", "CD68", "HLA-DR"],
        )
        for value in ("CD68", "DAPI,CD68,CD68", "DAPI,unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_pretraining_markers(value)


if __name__ == "__main__":
    unittest.main()
