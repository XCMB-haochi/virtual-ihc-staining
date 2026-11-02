from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from build_roi_ablation_packages import (
    archive_entries,
    build_leave_one_roi_out,
    file_sha256,
)


class RoiAblationPackageTests(unittest.TestCase):
    def test_replaces_only_selected_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "primary.zip"
            fallback_path = root / "fallback.zip"
            output_path = root / "output.zip"
            names = [
                "results/test/CD68/ROI025_00_00.jpg",
                "results/test/CD68/ROI026_00_00.jpg",
            ]
            with zipfile.ZipFile(primary_path, "w") as archive:
                for name in names:
                    archive.writestr(name, f"primary:{name}".encode())
            with zipfile.ZipFile(fallback_path, "w") as archive:
                for name in names:
                    archive.writestr(name, f"fallback:{name}".encode())

            report = build_leave_one_roi_out(
                primary_path,
                fallback_path,
                output_path,
                "CD68",
                "ROI025",
            )

            self.assertEqual(report["fallback_images"], 1)
            with zipfile.ZipFile(output_path) as archive:
                self.assertTrue(archive.read(names[0]).startswith(b"fallback:"))
                self.assertTrue(archive.read(names[1]).startswith(b"primary:"))
                self.assertIsNone(archive.testzip())

    def test_reverse_inputs_uses_scaled_only_for_selected_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.zip"
            scaled_path = root / "scaled.zip"
            output_path = root / "output.zip"
            names = [
                f"results/test/CD68/ROI0{roi}_00_00.jpg"
                for roi in range(25, 30)
            ]
            with zipfile.ZipFile(raw_path, "w") as archive:
                for name in names:
                    archive.writestr(name, f"raw:{name}".encode())
            with zipfile.ZipFile(scaled_path, "w") as archive:
                for name in names:
                    archive.writestr(name, f"scaled:{name}".encode())

            report = build_leave_one_roi_out(
                raw_path,
                scaled_path,
                output_path,
                "CD68",
                "ROI029",
            )

            self.assertEqual(report["fallback_images"], 1)
            self.assertEqual(report["primary_sha256"], file_sha256(raw_path))
            self.assertEqual(report["fallback_sha256"], file_sha256(scaled_path))
            self.assertEqual(report["sha256"], file_sha256(output_path))
            with zipfile.ZipFile(output_path) as archive:
                self.assertEqual(archive.namelist(), sorted(names))
                for name in names[:-1]:
                    self.assertTrue(archive.read(name).startswith(b"raw:"))
                self.assertTrue(archive.read(names[-1]).startswith(b"scaled:"))
                self.assertIsNone(archive.testzip())

    def test_rejects_missing_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "primary.zip"
            fallback_path = root / "fallback.zip"
            output_path = root / "output.zip"
            name = "results/test/CD68/ROI025_00_00.jpg"
            for path in (primary_path, fallback_path):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(name, b"jpeg")

            with self.assertRaisesRegex(ValueError, "ROI029"):
                build_leave_one_roi_out(
                    primary_path,
                    fallback_path,
                    output_path,
                    "CD68",
                    "ROI029",
                )
            self.assertFalse(output_path.exists())

    def test_rejects_output_equal_to_input_without_modifying_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.zip"
            name = "results/test/CD68/ROI029_00_00.jpg"
            with zipfile.ZipFile(input_path, "w") as archive:
                archive.writestr(name, b"jpeg")
            original_digest = file_sha256(input_path)

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                build_leave_one_roi_out(
                    input_path,
                    input_path,
                    input_path,
                    "CD68",
                    "ROI029",
                )

            self.assertEqual(file_sha256(input_path), original_digest)

    def test_rejects_mismatched_archive_sets_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "primary.zip"
            fallback_path = root / "fallback.zip"
            output_path = root / "output.zip"
            with zipfile.ZipFile(primary_path, "w") as archive:
                archive.writestr(
                    "results/test/CD68/ROI025_00_00.jpg",
                    b"primary",
                )
            with zipfile.ZipFile(fallback_path, "w") as archive:
                archive.writestr(
                    "results/test/CD68/ROI029_00_00.jpg",
                    b"fallback",
                )

            with self.assertRaisesRegex(ValueError, "file sets differ"):
                build_leave_one_roi_out(
                    primary_path,
                    fallback_path,
                    output_path,
                    "CD68",
                    "ROI029",
                )

            self.assertFalse(output_path.exists())

    def test_failed_build_preserves_existing_output_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "primary.zip"
            fallback_path = root / "fallback.zip"
            output_path = root / "output.zip"
            with zipfile.ZipFile(primary_path, "w") as archive:
                archive.writestr(
                    "results/test/CD68/ROI025_00_00.jpg",
                    b"primary",
                )
            with zipfile.ZipFile(fallback_path, "w") as archive:
                archive.writestr(
                    "results/test/CD68/ROI029_00_00.jpg",
                    b"fallback",
                )
            output_path.write_bytes(b"existing output")
            original_digest = file_sha256(output_path)

            with self.assertRaisesRegex(ValueError, "file sets differ"):
                build_leave_one_roi_out(
                    primary_path,
                    fallback_path,
                    output_path,
                    "CD68",
                    "ROI029",
                )

            self.assertEqual(file_sha256(output_path), original_digest)
            self.assertEqual(list(root.glob(f".{output_path.name}.*.tmp")), [])

    def test_rejects_duplicate_patch_filename_under_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "ambiguous.zip"
            patch_name = "ROI025_00_00.jpg"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"results/test/CD68/{patch_name}", b"canonical")
                archive.writestr(
                    f"results/test/CD68/nested/{patch_name}", b"ambiguous"
                )

            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "Duplicate patch filename"):
                    archive_entries(archive, "CD68")


if __name__ == "__main__":
    unittest.main()
