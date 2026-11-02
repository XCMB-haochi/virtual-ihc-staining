from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ihc_data import PatchId
from ihc_spatial import (
    LOCAL_FEATURE_NAMES,
    SPATIAL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
    SPATIAL_THUMBNAIL_SIDE,
    SPATIAL_WHOLE_ROI_CONDITION_DIMENSION,
    SPATIAL_WHOLE_ROI_SIDE,
    apply_spatial_stats_conditioning,
    apply_spatial_thumbnail_conditioning,
    apply_spatial_thumbnail_texture_conditioning,
    apply_spatial_thumbnail_multiscale_conditioning,
    apply_spatial_thumbnail_ring_conditioning,
    apply_spatial_whole_roi_conditioning,
    build_neighborhood_thumbnails,
    build_spatial_stats_conditioning,
    build_spatial_thumbnail_conditioning,
    build_spatial_thumbnail_texture_conditioning,
    build_spatial_thumbnail_multiscale_conditioning,
    build_spatial_thumbnail_ring_conditioning,
    build_spatial_context_features,
    build_spatial_whole_roi_conditioning,
    build_whole_roi_thumbnails,
    conditioning_values_report,
    downsample_tile_mean,
    downsample_tile_texture,
    local_intensity_features,
    spatial_conditioning_specification,
    spatial_thumbnail_conditioning_specification,
    spatial_thumbnail_texture_conditioning_specification,
    spatial_thumbnail_multiscale_conditioning_specification,
    spatial_thumbnail_ring_conditioning_specification,
    spatial_whole_roi_conditioning_specification,
)


class LocalIntensityFeatureTests(unittest.TestCase):
    def test_constant_image_features_are_normalized(self) -> None:
        features = local_intensity_features(np.full((4, 5), 51, dtype=np.uint8))
        self.assertEqual(features.shape, (len(LOCAL_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features[0]), 0.2, places=6)
        self.assertAlmostEqual(float(features[1]), 0.0, places=6)
        self.assertAlmostEqual(float(features[-2]), 0.0, places=6)
        self.assertAlmostEqual(float(features[-1]), 0.0, places=6)

    def test_rejects_non_grayscale_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "grayscale"):
            local_intensity_features(np.zeros((2, 3, 1), dtype=np.float32))

    def test_block_mean_thumbnail_is_normalized_and_exact(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        thumbnail = downsample_tile_mean(image, side=2)
        expected = np.asarray(
            (
                (image[:4, :4].mean(), image[:4, 4:].mean()),
                (image[4:, :4].mean(), image[4:, 4:].mean()),
            ),
            dtype=np.float32,
        ) / 255.0
        np.testing.assert_allclose(thumbnail, expected, rtol=0, atol=1e-7)

    def test_block_mean_thumbnail_rejects_nondivisible_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "evenly divide"):
            downsample_tile_mean(np.zeros((7, 7), dtype=np.float32), side=4)

    def test_texture_thumbnail_contains_mean_contrast_and_gradient_maps(self) -> None:
        image = np.zeros((8, 8), dtype=np.float32)
        image[:4, 4:] = 64.0
        image[:2, :4] = 32.0
        texture = downsample_tile_texture(image, side=2)
        self.assertEqual(texture.shape, (3, 2, 2))
        self.assertTrue(np.isfinite(texture).all())
        self.assertAlmostEqual(float(texture[0, 0, 0]), 16.0 / 255.0, places=7)
        self.assertAlmostEqual(float(texture[0, 0, 1]), 64.0 / 255.0, places=7)
        self.assertGreater(float(texture[1].max()), 0.0)
        self.assertGreater(float(texture[2].max()), 0.0)


class SpatialContextFeatureTests(unittest.TestCase):
    def test_center_tile_observes_complete_three_by_three_neighborhood(self) -> None:
        patches = [
            PatchId("ROI000", row, col)
            for row in range(3)
            for col in range(3)
        ]
        local = np.zeros((9, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)
        local[:, 0] = np.arange(1, 10, dtype=np.float32) / 10.0
        local[:, 1] = 0.1
        table = build_spatial_context_features(patches, local, radius=2)
        center = 4
        radius1_presence = table.names.index("radius1_presence")
        radius2_presence = table.names.index("radius2_presence")
        self.assertAlmostEqual(float(table.values[center, radius1_presence]), 1.0)
        self.assertAlmostEqual(float(table.values[center, radius2_presence]), 9 / 25)
        self.assertEqual(table.values.shape[0], len(patches))
        self.assertEqual(table.values.shape[1], len(table.names))
        self.assertTrue(np.isfinite(table.values).all())

    def test_rejects_duplicate_coordinates(self) -> None:
        patches = [PatchId("ROI000", 0, 0), PatchId("ROI000", 0, 0)]
        local = np.zeros((2, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_spatial_context_features(patches, local)

    def test_neighborhood_thumbnail_contains_intensity_and_presence(self) -> None:
        patches = [
            PatchId("ROI000", row, col)
            for row in range(3)
            for col in range(3)
        ]
        tiles = np.stack(
            [np.full((2, 2), index / 10.0, dtype=np.float32) for index in range(9)]
        )
        thumbnails = build_neighborhood_thumbnails(patches, tiles)
        self.assertEqual(thumbnails.shape, (9, 2, 10, 10))
        center = thumbnails[4]
        np.testing.assert_array_equal(center[0, 4:6, 4:6], tiles[4])
        self.assertEqual(float(center[1].sum()), 9 * 2 * 2)
        self.assertTrue(np.all(center[1, :2, :2] == 0.0))

    def test_whole_roi_thumbnail_letterboxes_layout_and_marks_target(self) -> None:
        patches = [
            PatchId("ROI000", row, col)
            for row in range(2)
            for col in range(3)
        ]
        local = np.zeros((len(patches), len(LOCAL_FEATURE_NAMES)), dtype=np.float32)
        local[:, 0] = np.arange(1, len(patches) + 1, dtype=np.float32) / 10.0
        local[:, 1] = 0.05
        thumbnails = build_whole_roi_thumbnails(patches, local, side=12)
        self.assertEqual(thumbnails.shape, (6, 4, 12, 12))
        self.assertTrue(np.all(thumbnails[:, :, :2] == 0.0))
        self.assertTrue(np.all(thumbnails[:, :, 10:] == 0.0))
        self.assertEqual(float(thumbnails[0, 2].sum()), 8 * 12)
        self.assertEqual(float(thumbnails[0, 3].sum()), 4 * 4)
        self.assertTrue(np.all(thumbnails[0, 3, 2:6, :4] == 1.0))
        self.assertTrue(np.all(thumbnails[-1, 3, 6:10, 8:12] == 1.0))


class SpatialConditioningTests(unittest.TestCase):
    def test_build_and_apply_use_patch_keys_and_fixed_specification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = []
            for row in range(2):
                for col in range(2):
                    name = f"ROI000_{row:02d}_{col:02d}.jpg"
                    names.append(name)
                    Image.fromarray(
                        np.full((8, 8), 20 + 10 * row + col, dtype=np.uint8)
                    ).save(directory / name, quality=100)

            conditions, specification = build_spatial_stats_conditioning(
                directory, names
            )

            self.assertEqual(specification, spatial_conditioning_specification())
            self.assertEqual(set(conditions), set(names))
            self.assertTrue(
                all(value.shape == (SPATIAL_CONDITION_DIMENSION,) for value in conditions.values())
            )
            reapplied = apply_spatial_stats_conditioning(
                directory, names, specification
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], reapplied[name])

    def test_apply_rejects_changed_specification(self) -> None:
        specification = spatial_conditioning_specification()
        specification["condition_dimension"] = SPATIAL_CONDITION_DIMENSION + 1
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                apply_spatial_stats_conditioning(Path(temporary), [], specification)

    def test_build_and_apply_thumbnail_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = []
            for row in range(2):
                for col in range(2):
                    name = f"ROI000_{row:02d}_{col:02d}.jpg"
                    names.append(name)
                    Image.fromarray(
                        np.full((32, 32), 32 + 16 * row + col, dtype=np.uint8)
                    ).save(directory / name, quality=100)

            conditions, specification = build_spatial_thumbnail_conditioning(
                directory,
                names,
            )
            self.assertEqual(
                specification,
                spatial_thumbnail_conditioning_specification(),
            )
            self.assertEqual(set(conditions), set(names))
            self.assertTrue(
                all(
                    value.shape == (SPATIAL_THUMBNAIL_CONDITION_DIMENSION,)
                    for value in conditions.values()
                )
            )
            first_thumbnail = conditions[names[0]][SPATIAL_CONDITION_DIMENSION:].reshape(
                2,
                SPATIAL_THUMBNAIL_SIDE,
                SPATIAL_THUMBNAIL_SIDE,
            )
            self.assertEqual(float(first_thumbnail[1].sum()), 4 * 16 * 16)
            reapplied = apply_spatial_thumbnail_conditioning(
                directory,
                names,
                specification,
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], reapplied[name])

            multiscale, multiscale_specification = (
                build_spatial_thumbnail_multiscale_conditioning(
                    directory,
                    names,
                )
            )
            self.assertEqual(
                multiscale_specification,
                spatial_thumbnail_multiscale_conditioning_specification(),
            )
            multiscale_reapplied = (
                apply_spatial_thumbnail_multiscale_conditioning(
                    directory,
                    names,
                    multiscale_specification,
                )
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], multiscale[name])
                np.testing.assert_array_equal(
                    multiscale[name],
                    multiscale_reapplied[name],
                )

            ring, ring_specification = build_spatial_thumbnail_ring_conditioning(
                directory,
                names,
            )
            self.assertEqual(
                ring_specification,
                spatial_thumbnail_ring_conditioning_specification(),
            )
            ring_reapplied = apply_spatial_thumbnail_ring_conditioning(
                directory,
                names,
                ring_specification,
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], ring[name])
                np.testing.assert_array_equal(ring[name], ring_reapplied[name])

            whole_roi, whole_roi_specification = (
                build_spatial_whole_roi_conditioning(
                    directory,
                    names,
                )
            )
            self.assertEqual(
                whole_roi_specification,
                spatial_whole_roi_conditioning_specification(),
            )
            self.assertTrue(
                all(
                    value.shape == (SPATIAL_WHOLE_ROI_CONDITION_DIMENSION,)
                    for value in whole_roi.values()
                )
            )
            first_whole_roi = whole_roi[names[0]][
                SPATIAL_THUMBNAIL_CONDITION_DIMENSION:
            ].reshape(4, SPATIAL_WHOLE_ROI_SIDE, SPATIAL_WHOLE_ROI_SIDE)
            self.assertEqual(float(first_whole_roi[2].sum()), 32 * 32)
            self.assertEqual(float(first_whole_roi[3].sum()), 16 * 16)
            whole_roi_reapplied = apply_spatial_whole_roi_conditioning(
                directory,
                names,
                whole_roi_specification,
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], whole_roi[name][
                    :SPATIAL_THUMBNAIL_CONDITION_DIMENSION
                ])
                np.testing.assert_array_equal(
                    whole_roi[name],
                    whole_roi_reapplied[name],
                )

    def test_build_and_apply_texture_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = []
            for row in range(2):
                for col in range(2):
                    name = f"ROI000_{row:02d}_{col:02d}.jpg"
                    names.append(name)
                    image = np.full((32, 32), 32 + 16 * row + col, dtype=np.uint8)
                    image[:16, 16:] = np.clip(image[:16, 16:] + 20, 0, 255)
                    Image.fromarray(image).save(directory / name, quality=100)
            conditions, specification = build_spatial_thumbnail_texture_conditioning(
                directory,
                names,
            )
            self.assertEqual(
                specification,
                spatial_thumbnail_texture_conditioning_specification(),
            )
            self.assertTrue(
                all(
                    value.shape == (SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,)
                    for value in conditions.values()
                )
            )
            thumbnail = conditions[names[0]][SPATIAL_CONDITION_DIMENSION:].reshape(
                SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
                SPATIAL_THUMBNAIL_SIDE,
                SPATIAL_THUMBNAIL_SIDE,
            )
            self.assertEqual(
                float(thumbnail[-1].sum()),
                4 * SPATIAL_THUMBNAIL_SIDE**2 / 25,
            )
            reapplied = apply_spatial_thumbnail_texture_conditioning(
                directory,
                names,
                specification,
            )
            for name in names:
                np.testing.assert_array_equal(conditions[name], reapplied[name])

    def test_apply_thumbnail_rejects_changed_specification(self) -> None:
        specification = spatial_thumbnail_conditioning_specification()
        specification["condition_dimension"] = (
            SPATIAL_THUMBNAIL_CONDITION_DIMENSION + 1
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                apply_spatial_thumbnail_conditioning(
                    Path(temporary),
                    [],
                    specification,
                )

    def test_vector_report_streaming_hash_matches_legacy_matrix_hash(self) -> None:
        values = {
            "ROI000_00_01.jpg": np.asarray([3.0, 4.0], dtype=np.float32),
            "ROI000_00_00.jpg": np.asarray([1.0, 2.0], dtype=np.float32),
        }
        expected = hashlib.sha256()
        for key in sorted(values):
            expected.update(key.encode("utf-8"))
            expected.update(b"\0")
        expected.update(
            np.stack([values[key] for key in sorted(values)])
            .astype("<f4", copy=False)
            .tobytes(order="C")
        )
        report = conditioning_values_report(values)
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["dimension"], 2)
        self.assertEqual(report["sha256"], expected.hexdigest())


if __name__ == "__main__":
    unittest.main()
