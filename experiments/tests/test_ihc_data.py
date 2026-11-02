from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ihc_data import (
    PatchId,
    apply_roi_mean_conditioning,
    choose_evenly,
    fit_roi_mean_conditioning,
    linear_fit,
    load_context_gray,
    parse_patch_name,
    rbf_roi_weights,
    roi_intensity_means,
    split_names_by_roi,
)
from ihc_torch import (
    CompetitionLoss,
    ExactCompetitionLoss,
    FeatureFiLM,
    PairedPatchDataset,
    UNet,
    aligned_condition_from_augmentation,
    apply_d4_transform,
    batch_condition_tensor,
    center_crop_like,
    checkpoint_conditioning,
    competition_score,
    crop_context_halo,
    invert_d4_transform,
    predict_with_stride_phase_tta,
    predict_with_tta,
    psnr_from_global_error,
    psnr_per_channel,
    ssim_per_channel,
    ssim_uniform_per_channel,
    shift_array_edge,
    shift_tensor_edge,
)
from ihc_spatial import (
    SPATIAL_CONDITION_DIMENSION,
    SPATIAL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
    SPATIAL_THUMBNAIL_CHANNELS,
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
    SPATIAL_THUMBNAIL_SIDE,
    build_spatial_thumbnail_aligned_conditioning,
    spatial_conditioning_specification,
    spatial_thumbnail_aligned_conditioning_specification,
    spatial_thumbnail_conditioning_specification,
    spatial_thumbnail_multiscale_conditioning_specification,
    spatial_thumbnail_ring_conditioning_specification,
)


class PatchNameTests(unittest.TestCase):
    def test_parse_patch_name(self) -> None:
        self.assertEqual(
            parse_patch_name("ROI025_03_14.jpg"), PatchId("ROI025", 3, 14)
        )

    def test_reject_unexpected_name(self) -> None:
        with self.assertRaises(ValueError):
            parse_patch_name("sample.jpg")


class SplitTests(unittest.TestCase):
    def test_split_by_roi(self) -> None:
        names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg", "ROI002_00_00.jpg"]
        train, val = split_names_by_roi(names, {"ROI002"})
        self.assertEqual(train, names[:2])
        self.assertEqual(val, names[2:])

    def test_split_by_explicit_training_roi_subset(self) -> None:
        names = [
            "ROI000_00_00.jpg",
            "ROI001_00_00.jpg",
            "ROI002_00_00.jpg",
            "ROI003_00_00.jpg",
        ]
        train, val = split_names_by_roi(
            names,
            {"ROI003"},
            {"ROI000", "ROI002"},
        )
        self.assertEqual(train, [names[0], names[2]])
        self.assertEqual(val, [names[3]])

    def test_explicit_training_rois_are_validated(self) -> None:
        names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_names_by_roi(names, {"ROI001"}, {"ROI001"})
        with self.assertRaisesRegex(ValueError, "Unknown training"):
            split_names_by_roi(names, {"ROI001"}, {"ROI002"})
        with self.assertRaisesRegex(ValueError, "Unknown validation"):
            split_names_by_roi(names, {"ROI002"}, {"ROI000"})

    def test_even_selection_includes_endpoints(self) -> None:
        names = [str(index) for index in range(10)]
        selected = choose_evenly(names, 4)
        self.assertEqual(selected[0], "0")
        self.assertEqual(selected[-1], "9")
        self.assertEqual(len(np.unique(selected)), 4)

    def test_linear_fit(self) -> None:
        slope, intercept = linear_fit([1.0, 2.0, 3.0], [3.0, 5.0, 7.0])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)

    def test_roi_intensity_means_and_rbf_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            images = {
                "ROI000_00_00.jpg": 10,
                "ROI000_00_01.jpg": 20,
                "ROI001_00_00.jpg": 40,
            }
            for name, value in images.items():
                Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(
                    directory / name,
                    quality=100,
                )

            means, center = roi_intensity_means(directory, list(images))
            self.assertAlmostEqual(means["ROI000"], 15.0)
            self.assertAlmostEqual(means["ROI001"], 40.0)
            self.assertAlmostEqual(center, 70.0 / 3.0)

            names = list(images)
            weights = rbf_roi_weights(names, means, 15.0, alpha=0.5, sigma=2.0)
            tile_mean = np.mean(
                [weights[parse_patch_name(name).roi] for name in names]
            )
            self.assertAlmostEqual(tile_mean, 1.0)
            self.assertGreater(weights["ROI000"], weights["ROI001"])

    def test_roi_mean_conditioning_uses_equal_roi_standardization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = {
                "ROI000_00_00.jpg": 10,
                "ROI000_00_01.jpg": 20,
                "ROI001_00_00.jpg": 40,
                "ROI002_00_00.jpg": 30,
            }
            for name, value in values.items():
                Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(
                    directory / name
                )
            names = list(values)
            fit_names = names[:3]

            conditions, report = fit_roi_mean_conditioning(
                directory, names, fit_names
            )

            self.assertAlmostEqual(report["normalization"]["center"], 27.5)
            self.assertAlmostEqual(report["normalization"]["scale"], 12.5)
            self.assertAlmostEqual(conditions["ROI000"], -1.0)
            self.assertAlmostEqual(conditions["ROI001"], 1.0)
            self.assertAlmostEqual(conditions["ROI002"], 0.2)
            reapplied = apply_roi_mean_conditioning(directory, names, report)
            self.assertEqual(conditions, reapplied)

    def test_roi_mean_conditioning_rejects_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = "ROI000_00_00.jpg"
            Image.fromarray(np.full((4, 4), 10, dtype=np.uint8)).save(
                directory / name
            )
            with self.assertRaisesRegex(ValueError, "metadata"):
                apply_roi_mean_conditioning(directory, [name], None)

    def test_rbf_roi_weights_reject_invalid_inputs(self) -> None:
        names = ["ROI000_00_00.jpg"]
        with self.assertRaisesRegex(ValueError, "missing"):
            rbf_roi_weights(names, {}, 10.0, alpha=0.5, sigma=1.5)
        with self.assertRaisesRegex(ValueError, "alpha"):
            rbf_roi_weights(names, {"ROI000": 10.0}, 10.0, alpha=-0.1, sigma=1.5)
        with self.assertRaisesRegex(ValueError, "sigma"):
            rbf_roi_weights(names, {"ROI000": 10.0}, 10.0, alpha=0.5, sigma=0.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            rbf_roi_weights(
                names,
                {"ROI000": float("nan")},
                10.0,
                alpha=0.5,
                sigma=1.5,
            )

    def test_context_uses_available_neighbors_and_reflects_missing_ones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            center = np.full((256, 256), 100, dtype=np.uint8)
            top = np.full((256, 256), 20, dtype=np.uint8)
            left = np.full((256, 256), 40, dtype=np.uint8)
            Image.fromarray(center).save(directory / "ROI000_01_01.jpg", quality=100)
            Image.fromarray(top).save(directory / "ROI000_00_01.jpg", quality=100)
            Image.fromarray(left).save(directory / "ROI000_01_00.jpg", quality=100)

            context = load_context_gray(directory, "ROI000_01_01.jpg", halo=8)
            self.assertEqual(context.shape, (272, 272))
            self.assertTrue(np.all(context[:8, 8:264] == 20))
            self.assertTrue(np.all(context[8:264, :8] == 40))
            self.assertTrue(np.all(context[8:264, 8:264] == 100))
            self.assertTrue(np.all(context[:8, :8] == 100))


class TorchComponentTests(unittest.TestCase):
    def test_aligned_thumbnail_crop_matches_context_bottleneck_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names: list[str] = []
            for tile_row in range(3):
                for tile_col in range(3):
                    name = f"ROI000_{tile_row:02d}_{tile_col:02d}.jpg"
                    block_values = (
                        tile_row * 61
                        + tile_col * 17
                        + np.add.outer(np.arange(16) * 3, np.arange(16))
                    ) % 256
                    image = np.repeat(
                        np.repeat(block_values.astype(np.uint8), 16, axis=0),
                        16,
                        axis=1,
                    )
                    Image.fromarray(image).save(directory / name, quality=100)
                    names.append(name)

            center_name = "ROI000_01_01.jpg"
            conditions, _ = build_spatial_thumbnail_aligned_conditioning(
                directory,
                names,
            )
            thumbnail = conditions[center_name][SPATIAL_CONDITION_DIMENSION:].reshape(
                SPATIAL_THUMBNAIL_CHANNELS,
                SPATIAL_THUMBNAIL_SIDE,
                SPATIAL_THUMBNAIL_SIDE,
            )
            context = load_context_gray(directory, center_name, halo=64) / 255.0
            bottleneck_grid = context.reshape(24, 16, 24, 16).mean(axis=(1, 3))

            np.testing.assert_allclose(
                thumbnail[0, 28:52, 28:52],
                bottleneck_grid,
                rtol=0,
                atol=2e-7,
            )
            np.testing.assert_array_equal(thumbnail[1, 28:52, 28:52], 1.0)

    def test_paired_dataset_reports_the_applied_augmentation(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi = root / "train" / "DAPI"
            cd68 = root / "train" / "CD68"
            dapi.mkdir(parents=True)
            cd68.mkdir(parents=True)
            image = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(image).save(dapi / "ROI000_00_00.jpg", quality=100)
            Image.fromarray(image).save(cd68 / "ROI000_00_00.jpg", quality=100)
            baseline = PairedPatchDataset(
                root,
                ["ROI000_00_00.jpg"],
                markers=("CD68",),
            )[0][0]
            dataset = PairedPatchDataset(
                root,
                ["ROI000_00_00.jpg"],
                markers=("CD68",),
                augment=True,
                return_augmentation=True,
            )
            with mock.patch("ihc_torch.random.random", side_effect=(0.1, 0.9)), mock.patch(
                "ihc_torch.random.randrange",
                return_value=1,
            ):
                source, _, augmentation, name = dataset[0]

            self.assertEqual(name, "ROI000_00_00.jpg")
            torch.testing.assert_close(
                source,
                torch.rot90(torch.flip(baseline, dims=(-1,)), 1, dims=(-2, -1)),
            )
            torch.testing.assert_close(augmentation, torch.tensor([1, 0, 1]))

    def test_source_photometric_augmentation_changes_only_dapi(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi = root / "train" / "DAPI"
            cd68 = root / "train" / "CD68"
            dapi.mkdir(parents=True)
            cd68.mkdir(parents=True)
            Image.fromarray(np.full((16, 16), 100, dtype=np.uint8)).save(
                dapi / "ROI000_00_00.jpg", quality=100
            )
            Image.fromarray(np.full((16, 16), 50, dtype=np.uint8)).save(
                cd68 / "ROI000_00_00.jpg", quality=100
            )
            dataset = PairedPatchDataset(
                root,
                ["ROI000_00_00.jpg"],
                markers=("CD68",),
                source_photometric_augmentation=True,
            )
            with mock.patch("ihc_torch.random.random", return_value=0.1), mock.patch(
                "ihc_torch.random.uniform", side_effect=(1.0, 1.0, 0.1)
            ):
                source, target, _ = dataset[0]

            self.assertGreater(float(source.mean()), 100 / 255.0)
            torch.testing.assert_close(
                target,
                torch.full_like(target, 50 / 255.0),
                rtol=0,
                atol=1 / 255.0,
            )

    def test_paired_affine_uses_shared_transform_and_safe_outer_context(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi = root / "train" / "DAPI"
            cd68 = root / "train" / "CD68"
            dapi.mkdir(parents=True)
            cd68.mkdir(parents=True)
            image = np.full((256, 256), 100, dtype=np.uint8)
            Image.fromarray(image).save(dapi / "ROI000_00_00.jpg", quality=100)
            Image.fromarray(image).save(cd68 / "ROI000_00_00.jpg", quality=100)
            dataset = PairedPatchDataset(
                root,
                ["ROI000_00_00.jpg"],
                markers=("CD68",),
                augment=True,
                context_halo=64,
                target_context_halo=64,
                paired_affine_probability=1.0,
                paired_affine_outer_halo=256,
                paired_affine_degrees=45.0,
                paired_affine_scale_min=0.72,
                paired_affine_scale_max=1.3,
            )
            parameters = (45.0, [0, 0], 0.72, [0.0, 0.0])
            with mock.patch(
                "ihc_torch.RandomAffine.get_params",
                return_value=parameters,
            ):
                source, target, name = dataset[0]

            self.assertEqual(name, "ROI000_00_00.jpg")
            self.assertEqual(tuple(source.shape), (1, 384, 384))
            self.assertEqual(tuple(target.shape), (1, 384, 384))
            torch.testing.assert_close(source, target, rtol=0, atol=0)
            self.assertGreater(float(source.min()), 0.35)

    def test_paired_affine_rejects_incompatible_modes_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "requires augment=True"):
                PairedPatchDataset(
                    root,
                    [],
                    paired_affine_probability=0.8,
                    paired_affine_outer_halo=256,
                    paired_affine_degrees=45.0,
                    paired_affine_scale_min=0.72,
                    paired_affine_scale_max=1.3,
                )
            with self.assertRaisesRegex(ValueError, "incompatible with aligned D4"):
                PairedPatchDataset(
                    root,
                    [],
                    augment=True,
                    return_augmentation=True,
                    paired_affine_probability=0.8,
                    paired_affine_outer_halo=256,
                    paired_affine_degrees=45.0,
                    paired_affine_scale_min=0.72,
                    paired_affine_scale_max=1.3,
                )

    def test_paired_dataset_can_supervise_the_full_context(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi = root / "train" / "DAPI"
            cd68 = root / "train" / "CD68"
            dapi.mkdir(parents=True)
            cd68.mkdir(parents=True)
            names_and_values = {
                "ROI000_01_01.jpg": (10, 100),
                "ROI000_00_01.jpg": (20, 30),
                "ROI000_01_00.jpg": (40, 50),
            }
            for name, (source_value, target_value) in names_and_values.items():
                Image.fromarray(
                    np.full((16, 16), source_value, dtype=np.uint8)
                ).save(dapi / name, quality=100)
                Image.fromarray(
                    np.full((16, 16), target_value, dtype=np.uint8)
                ).save(cd68 / name, quality=100)

            dataset = PairedPatchDataset(
                root,
                ["ROI000_01_01.jpg"],
                markers=("CD68",),
                context_halo=4,
                target_context_halo=4,
            )
            source, target, name = dataset[0]

            self.assertEqual(name, "ROI000_01_01.jpg")
            self.assertEqual(tuple(source.shape), (1, 24, 24))
            self.assertEqual(tuple(target.shape), (1, 24, 24))
            torch.testing.assert_close(
                target[0, 4:20, 4:20],
                torch.full((16, 16), 100 / 255.0),
                rtol=0,
                atol=1 / 255.0,
            )
            torch.testing.assert_close(
                target[0, :4, 4:20],
                torch.full((4, 16), 30 / 255.0),
                rtol=0,
                atol=1 / 255.0,
            )
            torch.testing.assert_close(
                target[0, 4:20, :4],
                torch.full((16, 4), 50 / 255.0),
                rtol=0,
                atol=1 / 255.0,
            )

            legacy = PairedPatchDataset(
                root,
                ["ROI000_01_01.jpg"],
                markers=("CD68",),
                context_halo=4,
            )
            self.assertEqual(tuple(legacy[0][1].shape), (1, 16, 16))

            with self.assertRaisesRegex(ValueError, "between 0 and context_halo"):
                PairedPatchDataset(
                    root,
                    ["ROI000_01_01.jpg"],
                    markers=("CD68",),
                    context_halo=4,
                    target_context_halo=5,
                )

    def test_target_source_offset_uses_same_edge_sampling_as_tensor_shift(self) -> None:
        import torch

        array = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        expected = np.array(
            [[[1, 2, 3, 3], [5, 6, 7, 7], [9, 10, 11, 11]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(shift_array_edge(array, (0, 1)), expected)
        torch.testing.assert_close(
            shift_tensor_edge(torch.from_numpy(array), (0, 1)),
            torch.from_numpy(expected),
        )
        self.assertIs(shift_array_edge(array, (0, 0)), array)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dapi = root / "train" / "DAPI"
            cd68 = root / "train" / "CD68"
            dapi.mkdir(parents=True)
            cd68.mkdir(parents=True)
            image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
            Image.fromarray(image).save(dapi / "ROI000_00_00.jpg", quality=100)
            Image.fromarray(image).save(cd68 / "ROI000_00_00.jpg", quality=100)
            plain = PairedPatchDataset(
                root, ["ROI000_00_00.jpg"], markers=("CD68",)
            )[0][1]
            shifted = PairedPatchDataset(
                root,
                ["ROI000_00_00.jpg"],
                markers=("CD68",),
                target_source_offset=(0, 1),
            )[0][1]
            torch.testing.assert_close(
                shifted,
                shift_tensor_edge(plain, (0, 1)),
                rtol=0,
                atol=0,
            )

        with self.assertRaisesRegex(ValueError, "target_source_offset"):
            PairedPatchDataset(Path("dataset"), [], target_source_offset=(0, 9))

    def test_unet_output_shape_and_range(self) -> None:
        import torch

        model = UNet(out_channels=4, base_channels=8)
        result = model(torch.rand(2, 1, 64, 64)).detach()
        self.assertEqual(tuple(result.shape), (2, 4, 64, 64))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_dense_unet_output_shape_and_backward(self) -> None:
        import torch

        model = UNet(out_channels=1, base_channels=8, block_type="dense")
        source = torch.rand(2, 1, 64, 64)
        result = model(source)
        result.mean().backward()
        self.assertEqual(tuple(result.shape), (2, 1, 64, 64))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_roi_mean_film_preserves_initial_output_and_receives_gradient(self) -> None:
        import torch

        torch.manual_seed(41)
        baseline = UNet(out_channels=1, base_channels=8)
        torch.manual_seed(41)
        conditioned = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode="roi-mean-film",
        )
        baseline_state = baseline.state_dict()
        conditioned_state = conditioned.state_dict()
        self.assertEqual(
            set(conditioned_state) - set(baseline_state),
            {"condition_film.weight", "condition_film.bias"},
        )
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, conditioned_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        condition = torch.tensor([[-1.0], [1.0]])
        expected = baseline(source)
        actual = conditioned(source, condition)
        self.assertTrue(torch.equal(expected, actual))

        actual.mean().backward()
        film_gradients = [
            parameter.grad
            for name, parameter in conditioned.named_parameters()
            if name.startswith("condition_film")
        ]
        self.assertTrue(all(gradient is not None for gradient in film_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in film_gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in film_gradients), 0)

    def test_conditioned_unet_validates_condition(self) -> None:
        import torch

        model = UNet(out_channels=1, base_channels=8, condition_mode="roi-mean-film")
        source = torch.rand(2, 1, 32, 32)
        with self.assertRaisesRegex(ValueError, "requires"):
            model(source)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            model(source, torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            model(source, torch.tensor([[0.0], [float("nan")]]))

        unconditioned = UNet(out_channels=1, base_channels=8)
        with self.assertRaisesRegex(ValueError, "Unconditioned"):
            unconditioned(source, torch.zeros(2, 1))

    def test_spatial_film_preserves_initial_output_and_accepts_patch_vectors(self) -> None:
        import torch

        torch.manual_seed(43)
        baseline = UNet(out_channels=1, base_channels=8)
        torch.manual_seed(43)
        conditioned = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_CONDITION_MODE,
        )
        baseline_state = baseline.state_dict()
        conditioned_state = conditioned.state_dict()
        added = set(conditioned_state) - set(baseline_state)
        self.assertTrue(added)
        self.assertTrue(all(name.startswith("condition_film.") for name in added))
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, conditioned_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        condition = torch.rand(2, SPATIAL_CONDITION_DIMENSION)
        expected = baseline(source)
        actual = conditioned(source, condition)
        self.assertTrue(torch.equal(expected, actual))
        actual.mean().backward()
        self.assertGreater(
            float(conditioned.condition_film.output.weight.grad.abs().sum()), 0.0
        )
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            conditioned(source, condition[:, :-1])

    def test_checkpoint_accepts_fixed_spatial_conditioning_specification(self) -> None:
        specification = spatial_conditioning_specification()
        self.assertEqual(
            checkpoint_conditioning(
                {
                    "conditioning": specification,
                    "args": {
                        "condition_mode": SPATIAL_CONDITION_MODE,
                        "conditioning_spec": specification,
                    },
                }
            ),
            specification,
        )

    def test_thumbnail_film_preserves_spatial_model_and_is_d4_invariant(self) -> None:
        import torch

        torch.manual_seed(47)
        spatial = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_CONDITION_MODE,
        )
        torch.manual_seed(47)
        thumbnail = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        spatial_state = spatial.state_dict()
        thumbnail_state = thumbnail.state_dict()
        added = set(thumbnail_state) - set(spatial_state)
        self.assertTrue(added)
        self.assertTrue(
            all(
                name.startswith("condition_film.thumbnail_")
                for name in added
            )
        )
        for name, value in spatial_state.items():
            self.assertTrue(torch.equal(value, thumbnail_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        stats = torch.rand(2, SPATIAL_CONDITION_DIMENSION)
        thumbnail_pixels = torch.rand(
            2,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )
        condition = torch.cat((stats, thumbnail_pixels.flatten(1)), dim=1)
        self.assertEqual(condition.shape[1], SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            spatial(source, stats),
            thumbnail(source, condition),
            rtol=0,
            atol=0,
        )

        embedding = thumbnail.condition_film.thumbnail_embedding(thumbnail_pixels)
        rotated_embedding = thumbnail.condition_film.thumbnail_embedding(
            torch.rot90(thumbnail_pixels, 1, dims=(-2, -1))
        )
        torch.testing.assert_close(embedding, rotated_embedding, rtol=1e-5, atol=1e-6)

        thumbnail(source, condition).mean().backward()
        self.assertGreater(
            float(thumbnail.condition_film.thumbnail_output.weight.grad.abs().sum()),
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            thumbnail(source, condition[:, :-1])

    def test_checkpoint_accepts_thumbnail_conditioning_specification(self) -> None:
        specification = spatial_thumbnail_conditioning_specification()
        self.assertEqual(
            checkpoint_conditioning(
                {
                    "conditioning": specification,
                    "args": {
                        "condition_mode": SPATIAL_THUMBNAIL_CONDITION_MODE,
                        "conditioning_spec": specification,
                    },
                }
            ),
            specification,
        )

    def test_aligned_thumbnail_fusion_preserves_source_and_receives_gradient(self) -> None:
        import torch

        torch.manual_seed(61)
        thumbnail = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        torch.manual_seed(61)
        aligned = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
        )
        thumbnail_state = thumbnail.state_dict()
        aligned_state = aligned.state_dict()
        added = set(aligned_state) - set(thumbnail_state)
        self.assertTrue(added)
        self.assertTrue(
            all(
                name.startswith("condition_film.aligned_")
                for name in added
            )
        )
        for name, value in thumbnail_state.items():
            self.assertTrue(torch.equal(value, aligned_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        condition = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            thumbnail(source, condition),
            aligned(source, condition),
            rtol=0,
            atol=0,
        )
        aligned(source, condition).mean().backward()
        self.assertGreater(
            float(aligned.condition_film.aligned_projection.weight.grad.abs().sum()),
            0.0,
        )

    def test_aligned_condition_uses_dataset_augmentation(self) -> None:
        import torch

        stats = torch.rand(2, SPATIAL_CONDITION_DIMENSION)
        thumbnail = torch.arange(
            2 * SPATIAL_THUMBNAIL_CHANNELS * SPATIAL_THUMBNAIL_SIDE**2,
            dtype=torch.float32,
        ).reshape(
            2,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )
        condition = torch.cat((stats, thumbnail.flatten(1)), dim=1)
        augmentation = torch.tensor(((1, 0, 1), (0, 1, 3)))
        transformed = aligned_condition_from_augmentation(condition, augmentation)
        actual = transformed[:, SPATIAL_CONDITION_DIMENSION:].reshape_as(thumbnail)
        expected0 = torch.rot90(
            torch.flip(thumbnail[0], dims=(-1,)),
            1,
            dims=(-2, -1),
        )
        expected1 = torch.rot90(
            torch.flip(thumbnail[1], dims=(-2,)),
            3,
            dims=(-2, -1),
        )
        torch.testing.assert_close(actual, torch.stack((expected0, expected1)))
        torch.testing.assert_close(
            transformed[:, :SPATIAL_CONDITION_DIMENSION],
            stats,
        )

    def test_checkpoint_accepts_aligned_conditioning_specification(self) -> None:
        specification = spatial_thumbnail_aligned_conditioning_specification()
        self.assertEqual(
            checkpoint_conditioning(
                {
                    "conditioning": specification,
                    "args": {
                        "condition_mode": SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
                        "conditioning_spec": specification,
                    },
                }
            ),
            specification,
        )

    def test_multiscale_film_preserves_thumbnail_model_and_updates_decoders(
        self,
    ) -> None:
        import torch

        torch.manual_seed(53)
        thumbnail = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        torch.manual_seed(53)
        multiscale = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
        )
        thumbnail_state = thumbnail.state_dict()
        multiscale_state = multiscale.state_dict()
        added = set(multiscale_state) - set(thumbnail_state)
        self.assertTrue(added)
        self.assertTrue(
            all(
                name.startswith("condition_film.decoder_outputs.")
                for name in added
            )
        )
        for name, value in thumbnail_state.items():
            self.assertTrue(torch.equal(value, multiscale_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        condition = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            thumbnail(source, condition),
            multiscale(source, condition),
            rtol=0,
            atol=0,
        )

        multiscale(source, condition).mean().backward()
        for index, output in enumerate(multiscale.condition_film.decoder_outputs):
            self.assertGreater(
                float(output.weight.grad.abs().sum()),
                0.0,
                f"decoder output {index}",
            )

    def test_checkpoint_accepts_multiscale_conditioning_specification(self) -> None:
        specification = spatial_thumbnail_multiscale_conditioning_specification()
        self.assertEqual(
            checkpoint_conditioning(
                {
                    "conditioning": specification,
                    "args": {
                        "condition_mode": (
                            SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE
                        ),
                        "conditioning_spec": specification,
                    },
                }
            ),
            specification,
        )

    def test_ring_film_preserves_thumbnail_model_and_is_d4_invariant(self) -> None:
        import torch

        torch.manual_seed(59)
        thumbnail = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        torch.manual_seed(59)
        ring = UNet(
            out_channels=1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
        )
        thumbnail_state = thumbnail.state_dict()
        ring_state = ring.state_dict()
        self.assertEqual(
            set(ring_state) - set(thumbnail_state),
            {
                "condition_film.ring_projection.weight",
                "condition_film.ring_projection.bias",
            },
        )
        for name, value in thumbnail_state.items():
            self.assertTrue(torch.equal(value, ring_state[name]), name)

        source = torch.rand(2, 1, 64, 64)
        condition = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            thumbnail(source, condition),
            ring(source, condition),
            rtol=0,
            atol=0,
        )

        thumbnail_pixels = condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
            2,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )
        embedding = ring.condition_film.thumbnail_embedding(thumbnail_pixels)
        rotated_embedding = ring.condition_film.thumbnail_embedding(
            torch.rot90(thumbnail_pixels, 1, dims=(-2, -1))
        )
        torch.testing.assert_close(embedding, rotated_embedding, rtol=1e-5, atol=1e-6)

        feature_map = torch.full((1, 1, 5, 5), 3.0)
        feature_map[:, :, 1:4, 1:4] = 2.0
        feature_map[:, :, 2, 2] = 1.0
        torch.testing.assert_close(
            ring.condition_film.ring_features(feature_map),
            torch.tensor([[1.0, 2.0, 3.0]]),
            rtol=0,
            atol=0,
        )

        with torch.no_grad():
            ring.condition_film.thumbnail_output.weight.fill_(0.01)
        ring.zero_grad(set_to_none=True)
        ring(source, condition).mean().backward()
        self.assertGreater(
            float(ring.condition_film.ring_projection.weight.grad.abs().sum()),
            0.0,
        )

    def test_checkpoint_accepts_ring_conditioning_specification(self) -> None:
        specification = spatial_thumbnail_ring_conditioning_specification()
        self.assertEqual(
            checkpoint_conditioning(
                {
                    "conditioning": specification,
                    "args": {
                        "condition_mode": SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
                        "conditioning_spec": specification,
                    },
                }
            ),
            specification,
        )

    def test_film_accepts_float_condition_for_half_model(self) -> None:
        import torch

        film = FeatureFiLM(4).half()
        features = torch.ones(2, 4, 2, 2, dtype=torch.float16)
        condition = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
        result = film(features, condition)

        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.equal(result, features))

    def test_checkpoint_conditioning_rejects_args_mismatch(self) -> None:
        spec = {
            "version": 1,
            "mode": "roi-mean-film",
            "feature": "dapi_roi_mean",
            "injection": "bottleneck",
            "normalization": {
                "fit_scope": "training_rois_only",
                "roi_weighting": "equal",
                "units": "uint8_0_255",
                "center": 30.0,
                "scale": 5.0,
            },
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            checkpoint_conditioning(
                {
                    "conditioning": spec,
                    "args": {"condition_mode": "none"},
                }
            )

    def test_batch_condition_tensor_maps_roi_names(self) -> None:
        import torch

        values = batch_condition_tensor(
            ["ROI001_00_00.jpg", "ROI002_00_00.jpg"],
            {"ROI001": -0.5, "ROI002": 1.5},
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(values, torch.tensor([[-0.5], [1.5]])))
        with self.assertRaisesRegex(ValueError, "missing"):
            batch_condition_tensor(
                ["ROI003_00_00.jpg"],
                {"ROI001": 0.0},
                torch.device("cpu"),
            )

        patch_values = batch_condition_tensor(
            ["ROI001_00_00.jpg", "ROI001_00_01.jpg"],
            {
                "ROI001_00_00.jpg": np.asarray([0.1, 0.2], dtype=np.float32),
                "ROI001_00_01.jpg": np.asarray([0.3, 0.4], dtype=np.float32),
            },
            torch.device("cpu"),
        )
        torch.testing.assert_close(
            patch_values,
            torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        )

    def test_d4_tta_reuses_condition_for_all_transforms(self) -> None:
        import torch

        class Recorder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conditions: list[torch.Tensor] = []

            def forward(
                self, inputs: torch.Tensor, condition: torch.Tensor
            ) -> torch.Tensor:
                self.conditions.append(condition.detach().clone())
                return inputs

        model = Recorder()
        image = torch.rand(2, 1, 8, 8)
        condition = torch.tensor([[-0.5], [1.5]])
        result = predict_with_tta(model, image, "d4", condition)

        self.assertTrue(torch.allclose(result, image))
        self.assertEqual(len(model.conditions), 8)
        self.assertTrue(
            all(torch.equal(seen, condition) for seen in model.conditions)
        )

    def test_d4_tta_jointly_transforms_aligned_thumbnail(self) -> None:
        import torch

        class Recorder(torch.nn.Module):
            condition_mode = SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE

            def __init__(self) -> None:
                super().__init__()
                self.conditions: list[torch.Tensor] = []

            def forward(
                self,
                inputs: torch.Tensor,
                condition: torch.Tensor,
            ) -> torch.Tensor:
                self.conditions.append(condition.detach().clone())
                return inputs

        model = Recorder()
        image = torch.rand(1, 1, 8, 8)
        stats = torch.rand(1, SPATIAL_CONDITION_DIMENSION)
        thumbnail = torch.arange(
            SPATIAL_THUMBNAIL_CHANNELS * SPATIAL_THUMBNAIL_SIDE**2,
            dtype=torch.float32,
        ).reshape(
            1,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )
        condition = torch.cat((stats, thumbnail.flatten(1)), dim=1)
        result = predict_with_tta(model, image, "d4", condition)

        torch.testing.assert_close(result, image)
        transform_order = (0, 4, 1, 5, 2, 6, 3, 7)
        self.assertEqual(len(model.conditions), len(transform_order))
        for seen, transform_index in zip(model.conditions, transform_order):
            torch.testing.assert_close(
                seen[:, :SPATIAL_CONDITION_DIMENSION],
                stats,
            )
            seen_thumbnail = seen[:, SPATIAL_CONDITION_DIMENSION:].reshape_as(
                thumbnail
            )
            torch.testing.assert_close(
                seen_thumbnail,
                apply_d4_transform(thumbnail, transform_index),
            )

    def test_identical_metrics(self) -> None:
        import torch

        image = torch.rand(2, 4, 32, 32)
        self.assertTrue(torch.allclose(ssim_per_channel(image, image), torch.ones(2, 4)))
        self.assertTrue(torch.all(psnr_per_channel(image, image) >= 99.0))

    def test_uniform_ssim_matches_scikit_image(self) -> None:
        import torch
        from skimage.metrics import structural_similarity

        generator = torch.Generator().manual_seed(7)
        first = torch.rand(1, 1, 32, 32, generator=generator)
        second = torch.rand(1, 1, 32, 32, generator=generator)
        actual = float(ssim_uniform_per_channel(first, second)[0, 0])
        expected = structural_similarity(
            first[0, 0].numpy(), second[0, 0].numpy(), data_range=1.0
        )
        self.assertAlmostEqual(actual, expected, places=5)

    def test_low_variance_ssim_stays_float32_under_cuda_amp(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the AMP regression test.")
        generator = torch.Generator(device="cuda").manual_seed(11)
        first = 0.02 + torch.rand(
            1, 1, 32, 32, generator=generator, device="cuda"
        ) * 1e-4
        second = first + torch.rand(
            1, 1, 32, 32, generator=generator, device="cuda"
        ) * 1e-5
        gaussian_reference = ssim_per_channel(first, second)
        uniform_reference = ssim_uniform_per_channel(first, second)
        with torch.amp.autocast("cuda"):
            gaussian_amp = ssim_per_channel(first, second)
            uniform_amp = ssim_uniform_per_channel(first, second)
        self.assertEqual(gaussian_amp.dtype, torch.float32)
        self.assertEqual(uniform_amp.dtype, torch.float32)
        self.assertTrue(torch.allclose(gaussian_amp, gaussian_reference, atol=1e-6))
        self.assertTrue(torch.allclose(uniform_amp, uniform_reference, atol=1e-6))
        self.assertLessEqual(float(gaussian_amp.max()), 1.00001)
        self.assertLessEqual(float(uniform_amp.max()), 1.00001)

    def test_competition_score(self) -> None:
        self.assertAlmostEqual(competition_score(0.8, 25.0), 71.0)
        self.assertAlmostEqual(competition_score(1.0, 60.0), 100.0)
        self.assertAlmostEqual(competition_score(0.0, -5.0), 0.0)

    def test_global_psnr_uses_pooled_mse(self) -> None:
        expected = 10.0 * np.log10(255.0**2 / 4.0)
        self.assertAlmostEqual(psnr_from_global_error(16.0, 4), expected)
        with self.assertRaises(ValueError):
            psnr_from_global_error(0.0, 0)

    def test_d4_tta_restores_orientation(self) -> None:
        import torch

        model = torch.nn.Identity()
        image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        result = predict_with_tta(model, image, "d4")
        self.assertTrue(torch.equal(result, image))

    def test_stride_phase_tta_realigns_identity_windows(self) -> None:
        import torch

        model = torch.nn.Identity()
        outer = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
        result = predict_with_stride_phase_tta(
            model,
            outer,
            target_spatial_shape=(6, 6),
            model_context_halo=1,
            mode="none",
        )
        torch.testing.assert_close(result, outer[..., 2:8, 2:8], rtol=0, atol=0)

    def test_stride_phase_tta_requires_one_pixel_outer_margin(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "one outer pixel"):
            predict_with_stride_phase_tta(
                torch.nn.Identity(),
                torch.zeros(1, 1, 9, 10),
                target_spatial_shape=(6, 6),
                model_context_halo=1,
                mode="none",
            )

    def test_all_indexed_d4_transforms_round_trip(self) -> None:
        import torch

        image = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        for transform_index in range(8):
            with self.subTest(transform_index=transform_index):
                transformed = apply_d4_transform(image, transform_index)
                restored = invert_d4_transform(transformed, transform_index)
                self.assertTrue(torch.equal(restored, image))

    def test_d4_transform_rejects_invalid_indices(self) -> None:
        import torch

        image = torch.zeros(1, 1, 8, 8)
        for transform_index in (-1, 8, 1.5, True):
            with self.subTest(transform_index=transform_index):
                with self.assertRaisesRegex(ValueError, "integer from 0 to 7"):
                    apply_d4_transform(image, transform_index)  # type: ignore[arg-type]
                with self.assertRaisesRegex(ValueError, "integer from 0 to 7"):
                    invert_d4_transform(image, transform_index)  # type: ignore[arg-type]

    def test_center_crop_like(self) -> None:
        import torch

        image = torch.arange(36).reshape(1, 1, 6, 6)
        reference = torch.empty(1, 1, 2, 4)
        expected = image[..., 2:4, 1:5]
        self.assertTrue(torch.equal(center_crop_like(image, reference), expected))

    def test_crop_context_halo_preserves_center(self) -> None:
        import torch

        image = torch.arange(120).reshape(1, 1, 10, 12)
        self.assertIs(crop_context_halo(image, 4, 4), image)
        self.assertTrue(
            torch.equal(crop_context_halo(image, 4, 2), image[..., 2:-2, 2:-2])
        )

    def test_crop_context_halo_rejects_invalid_plans(self) -> None:
        import torch

        image = torch.zeros(1, 1, 8, 8)
        for outer, inner in ((2, -1), (1, 2), (True, 0), (2, 1.5)):
            with self.subTest(outer=outer, inner=inner):
                with self.assertRaisesRegex(ValueError, "Context halos"):
                    crop_context_halo(image, outer, inner)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Cannot remove"):
            crop_context_halo(torch.zeros(1, 1, 4, 5), 3, 1)

    def test_pyramid_loss_backpropagates(self) -> None:
        import torch

        prediction = torch.rand(2, 1, 32, 32, requires_grad=True)
        target = torch.rand(2, 1, 32, 32)
        loss = CompetitionLoss(pyramid_weight=0.1)(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)

    def test_weighted_competition_loss_matches_individual_losses(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(23)
        prediction = torch.rand(2, 1, 32, 32, generator=generator)
        target = torch.rand(2, 1, 32, 32, generator=generator)
        weights = torch.tensor([1.25, 0.75])
        criterion = CompetitionLoss(ssim_mode="uniform7", pyramid_weight=0.1)

        actual = criterion(prediction, target, sample_weights=weights)
        individual = torch.stack(
            [
                criterion(
                    prediction[index : index + 1],
                    target[index : index + 1],
                )
                for index in range(2)
            ]
        )
        expected = (individual * weights).mean()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_per_sample_loss_matches_scalar_and_weighted_value_and_gradient(
        self,
    ) -> None:
        import torch

        generator = torch.Generator().manual_seed(43)
        prediction = torch.rand(3, 2, 16, 16, generator=generator)
        target = torch.rand(3, 2, 16, 16, generator=generator)
        sample_weights = torch.tensor([0.5, 1.0, 1.5])
        criterion = CompetitionLoss(ssim_mode="uniform7", pyramid_weight=0.1)

        scalar_prediction = prediction.clone().requires_grad_(True)
        vector_prediction = prediction.clone().requires_grad_(True)
        scalar = criterion(scalar_prediction, target)
        vector_mean = criterion.per_sample(vector_prediction, target).mean()
        scalar.backward()
        vector_mean.backward()

        self.assertTrue(torch.allclose(scalar, vector_mean, atol=1e-7, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                scalar_prediction.grad,
                vector_prediction.grad,
                atol=1e-7,
                rtol=0.0,
            )
        )

        weighted_prediction = prediction.clone().requires_grad_(True)
        manual_prediction = prediction.clone().requires_grad_(True)
        weighted = criterion(
            weighted_prediction,
            target,
            sample_weights=sample_weights,
        )
        manual = (
            criterion.per_sample(manual_prediction, target) * sample_weights
        ).mean()
        weighted.backward()
        manual.backward()

        self.assertTrue(torch.equal(weighted, manual))
        self.assertTrue(
            torch.equal(weighted_prediction.grad, manual_prediction.grad)
        )

    def test_target_weighted_l1_is_normalized_and_backpropagates(self) -> None:
        import torch

        target = torch.zeros(1, 1, 8, 8)
        target[..., 4:] = 1.0
        prediction = torch.ones_like(target)
        prediction[..., 4:] = 0.5
        prediction.requires_grad_(True)
        criterion = CompetitionLoss(
            ssim_weight=0.0,
            l1_weight=1.0,
            mse_weight=0.0,
            ssim_mode="uniform7",
            target_l1_weight_strength=2.0,
        )

        loss = criterion(prediction, target)
        loss.backward()

        self.assertTrue(torch.allclose(loss, torch.tensor(0.625), atol=1e-7))
        self.assertIsNotNone(prediction.grad)
        low_gradient = prediction.grad[..., :4].abs().mean()
        high_gradient = prediction.grad[..., 4:].abs().mean()
        self.assertTrue(torch.allclose(high_gradient / low_gradient, torch.tensor(3.0)))

    def test_target_weighted_l1_validates_strength(self) -> None:
        for strength in (-0.1, float("nan"), float("inf")):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "Target L1 weight strength"):
                    CompetitionLoss(target_l1_weight_strength=strength)

    def test_weighted_competition_loss_validates_weights(self) -> None:
        import torch

        prediction = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        criterion = CompetitionLoss(ssim_mode="uniform7")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            criterion(prediction, target, sample_weights=torch.ones(1))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            criterion(
                prediction,
                target,
                sample_weights=torch.tensor([1.0, 0.0]),
            )

    def test_single_channel_weight_matches_default_loss_and_gradient(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(31)
        prediction = torch.rand(2, 1, 16, 16, generator=generator)
        target = torch.rand(2, 1, 16, 16, generator=generator)
        sample_weights = torch.tensor([1.25, 0.75])
        default_prediction = prediction.clone().requires_grad_(True)
        channel_prediction = prediction.clone().requires_grad_(True)
        default_criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.1,
        )
        channel_criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.1,
            channel_weights=[7.0],
        )

        default_loss = default_criterion(
            default_prediction,
            target,
            sample_weights=sample_weights,
        )
        channel_loss = channel_criterion(
            channel_prediction,
            target,
            sample_weights=sample_weights,
        )
        default_loss.backward()
        channel_loss.backward()

        self.assertTrue(torch.allclose(default_loss, channel_loss, atol=1e-7))
        self.assertTrue(
            torch.allclose(
                default_prediction.grad,
                channel_prediction.grad,
                atol=1e-7,
            )
        )

    def test_equal_channel_weights_match_default_under_amp(self) -> None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(37)
        prediction = torch.rand(
            2,
            3,
            16,
            16,
            generator=generator,
            device=device,
        )
        target = torch.rand(
            2,
            3,
            16,
            16,
            generator=generator,
            device=device,
        )
        sample_weights = torch.tensor([1.5, 0.5], device=device)
        default_prediction = prediction.clone().requires_grad_(True)
        channel_prediction = prediction.clone().requires_grad_(True)
        default_criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.1,
        )
        channel_criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.1,
            channel_weights=[2.0, 2.0, 2.0],
        )

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            default_loss = default_criterion(
                default_prediction,
                target,
                sample_weights=sample_weights,
            )
            channel_loss = channel_criterion(
                channel_prediction,
                target,
                sample_weights=sample_weights,
            )
        default_loss.backward()
        channel_loss.backward()

        self.assertTrue(torch.allclose(default_loss, channel_loss, atol=2e-6))
        self.assertTrue(
            torch.allclose(
                default_prediction.grad,
                channel_prediction.grad,
                atol=2e-6,
            )
        )

    def test_nonuniform_channel_weights_match_manual_loss_and_gradient(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(41)
        prediction = torch.rand(2, 3, 16, 16, generator=generator)
        target = torch.rand(2, 3, 16, 16, generator=generator)
        sample_weights = torch.tensor([1.25, 0.75])
        normalized_channel_weights = torch.tensor([0.1, 0.3, 0.6])
        actual_prediction = prediction.clone().requires_grad_(True)
        expected_prediction = prediction.clone().requires_grad_(True)
        criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.15,
            channel_weights=[1.0, 3.0, 6.0],
        )
        single_channel_criterion = CompetitionLoss(
            ssim_mode="uniform7",
            pyramid_weight=0.15,
        )

        actual = criterion(
            actual_prediction,
            target,
            sample_weights=sample_weights,
        )
        per_sample = []
        for sample_index in range(prediction.shape[0]):
            per_channel = torch.stack(
                [
                    single_channel_criterion(
                        expected_prediction[
                            sample_index : sample_index + 1,
                            channel_index : channel_index + 1,
                        ],
                        target[
                            sample_index : sample_index + 1,
                            channel_index : channel_index + 1,
                        ],
                    )
                    for channel_index in range(prediction.shape[1])
                ]
            )
            per_sample.append((per_channel * normalized_channel_weights).sum())
        expected = (torch.stack(per_sample) * sample_weights).mean()
        actual.backward()
        expected.backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertTrue(
            torch.allclose(
                actual_prediction.grad,
                expected_prediction.grad,
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.equal(
                criterion.channel_weights,
                normalized_channel_weights,
            )
        )
        self.assertGreater(float(actual_prediction.grad.abs().sum()), 0.0)

    def test_channel_weights_validate_values_shape_and_channel_count(self) -> None:
        import torch

        invalid_weights = (
            [],
            [[1.0, 2.0]],
            [1.0, float("nan")],
            [1.0, float("inf")],
            [1.0, -0.1],
            [0.0, 0.0],
            [1e38, 1e38, 1e38, 1e38],
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    CompetitionLoss(channel_weights=weights)

        criterion = CompetitionLoss(
            ssim_mode="uniform7",
            channel_weights=[1.0, 1.0],
        )
        prediction = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        with self.assertRaisesRegex(ValueError, "prediction channels"):
            criterion(prediction, target)

        zero_channel = CompetitionLoss(channel_weights=[2.0, 0.0])
        self.assertTrue(
            torch.equal(
                zero_channel.channel_weights,
                torch.tensor([1.0, 0.0]),
            )
        )

    def test_unit_weights_match_unweighted_loss_and_gradient_under_amp(self) -> None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(29)
        prediction = torch.rand(
            2,
            1,
            32,
            32,
            generator=generator,
            device=device,
        )
        target = torch.rand(
            2,
            1,
            32,
            32,
            generator=generator,
            device=device,
        )
        unweighted_prediction = prediction.clone().requires_grad_(True)
        weighted_prediction = prediction.clone().requires_grad_(True)
        criterion = CompetitionLoss(ssim_mode="uniform7")

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            unweighted = criterion(unweighted_prediction, target)
            weighted = criterion(
                weighted_prediction,
                target,
                sample_weights=torch.ones(2, device=device),
            )
        unweighted.backward()
        weighted.backward()

        self.assertTrue(torch.allclose(unweighted, weighted, atol=2e-6, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                unweighted_prediction.grad,
                weighted_prediction.grad,
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_exact_competition_loss_matches_score_formula(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(19)
        prediction = torch.rand(2, 1, 32, 32, generator=generator, requires_grad=True)
        target = torch.rand(2, 1, 32, 32, generator=generator)
        actual = ExactCompetitionLoss(ssim_mode="uniform7")(prediction, target)
        expected_score = 0.7 * ssim_uniform_per_channel(
            prediction, target
        ).mean() + 0.3 * (
            psnr_per_channel(prediction, target).clamp(max=50.0) / 50.0
        ).mean()

        self.assertTrue(torch.allclose(actual, 1.0 - expected_score, atol=1e-7))
        actual.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_exact_competition_loss_is_zero_for_identical_images(self) -> None:
        import torch

        image = torch.rand(2, 1, 32, 32)
        loss = ExactCompetitionLoss(ssim_mode="uniform7")(image, image)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
