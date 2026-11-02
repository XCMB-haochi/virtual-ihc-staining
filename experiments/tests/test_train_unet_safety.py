from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import train_unet
from ihc_spatial import (
    SPATIAL_CONDITION_DIMENSION,
    SPATIAL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
    spatial_conditioning_specification,
    spatial_thumbnail_aligned_conditioning_specification,
    spatial_thumbnail_conditioning_specification,
    spatial_thumbnail_multiscale_conditioning_specification,
    spatial_thumbnail_ring_conditioning_specification,
)


class ValidationScheduleTests(unittest.TestCase):
    def test_default_schedule_validates_every_epoch(self) -> None:
        self.assertEqual(train_unet.parse_validation_epochs(None, 3), [1, 2, 3])

    def test_explicit_schedule_must_be_unique_increasing_and_in_range(self) -> None:
        self.assertEqual(
            train_unet.parse_validation_epochs("2,4,6", 6),
            [2, 4, 6],
        )
        for invalid in ("", "2,2", "3,2", "0,2", "2,7", "two"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    train_unet.parse_validation_epochs(invalid, 6)


class D4ConsistencyTests(unittest.TestCase):
    class RecordingConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)
            self.outputs: list[torch.Tensor] = []

        def forward(
            self,
            inputs: torch.Tensor,
            condition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.assert_no_condition(condition)
            output = torch.sigmoid(self.conv(inputs))
            output.retain_grad()
            self.outputs.append(output)
            return output

        @staticmethod
        def assert_no_condition(condition: torch.Tensor | None) -> None:
            if condition is not None:
                raise AssertionError("Unexpected condition tensor.")

    class RecordingGradientScope(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def forward(
            self,
            inputs: torch.Tensor,
            condition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.calls.append("all")
            return inputs

        def forward_with_decoder_only_auxiliary_gradients(
            self,
            inputs: torch.Tensor,
            condition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.calls.append("decoder-only")
            return inputs

    def test_supervised_prediction_dispatches_the_registered_gradient_scope(
        self,
    ) -> None:
        model = self.RecordingGradientScope()
        source = torch.rand(1, 1, 16, 16)
        self.assertIs(
            train_unet.supervised_prediction(model, source, None, "all"),
            source,
        )
        self.assertIs(
            train_unet.supervised_prediction(
                model,
                source,
                None,
                "decoder-only",
            ),
            source,
        )
        self.assertEqual(model.calls, ["all", "decoder-only"])
        with self.assertRaisesRegex(ValueError, "Unsupported auxiliary"):
            train_unet.supervised_prediction(model, source, None, "invalid")
        with self.assertRaisesRegex(TypeError, "compatible UNeXt2"):
            train_unet.supervised_prediction(
                self.RecordingConv(),
                source,
                None,
                "decoder-only",
            )

    def test_nonidentity_transform_schedule_cycles_globally(self) -> None:
        self.assertEqual(
            [train_unet.d4_consistency_transform_index(index) for index in range(14)],
            list(range(1, 8)) * 2,
        )
        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    train_unet.d4_consistency_transform_index(invalid)  # type: ignore[arg-type]

    def test_source_offset_parser_is_strict(self) -> None:
        self.assertEqual(train_unet.parse_source_offset("0,-1"), (0, -1))
        for invalid in ("0", "0,1,2", "0,9", "0,1.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "Source offset"):
                    train_unet.parse_source_offset(invalid)

    def test_validation_applies_fixed_output_source_offset(self) -> None:
        class Identity(nn.Module):
            def forward(
                self,
                inputs: torch.Tensor,
                condition: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return inputs

        source = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
        target = train_unet.shift_tensor_edge(source, (0, -1))
        with np.errstate(divide="ignore"):
            metrics = train_unet.validate(
                Identity(),
                [(source, target, ["ROI000_00_00.jpg"])],
                torch.device("cpu"),
                ("CD68",),
                jpeg_quality=0,
                output_source_offset=(0, -1),
            )
        self.assertAlmostEqual(metrics["mean_ssim"], 1.0)
        self.assertAlmostEqual(metrics["mean_competition_score"], 100.0)

    def test_zero_weight_uses_one_forward_and_legacy_loss(self) -> None:
        torch.manual_seed(11)
        model = self.RecordingConv()
        source = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        criterion = train_unet.CompetitionLoss(ssim_mode="uniform7")

        total, supervised, consistency, feature = train_unet.training_batch_losses(
            model,
            source,
            target,
            None,
            criterion,
            0.0,
        )

        self.assertEqual(len(model.outputs), 1)
        expected = criterion(model.outputs[0], target)
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(supervised, expected)
        self.assertEqual(float(consistency), 0.0)
        self.assertEqual(float(feature), 0.0)

    def test_positive_weight_uses_two_forwards_and_both_keep_gradients(self) -> None:
        torch.manual_seed(17)
        model = self.RecordingConv()
        source = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        criterion = train_unet.CompetitionLoss(ssim_mode="uniform7")

        total, supervised, consistency, feature = train_unet.training_batch_losses(
            model,
            source,
            target,
            None,
            criterion,
            0.1,
            transform_index=1,
        )
        total.backward()

        self.assertEqual(len(model.outputs), 2)
        torch.testing.assert_close(total, supervised + 0.1 * consistency)
        self.assertGreater(float(consistency.detach()), 0.0)
        self.assertIsNotNone(model.outputs[0].grad)
        self.assertIsNotNone(model.outputs[1].grad)
        self.assertGreater(float(model.outputs[1].grad.abs().sum()), 0.0)
        self.assertEqual(float(feature), 0.0)

    def test_option_validation_is_strict(self) -> None:
        train_unet.validate_d4_consistency_options(0.0, "exact-score")
        train_unet.validate_d4_consistency_options(0.1, "hybrid")
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    train_unet.validate_d4_consistency_options(invalid, "hybrid")
        with self.assertRaisesRegex(ValueError, "requires --loss-profile=hybrid"):
            train_unet.validate_d4_consistency_options(0.1, "exact-score")


class GradientAccumulationTests(unittest.TestCase):
    def test_sample_scale_handles_full_and_short_final_groups(self) -> None:
        scales = [
            train_unet.gradient_accumulation_sample_scale(
                batch_index,
                micro_batch_size=4,
                current_batch_size=(2 if batch_index == 4 else 4),
                accumulation_steps=2,
                dataset_size=18,
            )
            for batch_index in range(5)
        ]
        self.assertEqual(scales, [0.5, 0.5, 0.5, 0.5, 1.0])

    def test_accumulated_gradient_matches_effective_batch_mean(self) -> None:
        source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        target = torch.tensor([[0.0], [1.0], [1.5], [2.0]])
        full = nn.Linear(2, 1, bias=False)
        accumulated = nn.Linear(2, 1, bias=False)
        accumulated.load_state_dict(full.state_dict())

        full_loss = (full(source) - target).square().mean()
        full_loss.backward()
        for batch_index in range(2):
            batch_slice = slice(batch_index * 2, (batch_index + 1) * 2)
            micro_loss = (
                accumulated(source[batch_slice]) - target[batch_slice]
            ).square().mean()
            scale = train_unet.gradient_accumulation_sample_scale(
                batch_index,
                micro_batch_size=2,
                current_batch_size=2,
                accumulation_steps=2,
                dataset_size=4,
            )
            (micro_loss * scale).backward()

        torch.testing.assert_close(accumulated.weight.grad, full.weight.grad)

    def test_sample_scale_rejects_out_of_range_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds the dataset"):
            train_unet.gradient_accumulation_sample_scale(3, 4, 4, 2, 8)


class ContextSupervisionTests(unittest.TestCase):
    def test_context_halo_validation(self) -> None:
        for context_halo, target_halo in ((0, 0), (64, 0), (64, 64), (256, 256)):
            with self.subTest(context_halo=context_halo, target_halo=target_halo):
                train_unet.validate_context_halos(context_halo, target_halo)

        for context_halo, target_halo in ((-1, 0), (257, 0), (64, -1), (64, 65)):
            with self.subTest(context_halo=context_halo, target_halo=target_halo):
                with self.assertRaises(ValueError):
                    train_unet.validate_context_halos(context_halo, target_halo)


class OutputDirectoryTests(unittest.TestCase):
    def test_creates_new_directory_and_accepts_existing_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            train_unet.prepare_output_dir(output)
            self.assertTrue(output.is_dir())
            train_unet.prepare_output_dir(output)

    def test_rejects_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "config.json").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                train_unet.prepare_output_dir(output)

            self.assertEqual((output / "config.json").read_text(encoding="utf-8"), "old")

    def test_main_rejects_nonempty_directory_before_dataset_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "last.pt").write_bytes(b"existing")
            argv = [
                "train_unet.py",
                str(Path(temporary) / "missing-dataset"),
                "--output-dir",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(train_unet, "resolve_data_root") as resolve_data_root,
            ):
                with self.assertRaisesRegex(FileExistsError, "resume training is not supported"):
                    train_unet.main()
            resolve_data_root.assert_not_called()


class AtomicCheckpointTests(unittest.TestCase):
    def assert_no_temporary_checkpoints(self, root: Path) -> None:
        self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_atomic_save_writes_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "nested" / "model.pt"

            train_unet.atomic_torch_save({"tensor": torch.arange(4)}, output)

            result = torch.load(output, map_location="cpu", weights_only=False)
            torch.testing.assert_close(result["tensor"], torch.arange(4))
            self.assert_no_temporary_checkpoints(output.parent)

    def test_torch_save_failure_preserves_destination_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "last.pt"
            output.write_bytes(b"previous-checkpoint")

            def fail_after_partial_write(_payload: object, path: Path) -> None:
                Path(path).write_bytes(b"partial")
                raise OSError("disk full")

            with mock.patch.object(
                train_unet.torch, "save", side_effect=fail_after_partial_write
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    train_unet.atomic_torch_save({"epoch": 2}, output)

            self.assertEqual(output.read_bytes(), b"previous-checkpoint")
            self.assert_no_temporary_checkpoints(root)

    def test_replace_failure_preserves_destination_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "last.pt"
            output.write_bytes(b"previous-checkpoint")

            with mock.patch.object(
                train_unet.os, "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    train_unet.atomic_torch_save({"epoch": 2}, output)

            self.assertEqual(output.read_bytes(), b"previous-checkpoint")
            self.assert_no_temporary_checkpoints(root)

    def test_full_and_compact_checkpoints_use_atomic_writer(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        args = SimpleNamespace(
            base_channels=8,
            block_type="residual",
            conditioning_spec={"version": 1, "mode": "none"},
        )

        with mock.patch.object(train_unet, "atomic_torch_save") as atomic_save:
            train_unet.save_checkpoint(
                Path("full.pt"),
                model,
                optimizer,
                scheduler,
                1,
                70.0,
                args,
                ("CD68",),
            )
            train_unet.save_model_checkpoint(
                Path("compact.pt"),
                model,
                1,
                70.0,
                args,
                ("CD68",),
                "global-mse",
            )

        self.assertEqual(atomic_save.call_count, 2)
        self.assertEqual(atomic_save.call_args_list[0].args[1], Path("full.pt"))
        self.assertEqual(atomic_save.call_args_list[1].args[1], Path("compact.pt"))
        self.assertIn("initialization", atomic_save.call_args_list[0].args[0])
        self.assertIn("initialization", atomic_save.call_args_list[1].args[0])

    def test_explicit_model_spec_does_not_evaluate_legacy_unet_fields(self) -> None:
        specification = {"version": 1, "architecture": "nafnet-width32"}
        args = SimpleNamespace(model_spec=specification)
        self.assertIs(train_unet.model_spec_from_args(args), specification)


class ReproducibilityMetadataTests(unittest.TestCase):
    def test_marker_loss_weights_are_normalized_in_marker_order(self) -> None:
        weights = train_unet.parse_marker_loss_weights(
            "3,1,1,1", ("CD68", "CD45RO", "HLA-DR", "Vimentin")
        )
        self.assertEqual(weights, [0.5, 1 / 6, 1 / 6, 1 / 6])
        self.assertIsNone(train_unet.parse_marker_loss_weights(None, ("CD68",)))

    def test_marker_loss_weights_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            train_unet.parse_marker_loss_weights("1", ("CD68", "CD45RO"))
        for value in ("1,nan", "1,-1", "0,0", "one,1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_unet.parse_marker_loss_weights(value, ("CD68", "CD45RO"))

    def test_data_generator_is_independent_from_global_rng(self) -> None:
        first = train_unet.seeded_data_generator(2026)
        expected = torch.rand(8, generator=first)

        torch.manual_seed(9)
        torch.rand(100)

        second = train_unet.seeded_data_generator(2026)
        actual = torch.rand(8, generator=second)
        torch.testing.assert_close(actual, expected)

    def test_runtime_metadata_records_software_without_cuda(self) -> None:
        with (
            mock.patch.object(train_unet, "git_commit", return_value="a" * 40),
            mock.patch.object(train_unet.torch.cuda, "is_available", return_value=False),
            mock.patch.object(train_unet.torch.backends.cudnn, "version", return_value=None),
        ):
            metadata = train_unet.runtime_metadata(Path("project"))

        self.assertEqual(metadata["git_commit"], "a" * 40)
        self.assertEqual(metadata["python_version"], train_unet.platform.python_version())
        self.assertEqual(metadata["torch_version"], str(torch.__version__))
        self.assertFalse(metadata["cuda_available"])
        self.assertEqual(metadata["gpu_count"], 0)
        self.assertEqual(metadata["gpus"], [])
        self.assertIn("cuda_runtime_version", metadata)
        self.assertIn("cudnn_version", metadata)

    def test_runtime_metadata_records_visible_gpu_properties(self) -> None:
        properties = SimpleNamespace(
            name="NVIDIA A100-SXM4-40GB",
            total_memory=40 * 1024**3,
            major=8,
            minor=0,
        )
        with (
            mock.patch.object(train_unet, "git_commit", return_value="b" * 40),
            mock.patch.object(train_unet.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_unet.torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                train_unet.torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            metadata = train_unet.runtime_metadata(Path("project"))

        self.assertEqual(metadata["gpu_count"], 1)
        self.assertEqual(
            metadata["gpus"],
            [
                {
                    "index": 0,
                    "name": "NVIDIA A100-SXM4-40GB",
                    "total_memory_bytes": 40 * 1024**3,
                    "compute_capability": "8.0",
                }
            ],
        )


class GroupDROTests(unittest.TestCase):
    @staticmethod
    def options(**overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "robust_objective": "group-dro",
            "group_dro_eta": 5.0,
            "loss_profile": "hybrid",
            "roi_weight_mode": "none",
            "ema_decay": 0.0,
            "marker_loss_weights_resolved": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cli_defaults_and_none_mode_leave_group_state_disabled(self) -> None:
        parser = train_unet.argparse.ArgumentParser()
        train_unet.add_robust_objective_arguments(parser)
        args = parser.parse_args([])

        self.assertEqual(args.robust_objective, "none")
        self.assertEqual(args.group_dro_eta, 5.0)
        self.assertIsNone(
            train_unet.initialize_group_dro(
                args.robust_objective,
                ["ROI000_00_00.jpg"],
                args.group_dro_eta,
            )
        )
        legacy_options = self.options(
            robust_objective="none",
            roi_weight_mode="test-mean-rbf",
            ema_decay=0.9,
            marker_loss_weights_resolved=[1.0],
        )
        train_unet.validate_robust_objective_options(legacy_options)

    def test_group_dro_rejects_unregistered_combinations(self) -> None:
        invalid = (
            (self.options(group_dro_eta=0.0), "eta"),
            (self.options(loss_profile="exact-score"), "loss-profile"),
            (self.options(roi_weight_mode="test-mean-rbf"), "roi-weight-mode"),
            (self.options(ema_decay=0.9), "ema-decay"),
            (
                self.options(marker_loss_weights_resolved=[1.0]),
                "marker-loss-weights",
            ),
        )
        for options, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                train_unet.validate_robust_objective_options(options)

    def test_empirical_p_and_uniform_q_equalize_total_group_contribution(self) -> None:
        names = [
            "ROI000_00_00.jpg",
            "ROI001_00_00.jpg",
            "ROI001_00_01.jpg",
            "ROI001_00_02.jpg",
        ]
        p = train_unet.empirical_group_probabilities(names)
        q = {"ROI000": 0.5, "ROI001": 0.5}
        q_over_p = train_unet.group_dro_q_over_p(q, p)

        self.assertEqual(p, {"ROI000": 0.25, "ROI001": 0.75})
        self.assertEqual(q_over_p, {"ROI000": 2.0, "ROI001": 2.0 / 3.0})
        for group in q:
            self.assertAlmostEqual(p[group] * q_over_p[group], q[group])
        self.assertAlmostEqual(
            sum(p[group] * q_over_p[group] for group in q),
            1.0,
        )

    def test_group_state_uses_training_names_and_requires_post_limit_coverage(self) -> None:
        full_names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg"]
        state = train_unet.initialize_group_dro("group-dro", full_names, 5.0)

        self.assertEqual(state["groups"], ["ROI000", "ROI001"])
        self.assertEqual(state["counts"], {"ROI000": 1, "ROI001": 1})
        self.assertNotIn("ROI020", state["p"])
        self.assertEqual(
            train_unet.validate_training_group_coverage(full_names, full_names),
            ("ROI000", "ROI001"),
        )
        with self.assertRaisesRegex(ValueError, r"missing=\['ROI001'\]"):
            train_unet.validate_training_group_coverage(
                full_names,
                ["ROI000_00_00.jpg"],
            )

    def test_higher_risk_increases_q_and_constant_shift_is_invariant(self) -> None:
        q = {"ROI000": 0.5, "ROI001": 0.5}
        risks = {"ROI000": 0.1, "ROI001": 0.3}
        shifted = {group: risk + 7.0 for group, risk in risks.items()}

        updated = train_unet.update_group_dro_q(q, risks, eta=5.0)
        shifted_updated = train_unet.update_group_dro_q(q, shifted, eta=5.0)

        self.assertGreater(updated["ROI001"], q["ROI001"])
        self.assertLess(updated["ROI000"], q["ROI000"])
        for group in q:
            self.assertAlmostEqual(updated[group], shifted_updated[group], places=14)

    def test_epoch_report_records_sorted_state_and_inverse_simpson(self) -> None:
        p = {"ROI001": 0.75, "ROI000": 0.25}
        q = {"ROI001": 0.5, "ROI000": 0.5}
        risks = {"ROI001": 0.3, "ROI000": 0.1}
        counts = {"ROI001": 3, "ROI000": 1}

        report, q_after = train_unet.group_dro_epoch_report(
            p,
            q,
            risks,
            counts,
            eta=5.0,
        )

        self.assertEqual(list(report["q_before"]), ["ROI000", "ROI001"])
        self.assertEqual(list(report["risks"]), ["ROI000", "ROI001"])
        self.assertEqual(report["q_over_p"], {"ROI000": 2.0, "ROI001": 2 / 3})
        self.assertEqual(report["effective_groups"]["before"], 2.0)
        expected_after = 1.0 / sum(value**2 for value in q_after.values())
        self.assertAlmostEqual(
            report["effective_groups"]["after"],
            expected_after,
        )

    def test_nonfinite_missing_and_invalid_group_state_fail(self) -> None:
        q = {"ROI000": 0.5, "ROI001": 0.5}
        valid_risks = {"ROI000": 0.1, "ROI001": 0.2}
        invalid_updates = (
            ({"ROI000": 1.0}, valid_risks),
            ({"ROI000": 0.0, "ROI001": 1.0}, valid_risks),
            ({"ROI000": 0.4, "ROI001": 0.4}, valid_risks),
            ({"ROI000": float("nan"), "ROI001": 0.5}, valid_risks),
            (q, {"ROI000": 0.1}),
            (q, {"ROI000": 0.1, "ROI001": float("nan")}),
        )
        for invalid_q, risks in invalid_updates:
            with self.subTest(q=invalid_q, risks=risks), self.assertRaises(ValueError):
                train_unet.update_group_dro_q(invalid_q, risks, eta=5.0)

        with self.assertRaisesRegex(ValueError, "float32"):
            train_unet.group_dro_q_over_p(
                {"ROI000": 1e-50, "ROI001": 1.0},
                q,
            )
        with self.assertRaisesRegex(ValueError, "risk counts"):
            train_unet.group_dro_epoch_report(
                q,
                q,
                valid_risks,
                {"ROI000": 1},
                eta=5.0,
            )

    def test_risk_loader_has_isolated_rng_and_disables_augmentation(self) -> None:
        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
                image = torch.zeros(1, 8, 8)
                return image, image, "ROI000_00_00.jpg"

        with mock.patch.object(
            train_unet,
            "PairedPatchDataset",
            return_value=TinyDataset(),
        ) as dataset:
            loader = train_unet.build_group_risk_loader(
                Path("dataset"),
                ["ROI000_00_00.jpg"],
                ("CD68",),
                context_halo=0,
                batch_size=1,
                workers=0,
                pin_memory=False,
                seed=2026,
            )

        dataset.assert_called_once_with(
            Path("dataset"),
            ["ROI000_00_00.jpg"],
            markers=("CD68",),
            augment=False,
            context_halo=0,
            target_context_halo=0,
            target_source_offset=(0, 0),
        )
        torch.manual_seed(53)
        expected = torch.rand(4)
        torch.manual_seed(53)
        next(iter(loader))
        actual = torch.rand(4)
        torch.testing.assert_close(actual, expected)

    def test_risk_pass_uses_eval_float32_and_restores_model_mode(self) -> None:
        class RecordingIdentity(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[bool, torch.dtype]] = []

            def forward(
                self,
                inputs: torch.Tensor,
                condition: torch.Tensor | None = None,
            ) -> torch.Tensor:
                self.calls.append((self.training, inputs.dtype))
                return inputs

        source = torch.zeros(2, 1, 8, 8, dtype=torch.float64)
        target = source.clone()
        target[1].fill_(0.5)
        loader = [
            (
                source,
                target,
                ("ROI000_00_00.jpg", "ROI001_00_00.jpg"),
            )
        ]
        model = RecordingIdentity().train()
        criterion = train_unet.CompetitionLoss(ssim_mode="uniform7")

        risks, counts = train_unet.evaluate_group_risks(
            model,
            loader,
            criterion,
            torch.device("cpu"),
            {"ROI000": 1, "ROI001": 1},
        )

        self.assertEqual(counts, {"ROI000": 1, "ROI001": 1})
        self.assertAlmostEqual(risks["ROI000"], 0.0, places=6)
        self.assertGreater(risks["ROI001"], 0.0)
        self.assertEqual(model.calls, [(False, torch.float32)])
        self.assertTrue(model.training)

        mismatched_loaders = (
            (
                loader,
                {"ROI000": 2, "ROI001": 1},
            ),
            (
                [
                    (
                        source[[0, 0, 1]],
                        target[[0, 0, 1]],
                        (
                            "ROI000_00_00.jpg",
                            "ROI000_00_00.jpg",
                            "ROI001_00_00.jpg",
                        ),
                    )
                ],
                {"ROI000": 1, "ROI001": 1},
            ),
        )
        for mismatched_loader, expected_counts in mismatched_loaders:
            with self.subTest(expected_counts=expected_counts):
                with self.assertRaisesRegex(ValueError, "risk counts do not match"):
                    train_unet.evaluate_group_risks(
                        model,
                        mismatched_loader,
                        criterion,
                        torch.device("cpu"),
                        expected_counts,
                    )


class CheckpointInitializationTests(unittest.TestCase):
    @staticmethod
    def unet_spec(base_channels: int = 8) -> dict[str, object]:
        return train_unet.model_specification(
            "unet", base_channels=base_channels, block_type="residual"
        )

    @staticmethod
    def no_conditioning() -> dict[str, object]:
        return {"version": 1, "mode": "none"}

    def checkpoint(
        self,
        markers: tuple[str, ...],
        *,
        base_channels: int = 8,
        include_modern_metadata: bool = True,
    ) -> tuple[nn.Module, dict[str, object]]:
        model = train_unet.build_model(
            "unet", len(markers), base_channels=base_channels
        )
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                parameter.fill_(index / 100.0)
            for index in range(len(markers)):
                model.head.weight[index].fill_(10.0 + index)
                model.head.bias[index].fill_(20.0 + index)
        checkpoint: dict[str, object] = {
            "model": model.state_dict(),
            "markers": markers,
            "base_channels": base_channels,
            "block_type": "residual",
        }
        if include_modern_metadata:
            checkpoint["model_spec"] = self.unet_spec(base_channels)
            checkpoint["conditioning"] = self.no_conditioning()
        return model, checkpoint

    def test_marker_subset_transfer_reorders_head_and_copies_shared_weights(self) -> None:
        source, checkpoint = self.checkpoint(("HLA-DR", "CD68", "CD45RO"))
        target = train_unet.build_model("unet", 2, base_channels=8)

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD45RO", "CD68"),
            self.unet_spec(),
            self.no_conditioning(),
            marker_subset=True,
        )

        torch.testing.assert_close(target.head.weight[0], source.head.weight[2])
        torch.testing.assert_close(target.head.weight[1], source.head.weight[1])
        torch.testing.assert_close(target.head.bias, source.head.bias[[2, 1]])
        torch.testing.assert_close(
            target.encoders[0].block[0].weight,
            source.encoders[0].block[0].weight,
        )
        self.assertEqual(report["mode"], "marker-subset")
        self.assertEqual(report["target_to_source_indices"], [2, 1])
        self.assertEqual(
            report["head_channel_mapping"],
            [
                {"marker": "CD45RO", "target_index": 0, "source_index": 2},
                {"marker": "CD68", "target_index": 1, "source_index": 1},
            ],
        )
        self.assertEqual(report["sliced_head_tensors"], ["head.weight", "head.bias"])

    def test_subset_transfer_rejects_invalid_or_missing_marker_metadata(self) -> None:
        target = train_unet.build_model("unet", 1, base_channels=8)
        _, valid = self.checkpoint(("CD45RO",))
        cases = [
            ({**valid, "markers": None}, "markers metadata"),
            ({**valid, "markers": ("CD45RO", "CD45RO")}, "duplicate"),
            (valid, "missing from initial checkpoint"),
        ]
        for checkpoint, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                train_unet.initialize_model_from_checkpoint(
                    target,
                    checkpoint,
                    ("CD68",),
                    self.unet_spec(),
                    self.no_conditioning(),
                    marker_subset=True,
                )

    def test_subset_transfer_rejects_non_unet_architecture(self) -> None:
        target = train_unet.build_model("unet", 1, base_channels=8)
        checkpoint = {
            "model": {},
            "markers": ("CD68",),
            "model_spec": train_unet.model_specification("nafnet-width32"),
            "conditioning": self.no_conditioning(),
            "args": {"model_arch": "nafnet-width32"},
        }
        with self.assertRaisesRegex(ValueError, "only for U-Net"):
            train_unet.initialize_model_from_checkpoint(
                target,
                checkpoint,
                ("CD68",),
                self.unet_spec(),
                self.no_conditioning(),
                marker_subset=True,
            )

    def test_strict_mode_remains_default_and_accepts_legacy_unet_metadata(self) -> None:
        source, checkpoint = self.checkpoint(
            ("CD68",), include_modern_metadata=False
        )
        target = train_unet.build_model("unet", 1, base_channels=8)

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            self.no_conditioning(),
        )

        self.assertEqual(report["mode"], "strict")
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], expected)

        _, multi_checkpoint = self.checkpoint(("CD68", "CD45RO"))
        with self.assertRaisesRegex(ValueError, "do not match"):
            train_unet.initialize_model_from_checkpoint(
                target,
                multi_checkpoint,
                ("CD68",),
                self.unet_spec(),
                self.no_conditioning(),
            )

    def test_add_conditioning_copies_shared_weights_and_preserves_output(self) -> None:
        source, checkpoint = self.checkpoint(("CD68",))
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_CONDITION_MODE,
        )
        specification = spatial_conditioning_specification()

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            specification,
            add_conditioning=True,
        )

        self.assertEqual(report["mode"], "add-conditioning")
        self.assertTrue(report["added_tensors"])
        self.assertTrue(
            all(name.startswith("condition_film.") for name in report["added_tensors"])
        )
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
        inputs = torch.rand(2, 1, 32, 32)
        conditions = torch.rand(2, SPATIAL_CONDITION_DIMENSION)
        torch.testing.assert_close(source(inputs), target(inputs, conditions), rtol=0, atol=0)

    def test_add_conditioning_rejects_conditioned_source(self) -> None:
        _, checkpoint = self.checkpoint(("CD68",))
        checkpoint["conditioning"] = spatial_conditioning_specification()
        checkpoint["args"] = {"condition_mode": SPATIAL_CONDITION_MODE}
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_CONDITION_MODE,
        )
        with self.assertRaisesRegex(ValueError, "unconditioned source"):
            train_unet.initialize_model_from_checkpoint(
                target,
                checkpoint,
                ("CD68",),
                self.unet_spec(),
                spatial_conditioning_specification(),
                add_conditioning=True,
            )

    def test_add_thumbnail_conditioning_preserves_spatial_checkpoint(self) -> None:
        source = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_CONDITION_MODE,
        )
        source_specification = spatial_conditioning_specification()
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "base_channels": 8,
            "block_type": "residual",
            "model_spec": self.unet_spec(),
            "conditioning": source_specification,
            "args": {
                "condition_mode": SPATIAL_CONDITION_MODE,
                "conditioning_spec": source_specification,
            },
        }
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        target_specification = spatial_thumbnail_conditioning_specification()

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            target_specification,
            add_conditioning=True,
        )

        self.assertEqual(report["mode"], "add-conditioning")
        self.assertEqual(
            report["transition"],
            "spatial-stats-film-to-spatial-thumbnail-film",
        )
        self.assertTrue(
            all(
                name.startswith("condition_film.thumbnail_")
                for name in report["added_tensors"]
            )
        )
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
        inputs = torch.rand(2, 1, 32, 32)
        spatial_conditions = torch.rand(2, SPATIAL_CONDITION_DIMENSION)
        thumbnail_conditions = torch.rand(
            2,
            SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
        )
        thumbnail_conditions[:, :SPATIAL_CONDITION_DIMENSION] = spatial_conditions
        torch.testing.assert_close(
            source(inputs, spatial_conditions),
            target(inputs, thumbnail_conditions),
            rtol=0,
            atol=0,
        )

    def test_add_multiscale_conditioning_preserves_thumbnail_checkpoint(self) -> None:
        source = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        source_specification = spatial_thumbnail_conditioning_specification()
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "base_channels": 8,
            "block_type": "residual",
            "model_spec": self.unet_spec(),
            "conditioning": source_specification,
            "args": {
                "condition_mode": SPATIAL_THUMBNAIL_CONDITION_MODE,
                "conditioning_spec": source_specification,
            },
        }
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
        )
        target_specification = (
            spatial_thumbnail_multiscale_conditioning_specification()
        )

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            target_specification,
            add_conditioning=True,
        )

        self.assertEqual(
            report["transition"],
            "spatial-thumbnail-film-to-spatial-thumbnail-multiscale-film",
        )
        self.assertTrue(
            all(
                name.startswith("condition_film.decoder_outputs.")
                for name in report["added_tensors"]
            )
        )
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
        inputs = torch.rand(2, 1, 32, 32)
        conditions = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            source(inputs, conditions),
            target(inputs, conditions),
            rtol=0,
            atol=0,
        )

    def test_add_ring_conditioning_preserves_thumbnail_checkpoint(self) -> None:
        source = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        source_specification = spatial_thumbnail_conditioning_specification()
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "base_channels": 8,
            "block_type": "residual",
            "model_spec": self.unet_spec(),
            "conditioning": source_specification,
            "args": {
                "condition_mode": SPATIAL_THUMBNAIL_CONDITION_MODE,
                "conditioning_spec": source_specification,
            },
        }
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
        )
        target_specification = spatial_thumbnail_ring_conditioning_specification()

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            target_specification,
            add_conditioning=True,
        )

        self.assertEqual(
            report["transition"],
            "spatial-thumbnail-film-to-spatial-thumbnail-ring-film",
        )
        self.assertEqual(
            set(report["added_tensors"]),
            {
                "condition_film.ring_projection.weight",
                "condition_film.ring_projection.bias",
            },
        )
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
        inputs = torch.rand(2, 1, 32, 32)
        conditions = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            source(inputs, conditions),
            target(inputs, conditions),
            rtol=0,
            atol=0,
        )

    def test_add_aligned_conditioning_preserves_thumbnail_checkpoint(self) -> None:
        source = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_CONDITION_MODE,
        )
        source_specification = spatial_thumbnail_conditioning_specification()
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "base_channels": 8,
            "block_type": "residual",
            "model_spec": self.unet_spec(),
            "conditioning": source_specification,
            "args": {
                "condition_mode": SPATIAL_THUMBNAIL_CONDITION_MODE,
                "conditioning_spec": source_specification,
            },
        }
        target = train_unet.build_model(
            "unet",
            1,
            base_channels=8,
            condition_mode=SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
        )
        target_specification = spatial_thumbnail_aligned_conditioning_specification()

        report = train_unet.initialize_model_from_checkpoint(
            target,
            checkpoint,
            ("CD68",),
            self.unet_spec(),
            target_specification,
            add_conditioning=True,
        )

        self.assertEqual(
            report["transition"],
            "spatial-thumbnail-film-to-spatial-thumbnail-aligned-fusion",
        )
        self.assertTrue(report["added_tensors"])
        self.assertTrue(
            all(
                name.startswith("condition_film.aligned_")
                for name in report["added_tensors"]
            )
        )
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
        inputs = torch.rand(2, 1, 32, 32)
        conditions = torch.rand(2, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        torch.testing.assert_close(
            source(inputs, conditions),
            target(inputs, conditions),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
