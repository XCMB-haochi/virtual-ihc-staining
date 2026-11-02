from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import (
    CYTOLAND_VSCYTO2D_INITIALIZATION,
    candidate_output_source_offset,
    OFFICIAL_CONVNEXTV2_FCMAE_BASE_INITIALIZATION,
    OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
    OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION,
    evenly_spaced_epochs,
    fcmae_encoder_checkpoint,
    fcmae_pretraining_directory,
    inference_checkpoint_filename,
    initialization_checkpoint,
    load_protocol,
    recipe_inference_context_halo,
    run_directory,
    sourced_run_directory,
    validate_protocol,
)
from evaluate_cv_ensemble import (
    protocol_evaluation_names,
    shift_prediction_edge,
    weighted_prediction,
)
from predict_cv_ensemble import effective_member_weights
from run_dapi_fcmae_pretraining import build_command as build_fcmae_command
from run_cv_training import (
    build_train_command,
    completed_run,
    selected_protocol_folds,
    selected_protocol_recipes,
)


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round26_cv5.json"
ROUND27_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round27_cv5_spatial_film.json"
ROUND29_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round29_cv5_thumbnail_film.json"
ROUND30_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round30_cv5_multiscale_film.json"
ROUND31_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round31_cv5_thumbnail_ring_film.json"
ROUND32_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round32_cv5_thumbnail_ema.json"
ROUND33_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round33_cv5_thumbnail_base48.json"
ROUND34_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round34_cv5_thumbnail_base48_continue.json"
)
ROUND35_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round35_cv5_base32_base48_shrink30.json"
)
ROUND36_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round36_cv5_context128_continue.json"
)
ROUND38_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round38_cv5_aligned_thumbnail_fusion.json"
)
ROUND39_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round39_h1h2_restormer_screen.json"
)
ROUND41_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round41_cv5_official_checkpoint_selection.json"
)
ROUND42_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round42_cv5_mixstyle_domain_generalization.json"
)
ROUND44_PROMOTION_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round44_cv5_unext2_fcmae_promotion.json"
)
ROUND48_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round48_cv2_unext2_whole_roi.json"
)
ROUND49_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round49_cv2_unext2_widehead.json"
)
ROUND50_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round50_cv2_unext2_paired_affine.json"
)
ROUND51_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round51_cv2_unext2_auxiliary_markers.json"
)
ROUND52_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round52_cv2_unext2_official_fcmae.json"
)
ROUND52_PROMOTION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round52_cv5_unext2_official_fcmae_promotion.json"
)
ROUND53_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round53_cv5_official_fcmae_weight_surface.json"
)
ROUND54_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round54_cv2_unext2_supervised22k_fcmae.json"
)
ROUND55_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round55_cv2_unext2_official_auxiliary.json"
)
ROUND55_PROMOTION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round55_cv5_unext2_official_auxiliary_promotion.json"
)
ROUND56_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round56_cv2_unext2_auxiliary_warm_primary.json"
)
ROUND57_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round57_cv2_unext2_primary_warm_auxiliary.json"
)
ROUND58_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round58_cv2_unext2_official_fcmae1200.json"
)
ROUND59_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round59_fold0_label_scale.json"
)
ROUND60_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round60_cv2_unext2_marker_decoders.json"
)
ROUND61_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round61_cv2_unext2_decoder_only_auxiliary.json"
)
ROUND62_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round62_cv2_unext2_base.json"
)
ROUND63_PROMOTION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round63_cv5_unext2_base_diversity_promotion.json"
)
ROUND64_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round64_cv5_auxiliary_weight_calibration.json"
)
ROUND65_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round65_cv2_unext2_all_dapi_fcmae.json"
)
ROUND66_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round66_cv5_all_dapi_diversity_promotion.json"
)
ROUND67_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round67_cv5_auxiliary_all_dapi_composition.json"
)
ROUND70_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round70_cv5_base_composition_interpolation.json"
)
ROUND73_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round73_cv2_unext2_encoder_lr.json"
)
ROUND74_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round74_cv2_unext2_drop_path.json"
)
ROUND75_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round75_cv2_unext2_cytoland_fcmae.json"
)
ROUND76_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round76_cv2_unext2_cytoland_direct.json"
)
ROUND77_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round77_cv2_cytoland_encoder_composition.json"
)
ROUND78_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round78_cv2_unext2_target_l1.json"
)
ROUND79_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round79_cv2_base_auxiliary.json"
)
ROUND80_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round80_cv2_fixed_output_shift.json"
)
ROUND81_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round81_cv5_fixed_output_shift.json"
)
ROUND83_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round83_cv2_aligned_target_training.json"
)
ROUND84_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "round84_cv2_official_fcmae_independent_seed.json"
)
ROUND85_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round85_cv5_context128_inference.json"
)
ROUND86_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round86_cv2_context128_training.json"
)
ROUND87_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round87_cv2_context128_fulltarget.json"
)
ROUND97_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "round97_cv5_context256_inference.json"
)


class CVProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_round26_protocol_is_valid_and_covers_rois_once(self) -> None:
        parsed = load_protocol(PROTOCOL_PATH)
        rois = [roi for fold in parsed["folds"] for roi in fold["val_rois"]]
        self.assertEqual(len(rois), 25)
        self.assertEqual(set(rois), set(parsed["expected_rois"]))

    def test_duplicate_fold_roi_is_rejected(self) -> None:
        duplicate = copy.deepcopy(self.protocol)
        duplicate["folds"][1]["val_rois"][0] = duplicate["folds"][0]["val_rois"][0]
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            validate_protocol(duplicate)

    def test_candidate_weights_must_sum_to_one(self) -> None:
        invalid = copy.deepcopy(self.protocol)
        invalid["candidates"]["base32_fulltarget_50"]["base32"] = 0.6
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_protocol(invalid)

    def test_candidate_output_source_offsets_are_validated_and_applied(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        candidate = protocol["primary_candidate"]
        protocol["candidate_output_source_offsets"] = {candidate: [0, -1]}
        parsed = validate_protocol(protocol)
        self.assertEqual(candidate_output_source_offset(parsed, candidate), (0, -1))
        self.assertEqual(
            candidate_output_source_offset(parsed, parsed["baseline_candidate"]),
            (0, 0),
        )

        image = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        shifted = shift_prediction_edge(image, (0, -1))
        np.testing.assert_array_equal(
            shifted,
            np.array([[[0, 0, 1], [3, 3, 4], [6, 6, 7]]], dtype=np.float32),
        )
        self.assertIs(shift_prediction_edge(image, (0, 0)), image)

        for offset in ([0], [0, 1, 2], [0, 9], [0, 1.0], "0,-1"):
            invalid = copy.deepcopy(protocol)
            invalid["candidate_output_source_offsets"][candidate] = offset
            with self.subTest(offset=offset):
                with self.assertRaisesRegex(ValueError, "source offset"):
                    validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["candidate_output_source_offsets"] = {"unknown": [0, -1]}
        with self.assertRaisesRegex(ValueError, "unknown candidates"):
            validate_protocol(invalid)

    def test_ema_decay_must_be_numeric_and_less_than_one(self) -> None:
        invalid_type = copy.deepcopy(self.protocol)
        invalid_type["recipes"]["base32"]["ema_decay"] = "invalid"
        with self.assertRaisesRegex(ValueError, "ema_decay must be numeric"):
            validate_protocol(invalid_type)

        invalid_range = copy.deepcopy(self.protocol)
        invalid_range["recipes"]["base32"]["ema_decay"] = 1.0
        with self.assertRaisesRegex(ValueError, r"ema_decay must be in \[0, 1\)"):
            validate_protocol(invalid_range)

    def test_recipe_model_fields_are_validated(self) -> None:
        invalid_arch = copy.deepcopy(self.protocol)
        invalid_arch["recipes"]["base32"]["model_arch"] = "unknown"
        with self.assertRaisesRegex(ValueError, "invalid model_arch"):
            validate_protocol(invalid_arch)

        invalid_width = copy.deepcopy(self.protocol)
        invalid_width["recipes"]["base32"]["base_channels"] = 0
        with self.assertRaisesRegex(ValueError, "invalid base_channels"):
            validate_protocol(invalid_width)

        invalid_block = copy.deepcopy(self.protocol)
        invalid_block["recipes"]["base32"]["block_type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "invalid block_type"):
            validate_protocol(invalid_block)

    def test_recipe_training_subset_and_epoch_override_are_validated(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        recipe = protocol["recipes"]["fulltarget"]
        recipe["epochs_per_stage"] = 19
        recipe["train_rois_by_fold"] = {
            fold["name"]: [
                protocol["folds"][(index + 1) % len(protocol["folds"])][
                    "val_rois"
                ][0]
            ]
            for index, fold in enumerate(protocol["folds"])
        }
        parsed = validate_protocol(protocol)
        train_rois = parsed["recipes"]["fulltarget"]["train_rois_by_fold"][
            "fold0"
        ]
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            parsed["folds"][0]["val_rois"],
            "CD68",
            parsed["recipes"]["fulltarget"],
            parsed["training"],
            1,
            None,
            "cuda:0",
            train_rois,
        )
        self.assertEqual(command[command.index("--epochs") + 1], "19")
        self.assertEqual(
            command[command.index("--train-rois") + 1],
            ",".join(train_rois),
        )

        invalid_epochs = copy.deepcopy(protocol)
        invalid_epochs["recipes"]["fulltarget"]["epochs_per_stage"] = 0
        with self.assertRaisesRegex(ValueError, "epochs_per_stage"):
            validate_protocol(invalid_epochs)

        missing_fold = copy.deepcopy(protocol)
        del missing_fold["recipes"]["fulltarget"]["train_rois_by_fold"]["fold4"]
        with self.assertRaisesRegex(ValueError, "define every fold"):
            validate_protocol(missing_fold)

        overlapping = copy.deepcopy(protocol)
        overlapping["recipes"]["fulltarget"]["train_rois_by_fold"]["fold0"] = [
            overlapping["folds"][0]["val_rois"][0]
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_protocol(overlapping)

    def test_evenly_spaced_validation_epochs_are_validated(self) -> None:
        self.assertEqual(evenly_spaced_epochs(14, 14), list(range(1, 15)))
        self.assertEqual(evenly_spaced_epochs(28, 14), list(range(2, 29, 2)))
        self.assertEqual(evenly_spaced_epochs(56, 14), list(range(4, 57, 4)))
        self.assertEqual(len(set(evenly_spaced_epochs(19, 14))), 14)
        with self.assertRaisesRegex(ValueError, "evaluation_count"):
            evenly_spaced_epochs(14, 15)

        invalid = copy.deepcopy(self.protocol)
        invalid["recipes"]["fulltarget"]["validation_evaluations"] = (
            invalid["training"]["epochs_per_stage"] + 1
        )
        with self.assertRaisesRegex(ValueError, "validation_evaluations"):
            validate_protocol(invalid)

    def test_paired_affine_requires_complete_safe_geometry(self) -> None:
        valid = copy.deepcopy(self.protocol)
        valid["recipes"]["base32"]["paired_affine_augmentation"] = {
            "probability": 0.8,
            "outer_halo": 256,
            "degrees": 45.0,
            "scale_min": 0.72,
            "scale_max": 1.3,
        }
        validate_protocol(valid)

        unsafe = copy.deepcopy(valid)
        unsafe["recipes"]["base32"]["paired_affine_augmentation"][
            "outer_halo"
        ] = 64
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            validate_protocol(unsafe)

        incomplete = copy.deepcopy(valid)
        del incomplete["recipes"]["base32"]["paired_affine_augmentation"][
            "scale_max"
        ]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_protocol(incomplete)

    def test_recipe_training_markers_and_loss_weights_are_validated(self) -> None:
        valid = copy.deepcopy(self.protocol)
        valid["recipes"]["fulltarget"]["training_markers"] = [
            "CD68",
            "CD45RO",
        ]
        valid["recipes"]["fulltarget"]["marker_loss_weights"] = [0.9, 0.1]
        validate_protocol(valid)

        missing_primary = copy.deepcopy(valid)
        missing_primary["recipes"]["fulltarget"]["training_markers"] = [
            "CD45RO",
            "HLA-DR",
        ]
        with self.assertRaisesRegex(ValueError, "protocol marker"):
            validate_protocol(missing_primary)

        wrong_length = copy.deepcopy(valid)
        wrong_length["recipes"]["fulltarget"]["marker_loss_weights"] = [1.0]
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_protocol(wrong_length)

        wrong_sum = copy.deepcopy(valid)
        wrong_sum["recipes"]["fulltarget"]["marker_loss_weights"] = [0.8, 0.1]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_protocol(wrong_sum)

    def test_recipe_gradient_accumulation_preserves_effective_batch(self) -> None:
        protocol = json.loads(ROUND39_PROTOCOL_PATH.read_text(encoding="utf-8"))
        parsed = validate_protocol(copy.deepcopy(protocol))
        recipe = parsed["recipes"]["restormersmall"]
        self.assertEqual(recipe["batch_size"], 4)
        self.assertEqual(recipe["gradient_accumulation_steps"], 2)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["restormersmall"]["gradient_accumulation_steps"] = 1
        with self.assertRaisesRegex(ValueError, "effective batch"):
            validate_protocol(invalid)

    def test_roi_scope_is_explicitly_validated(self) -> None:
        protocol = json.loads(ROUND39_PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(validate_protocol(protocol)["roi_scope"], "subset")

        invalid = copy.deepcopy(protocol)
        invalid["roi_scope"] = "partial"
        with self.assertRaisesRegex(ValueError, "roi_scope"):
            validate_protocol(invalid)

    def test_subset_roi_scope_filters_oof_universe_and_rejects_missing_rois(self) -> None:
        names = [
            "ROI000_00_00.jpg",
            "ROI001_00_00.jpg",
            "ROI002_00_00.jpg",
        ]
        protocol = {"expected_rois": ["ROI000", "ROI002"], "roi_scope": "subset"}
        selected, actual_rois = protocol_evaluation_names(names, protocol)
        self.assertEqual(selected, [names[0], names[2]])
        self.assertEqual(actual_rois, {"ROI000", "ROI001", "ROI002"})

        missing = {"expected_rois": ["ROI000", "ROI003"], "roi_scope": "subset"}
        with self.assertRaisesRegex(ValueError, "missing ROIs"):
            protocol_evaluation_names(names, missing)

    def test_complete_roi_scope_still_rejects_extra_dataset_rois(self) -> None:
        names = ["ROI000_00_00.jpg", "ROI001_00_00.jpg"]
        protocol = {"expected_rois": ["ROI000"]}
        with self.assertRaisesRegex(ValueError, "does not match"):
            protocol_evaluation_names(names, protocol)

    def test_recipe_inference_halo_defaults_and_overrides(self) -> None:
        default_protocol = validate_protocol(copy.deepcopy(self.protocol))
        self.assertEqual(
            recipe_inference_context_halo(default_protocol, "base32"),
            default_protocol["inference"]["context_halo"],
        )

        mixed = copy.deepcopy(self.protocol)
        mixed["inference"]["context_halo"] = 128
        mixed["recipes"]["base32"]["inference_context_halo"] = 64
        parsed = validate_protocol(mixed)
        self.assertEqual(recipe_inference_context_halo(parsed, "base32"), 64)
        with self.assertRaisesRegex(ValueError, "Unknown recipe"):
            recipe_inference_context_halo(parsed, "missing")

    def test_inference_checkpoint_filename_is_explicit_and_validated(self) -> None:
        legacy = load_protocol(ROUND35_PROTOCOL_PATH)
        self.assertEqual(
            inference_checkpoint_filename(legacy),
            "best_global_mse.pt",
        )

        official = load_protocol(ROUND41_PROTOCOL_PATH)
        self.assertEqual(
            inference_checkpoint_filename(official),
            "best_mean_image.pt",
        )
        self.assertFalse(official["inference"]["test_like_diagnostic"])

        invalid = copy.deepcopy(legacy)
        invalid["inference"]["checkpoint_filename"] = "../epoch_001.pt"
        with self.assertRaisesRegex(ValueError, "checkpoint_filename"):
            validate_protocol(invalid)

        invalid = copy.deepcopy(legacy)
        invalid["inference"]["test_like_diagnostic"] = "false"
        with self.assertRaisesRegex(ValueError, "test_like_diagnostic"):
            validate_protocol(invalid)

    def test_recipe_inference_halo_is_bounded_by_loader_halo(self) -> None:
        for invalid_value in (-1, 257, True, 1.5):
            with self.subTest(invalid_value=invalid_value):
                invalid = copy.deepcopy(self.protocol)
                invalid["recipes"]["base32"]["inference_context_halo"] = invalid_value
                with self.assertRaisesRegex(ValueError, "inference_context_halo"):
                    validate_protocol(invalid)

        exceeds_loader = copy.deepcopy(self.protocol)
        exceeds_loader["recipes"]["base32"]["inference_context_halo"] = 128
        with self.assertRaisesRegex(ValueError, "exceeds the loader halo"):
            validate_protocol(exceeds_loader)

        for invalid_loader_halo in (257, True):
            with self.subTest(invalid_loader_halo=invalid_loader_halo):
                invalid = copy.deepcopy(self.protocol)
                invalid["inference"]["context_halo"] = invalid_loader_halo
                with self.assertRaisesRegex(ValueError, "inference.context_halo"):
                    validate_protocol(invalid)

    def test_run_directory_is_stable(self) -> None:
        path = run_directory(Path("runs"), "round26_cv5", "fold2", "fulltarget", 2)
        self.assertEqual(path, Path("runs/round26_cv5_fold2_fulltarget_stage2"))

    def test_weighted_prediction(self) -> None:
        predictions = {
            "a": np.ones((2, 3, 3), dtype=np.float32),
            "b": np.full((2, 3, 3), 3.0, dtype=np.float32),
        }
        result = weighted_prediction(predictions, {"a": 0.25, "b": 0.75})
        np.testing.assert_array_equal(result, np.full((2, 3, 3), 2.5, dtype=np.float32))

    def test_primary_candidate_has_ten_equal_members(self) -> None:
        weights = effective_member_weights(
            self.protocol, self.protocol["primary_candidate"]
        )
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.1) < 1e-12 for weight in weights.values()))

    def test_round59_preregisters_nested_update_matched_scale_study(self) -> None:
        protocol = load_protocol(ROUND59_PROTOCOL_PATH)
        study = protocol["label_scale_study"]
        self.assertEqual(study["initial_fold"], "fold0")
        self.assertEqual(study["baseline_recipe"], "full20")
        self.assertEqual(set(study["recipes"]), set(protocol["recipes"]))
        self.assertEqual(protocol["recipes"]["full20"]["train_stages"], [])
        self.assertEqual(study["selection_evaluations_per_stage"], 14)
        self.assertFalse(protocol["training"]["save_every_epoch"])

        for fold in ("fold0", "fold2"):
            baseline_updates = study["recipes"]["full20"][
                "expected_optimizer_updates_per_stage"
            ][fold]
            for draw in ("a", "b", "c"):
                nested = []
                for scale in (5, 10, 15):
                    recipe_name = f"{draw}{scale}"
                    recipe = protocol["recipes"][recipe_name]
                    self.assertEqual(recipe["validation_evaluations"], 14)
                    train_rois = set(recipe["train_rois_by_fold"][fold])
                    self.assertEqual(len(train_rois), scale)
                    nested.append(train_rois)
                    updates = study["recipes"][recipe_name][
                        "expected_optimizer_updates_per_stage"
                    ][fold]
                    self.assertLessEqual(
                        abs(updates - baseline_updates) / baseline_updates,
                        0.04,
                    )
                self.assertLess(nested[0], nested[1])
                self.assertLess(nested[1], nested[2])

        recipe = protocol["recipes"]["a5"]
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][0]["val_rois"],
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
            recipe["train_rois_by_fold"]["fold0"],
        )
        self.assertEqual(command[command.index("--epochs") + 1], "56")
        self.assertEqual(
            command[command.index("--train-rois") + 1],
            ",".join(recipe["train_rois_by_fold"]["fold0"]),
        )
        self.assertEqual(
            command[command.index("--validation-epochs") + 1],
            ",".join(str(epoch) for epoch in range(4, 57, 4)),
        )

    def test_completed_run_requires_exact_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "best_global_mse.pt").write_bytes(b"checkpoint")
            (output / "metrics.jsonl").write_text(
                '{"epoch": 1}\n{"epoch": 2}\n', encoding="utf-8"
            )
            self.assertTrue(completed_run(output, 2))
            self.assertFalse(completed_run(output, 3))

    def test_fold_subset_is_validated_and_keeps_protocol_order(self) -> None:
        protocol = load_protocol(ROUND34_PROTOCOL_PATH)
        selected = selected_protocol_folds(protocol, "fold4,fold0,fold2")
        self.assertEqual(
            [fold["name"] for fold in selected],
            ["fold0", "fold2", "fold4"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selected_protocol_folds(protocol, "fold0,fold0")
        with self.assertRaisesRegex(ValueError, "unknown"):
            selected_protocol_folds(protocol, "fold5")
        with self.assertRaisesRegex(ValueError, "at least one"):
            selected_protocol_folds(protocol, ",")

    def test_recipe_subset_is_validated_and_keeps_protocol_order(self) -> None:
        protocol = load_protocol(ROUND55_PROTOCOL_PATH)
        selected = selected_protocol_recipes(
            protocol,
            "officialauxiliary15,officialauxcontrol",
        )
        self.assertEqual(
            [name for name, _ in selected],
            ["officialauxcontrol", "officialauxiliary15"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selected_protocol_recipes(
                protocol,
                "officialauxcontrol,officialauxcontrol",
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            selected_protocol_recipes(protocol, "missing")
        with self.assertRaisesRegex(ValueError, "at least one"):
            selected_protocol_recipes(protocol, ",")

    def test_stage2_command_uses_stage1_checkpoint_and_full_target_halo(self) -> None:
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            ["ROI000", "ROI001"],
            "CD68",
            self.protocol["recipes"]["fulltarget"],
            self.protocol["training"],
            2,
            Path("runs/stage1/best_global_mse.pt"),
            "cuda:0",
        )
        self.assertEqual(command[command.index("--context-halo") + 1], "64")
        self.assertEqual(command[command.index("--train-target-context-halo") + 1], "64")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0001")
        self.assertEqual(
            command[command.index("--init-checkpoint") + 1],
            str(Path("runs/stage1/best_global_mse.pt")),
        )

    def test_round39_restormer_command_uses_fixed_long_budget(self) -> None:
        protocol = load_protocol(ROUND39_PROTOCOL_PATH)
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][0]["val_rois"],
            "CD68",
            protocol["recipes"]["restormersmall"],
            protocol["training"],
            1,
            None,
            "cuda:0",
        )

        self.assertEqual(command[command.index("--model-arch") + 1], "restormer-small")
        self.assertEqual(command[command.index("--batch-size") + 1], "4")
        self.assertEqual(
            command[command.index("--gradient-accumulation-steps") + 1],
            "2",
        )
        self.assertEqual(
            command[command.index("--minimum-learning-rate") + 1],
            "1e-06",
        )
        self.assertEqual(command[command.index("--gradient-clip-norm") + 1], "1.0")
        self.assertNotIn("--init-checkpoint", command)
        self.assertNotIn("--save-every-epoch", command)

        weights = effective_member_weights(
            protocol,
            protocol["primary_candidate"],
        )
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(abs(weight - 0.25) < 1e-12 for weight in weights.values()))

    def test_round27_reuses_control_and_trains_only_conditioned_stage2(self) -> None:
        protocol = load_protocol(ROUND27_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(
            protocol["recipes"]["fulltarget"]["train_stages"],
            [],
        )
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold3", "fulltarget"),
            Path("runs/round26_cv5_fold3_fulltarget_stage2"),
        )
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold3", "spatialfilm"),
            Path("runs/round27_cv5_spatial_film_fold3_spatialfilm_stage2"),
        )
        init_checkpoint = initialization_checkpoint(
            output_root,
            protocol,
            "fold3",
            "spatialfilm",
            2,
        )
        self.assertEqual(
            init_checkpoint,
            Path("runs/round26_cv5_fold3_fulltarget_stage1/best_global_mse.pt"),
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][3]["val_rois"],
            "CD68",
            protocol["recipes"]["spatialfilm"],
            protocol["training"],
            2,
            init_checkpoint,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--condition-mode") + 1],
            "spatial-stats-film",
        )
        self.assertIn("--init-add-conditioning", command)

    def test_round27_primary_candidate_has_ten_equal_members(self) -> None:
        protocol = load_protocol(ROUND27_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.1) < 1e-12 for weight in weights.values()))

    def test_round29_has_matched_continuation_and_thumbnail_runs(self) -> None:
        protocol = load_protocol(ROUND29_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(protocol["recipes"]["spatialbase"]["train_stages"], [])
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold2", "spatialbase"),
            Path("runs/round27_cv5_spatial_film_fold2_spatialfilm_stage2"),
        )
        for recipe_name in ("spatialcontinue", "thumbnailfilm"):
            self.assertEqual(
                initialization_checkpoint(
                    output_root,
                    protocol,
                    "fold2",
                    recipe_name,
                    2,
                ),
                Path(
                    "runs/round27_cv5_spatial_film_fold2_"
                    "spatialfilm_stage2/best_global_mse.pt"
                ),
            )

        thumbnail_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailfilm"],
            protocol["training"],
            2,
            Path(
                "runs/round27_cv5_spatial_film_fold2_"
                "spatialfilm_stage2/best_global_mse.pt"
            ),
            "cuda:0",
        )
        self.assertEqual(
            thumbnail_command[thumbnail_command.index("--condition-mode") + 1],
            "spatial-thumbnail-film",
        )
        self.assertIn("--init-add-conditioning", thumbnail_command)

        control_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-control"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            protocol["recipes"]["spatialcontinue"],
            protocol["training"],
            2,
            Path(
                "runs/round27_cv5_spatial_film_fold2_"
                "spatialfilm_stage2/best_global_mse.pt"
            ),
            "cuda:0",
        )
        self.assertEqual(
            control_command[control_command.index("--condition-mode") + 1],
            "spatial-stats-film",
        )
        self.assertNotIn("--init-add-conditioning", control_command)

    def test_round29_primary_candidate_has_five_equal_members(self) -> None:
        protocol = load_protocol(ROUND29_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.2) < 1e-12 for weight in weights.values()))

    def test_round30_has_matched_continuation_and_multiscale_runs(self) -> None:
        protocol = load_protocol(ROUND30_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(protocol["recipes"]["thumbnailbase"]["train_stages"], [])
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold2", "thumbnailbase"),
            Path(
                "runs/round29_cv5_thumbnail_film_fold2_"
                "thumbnailfilm_stage2"
            ),
        )
        source_checkpoint = Path(
            "runs/round29_cv5_thumbnail_film_fold2_"
            "thumbnailfilm_stage2/best_global_mse.pt"
        )
        for recipe_name in ("thumbnailcontinue", "multiscalefilm"):
            self.assertEqual(
                initialization_checkpoint(
                    output_root,
                    protocol,
                    "fold2",
                    recipe_name,
                    2,
                ),
                source_checkpoint,
            )

        multiscale_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            protocol["recipes"]["multiscalefilm"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertEqual(
            multiscale_command[multiscale_command.index("--condition-mode") + 1],
            "spatial-thumbnail-multiscale-film",
        )
        self.assertIn("--init-add-conditioning", multiscale_command)

        control_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-control"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailcontinue"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertEqual(
            control_command[control_command.index("--condition-mode") + 1],
            "spatial-thumbnail-film",
        )
        self.assertNotIn("--init-add-conditioning", control_command)

    def test_round30_primary_candidate_has_five_equal_members(self) -> None:
        protocol = load_protocol(ROUND30_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.2) < 1e-12 for weight in weights.values()))

    def test_round31_has_matched_continuation_and_ring_runs(self) -> None:
        protocol = load_protocol(ROUND31_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(protocol["recipes"]["thumbnailbase"]["train_stages"], [])
        source_checkpoint = Path(
            "runs/round30_cv5_multiscale_film_fold3_"
            "thumbnailcontinue_stage2/best_global_mse.pt"
        )
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold3", "thumbnailbase"),
            source_checkpoint.parent,
        )
        for recipe_name in ("thumbnailcontinue", "ringfilm"):
            self.assertEqual(
                initialization_checkpoint(
                    output_root,
                    protocol,
                    "fold3",
                    recipe_name,
                    2,
                ),
                source_checkpoint,
            )

        ring_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-ring"),
            protocol["folds"][3]["val_rois"],
            "CD68",
            protocol["recipes"]["ringfilm"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertEqual(
            ring_command[ring_command.index("--condition-mode") + 1],
            "spatial-thumbnail-ring-film",
        )
        self.assertIn("--init-add-conditioning", ring_command)

        control_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-control"),
            protocol["folds"][3]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailcontinue"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertEqual(
            control_command[control_command.index("--condition-mode") + 1],
            "spatial-thumbnail-film",
        )
        self.assertNotIn("--init-add-conditioning", control_command)

    def test_round31_primary_candidate_has_five_equal_members(self) -> None:
        protocol = load_protocol(ROUND31_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.2) < 1e-12 for weight in weights.values()))

    def test_round32_has_matched_online_and_ema_continuations(self) -> None:
        protocol = load_protocol(ROUND32_PROTOCOL_PATH)
        output_root = Path("runs")
        source_checkpoint = Path(
            "runs/round31_cv5_thumbnail_ring_film_fold1_"
            "thumbnailcontinue_stage2/best_global_mse.pt"
        )
        self.assertEqual(protocol["recipes"]["thumbnailbase"]["train_stages"], [])
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold1", "thumbnailbase"),
            source_checkpoint.parent,
        )
        for recipe_name in ("thumbnailcontinue", "thumbnailema"):
            self.assertEqual(
                initialization_checkpoint(
                    output_root,
                    protocol,
                    "fold1",
                    recipe_name,
                    2,
                ),
                source_checkpoint,
            )

        control_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-control"),
            protocol["folds"][1]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailcontinue"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        ema_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-ema"),
            protocol["folds"][1]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailema"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertNotIn("--ema-decay", control_command)
        self.assertEqual(ema_command[ema_command.index("--ema-decay") + 1], "0.999")
        for command in (control_command, ema_command):
            self.assertEqual(
                command[command.index("--condition-mode") + 1],
                "spatial-thumbnail-film",
            )
            self.assertEqual(
                command[command.index("--init-checkpoint") + 1],
                str(source_checkpoint),
            )
            self.assertNotIn("--init-add-conditioning", command)

    def test_round32_primary_candidate_has_five_equal_members(self) -> None:
        protocol = load_protocol(ROUND32_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.2) < 1e-12 for weight in weights.values()))

    def test_round33_trains_base48_in_two_stages_and_reuses_base32(self) -> None:
        protocol = load_protocol(ROUND33_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(protocol["recipes"]["thumbnailbase32"]["train_stages"], [])
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold4", "thumbnailbase32"),
            Path(
                "runs/round31_cv5_thumbnail_ring_film_fold4_"
                "thumbnailcontinue_stage2"
            ),
        )
        self.assertIsNone(
            initialization_checkpoint(
                output_root,
                protocol,
                "fold4",
                "thumbnailbase48",
                1,
            )
        )
        stage1_checkpoint = Path(
            "runs/round33_cv5_thumbnail_base48_fold4_"
            "thumbnailbase48_stage1/best_global_mse.pt"
        )
        self.assertEqual(
            initialization_checkpoint(
                output_root,
                protocol,
                "fold4",
                "thumbnailbase48",
                2,
            ),
            stage1_checkpoint,
        )

        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-base48"),
            protocol["folds"][4]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailbase48"],
            protocol["training"],
            2,
            stage1_checkpoint,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--model-arch") + 1], "unet")
        self.assertEqual(command[command.index("--base-channels") + 1], "48")
        self.assertEqual(command[command.index("--block-type") + 1], "residual")
        self.assertEqual(
            command[command.index("--condition-mode") + 1],
            "spatial-thumbnail-film",
        )
        self.assertEqual(
            command[command.index("--init-checkpoint") + 1],
            str(stage1_checkpoint),
        )

    def test_round33_primary_candidate_has_ten_equal_members(self) -> None:
        protocol = load_protocol(ROUND33_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.1) < 1e-12 for weight in weights.values()))

    def test_round34_continues_same_fold_base48_and_reuses_base32(self) -> None:
        protocol = load_protocol(ROUND34_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertEqual(protocol["recipes"]["thumbnailbase32"]["train_stages"], [])
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold2", "thumbnailbase32"),
            Path(
                "runs/round31_cv5_thumbnail_ring_film_fold2_"
                "thumbnailcontinue_stage2"
            ),
        )
        source_checkpoint = Path(
            "runs/round33_cv5_thumbnail_base48_fold2_"
            "thumbnailbase48_stage2/best_global_mse.pt"
        )
        self.assertEqual(
            initialization_checkpoint(
                output_root,
                protocol,
                "fold2",
                "thumbnailbase48continue",
                2,
            ),
            source_checkpoint,
        )

        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-base48-continue"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            protocol["recipes"]["thumbnailbase48continue"],
            protocol["training"],
            2,
            source_checkpoint,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--base-channels") + 1], "48")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0001")
        self.assertEqual(
            command[command.index("--condition-mode") + 1],
            "spatial-thumbnail-film",
        )
        self.assertEqual(
            command[command.index("--init-checkpoint") + 1],
            str(source_checkpoint),
        )

    def test_round34_primary_candidate_has_ten_equal_members(self) -> None:
        protocol = load_protocol(ROUND34_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(abs(weight - 0.1) < 1e-12 for weight in weights.values()))

    def test_round35_reuses_fixed_base32_and_continued_base48(self) -> None:
        protocol = load_protocol(ROUND35_PROTOCOL_PATH)
        output_root = Path("runs")
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            sourced_run_directory(output_root, protocol, "fold0", "thumbnailbase32"),
            Path(
                "runs/round31_cv5_thumbnail_ring_film_fold0_"
                "thumbnailcontinue_stage2"
            ),
        )
        self.assertEqual(
            sourced_run_directory(
                output_root,
                protocol,
                "fold0",
                "thumbnailbase48continue",
            ),
            Path(
                "runs/round34_cv5_thumbnail_base48_continue_fold0_"
                "thumbnailbase48continue_stage2"
            ),
        )

    def test_round35_primary_candidate_has_fixed_effective_weights(self) -> None:
        protocol = load_protocol(ROUND35_PROTOCOL_PATH)
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        for member, weight in weights.items():
            expected = 0.14 if member.endswith("/thumbnailbase32") else 0.06
            self.assertAlmostEqual(weight, expected)

    def test_round36_changes_only_context_for_matched_continuation(self) -> None:
        protocol = load_protocol(ROUND36_PROTOCOL_PATH)
        control = protocol["recipes"]["context64control"]
        candidate = protocol["recipes"]["context128continue"]
        self.assertEqual(protocol["training"]["seed"], 2032)
        self.assertEqual(control["train_stages"], [])
        self.assertEqual(candidate["train_stages"], [2])
        self.assertEqual(control["context_halo"], 64)
        self.assertEqual(candidate["context_halo"], 128)
        self.assertEqual(control["train_target_context_halo"], 64)
        self.assertEqual(candidate["train_target_context_halo"], 64)
        self.assertEqual(
            recipe_inference_context_halo(protocol, "context64control"), 64
        )
        self.assertEqual(
            recipe_inference_context_halo(protocol, "context128continue"), 128
        )
        self.assertEqual(
            candidate["init_source"],
            {
                "protocol_name": "round31_cv5_thumbnail_ring_film",
                "recipe_name": "thumbnailcontinue",
                "stage": 2,
            },
        )
        self.assertEqual(
            sourced_run_directory(Path("runs"), protocol, "fold0", "context64control"),
            Path("runs/round32_cv5_thumbnail_ema_fold0_" "thumbnailcontinue_stage2"),
        )

    def test_round36_primary_candidate_is_fixed_half_context128(self) -> None:
        protocol = load_protocol(ROUND36_PROTOCOL_PATH)
        self.assertEqual(
            protocol["candidates"]["round35_context128_50"],
            {
                "round35base32": 0.35,
                "round35base48": 0.15,
                "context128continue": 0.5,
            },
        )
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 15)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round38_trains_only_the_aligned_branch_from_round31(self) -> None:
        protocol = load_protocol(ROUND38_PROTOCOL_PATH)
        candidate = protocol["recipes"]["alignedcontext"]
        self.assertEqual(candidate["train_stages"], [2])
        self.assertEqual(candidate["condition_mode"], "spatial-thumbnail-aligned-fusion")
        self.assertTrue(candidate["init_add_conditioning"])
        self.assertEqual(
            initialization_checkpoint(
                Path("runs"),
                protocol,
                "fold2",
                "alignedcontext",
                2,
            ),
            Path(
                "runs/round31_cv5_thumbnail_ring_film_fold2_"
                "thumbnailcontinue_stage2/best_global_mse.pt"
            ),
        )
        self.assertEqual(
            sourced_run_directory(
                Path("runs"), protocol, "fold2", "context64control"
            ),
            Path(
                "runs/round32_cv5_thumbnail_ema_fold2_"
                "thumbnailcontinue_stage2"
            ),
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-aligned"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            candidate,
            protocol["training"],
            2,
            Path("runs/source/best_global_mse.pt"),
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--condition-mode") + 1],
            "spatial-thumbnail-aligned-fusion",
        )
        self.assertIn("--init-add-conditioning", command)

    def test_round38_gate_schema_rejects_unknown_dependencies(self) -> None:
        protocol = json.loads(ROUND38_PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["gates"]["primary"]["requires"] = "missing"
        with self.assertRaisesRegex(ValueError, "requires unknown gate"):
            validate_protocol(protocol)

    def test_round42_mixstyle_is_matched_and_explicitly_orchestrated(self) -> None:
        protocol = load_protocol(ROUND42_PROTOCOL_PATH)
        recipe = protocol["recipes"]["mixstylecontinue"]
        source = Path(
            "runs/round31_cv5_thumbnail_ring_film_fold2_"
            "thumbnailcontinue_stage2/best_global_mse.pt"
        )
        self.assertEqual(recipe["train_stages"], [2])
        self.assertEqual(recipe["mixstyle_probability"], 0.5)
        self.assertEqual(recipe["mixstyle_alpha"], 0.1)
        self.assertEqual(
            initialization_checkpoint(
                Path("runs"),
                protocol,
                "fold2",
                "mixstylecontinue",
                2,
            ),
            source,
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output-mixstyle"),
            protocol["folds"][2]["val_rois"],
            "CD68",
            recipe,
            protocol["training"],
            2,
            source,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--mixstyle-probability") + 1],
            "0.5",
        )
        self.assertEqual(command[command.index("--mixstyle-alpha") + 1], "0.1")
        self.assertEqual(command[command.index("--seed") + 1], "2032")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0001")
        self.assertFalse(protocol["inference"]["test_like_diagnostic"])

    def test_round42_mixstyle_recipe_and_gate_ranges_are_validated(self) -> None:
        protocol = json.loads(ROUND42_PROTOCOL_PATH.read_text(encoding="utf-8"))
        for field, value, message in (
            ("mixstyle_probability", 1.1, "probability"),
            ("mixstyle_alpha", 0.0, "alpha"),
        ):
            invalid = copy.deepcopy(protocol)
            invalid["recipes"]["mixstylecontinue"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, message
            ):
                validate_protocol(invalid)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["mixstylecontinue"]["model_arch"] = "restormer-small"
        with self.assertRaisesRegex(ValueError, "non-U-Net"):
            validate_protocol(invalid)

        invalid = copy.deepcopy(protocol)
        invalid["distribution_gates"]["primary"]["minimum_positive_rois"] = 26
        with self.assertRaisesRegex(ValueError, "minimum_positive_rois"):
            validate_protocol(invalid)

    def test_round42_primary_candidate_has_fixed_fifteen_members(self) -> None:
        protocol = load_protocol(ROUND42_PROTOCOL_PATH)
        self.assertEqual(
            protocol["candidates"]["round35_mixstyle35"],
            {
                "round31base32": 0.35,
                "mixstylecontinue": 0.35,
                "round34base48": 0.3,
            },
        )
        weights = effective_member_weights(protocol, protocol["primary_candidate"])
        self.assertEqual(len(weights), 15)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round44_promotion_reuses_screen_namespace_and_fixed_blend(self) -> None:
        protocol = load_protocol(ROUND44_PROMOTION_PROTOCOL_PATH)
        self.assertEqual(protocol["name"], "round44_cv2_unext2_fcmae")
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(
            [
                fold["name"]
                for fold in selected_protocol_folds(
                    protocol, "fold1,fold3,fold4"
                )
            ],
            ["fold1", "fold3", "fold4"],
        )
        self.assertEqual(
            protocol["run_sources"]["unext2fcmae"]["protocol_name"],
            "round44_cv2_unext2_fcmae",
        )
        self.assertEqual(
            protocol["candidates"]["round35_fcmae20"],
            {
                "round35base32": 0.56,
                "round35base48": 0.24,
                "unext2fcmae": 0.2,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["five_fold_promotion"],
            {
                "candidate": "round35_fcmae20",
                "baseline": "round35",
                "minimum_q95_gain": 0.08,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 17,
                "minimum_roi_q95_gain": -0.08,
            },
        )

    def test_round48_is_fresh_fcmae800_whole_roi_screen_with_fixed_weights(self) -> None:
        protocol = load_protocol(ROUND48_PROTOCOL_PATH)
        recipe = protocol["recipes"]["wholeroi"]
        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(
            recipe["condition_mode"],
            "spatial-thumbnail-whole-roi-film",
        )
        self.assertEqual(
            protocol["candidates"]["round47_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "wholeroi": 0.30,
            },
        )
        self.assertNotIn(
            "requires",
            protocol["distribution_gates"]["whole_roi_diversity"],
        )
        self.assertEqual(
            sourced_run_directory(Path("runs"), protocol, "fold2", "wholeroi"),
            Path(
                "runs/round48_cv2_unext2_whole_roi_fold2_"
                "wholeroi_stage2"
            ),
        )
        weights = effective_member_weights(protocol, "round47_split30")
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round49_is_fresh_matched_wide_head_screen(self) -> None:
        protocol = load_protocol(ROUND49_PROTOCOL_PATH)
        recipe = protocol["recipes"]["widehead"]
        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny-widehead")
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(recipe["condition_mode"], "spatial-thumbnail-film")
        self.assertEqual(
            protocol["candidates"]["round47_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "widehead": 0.30,
            },
        )
        self.assertNotIn(
            "requires",
            protocol["distribution_gates"]["widehead_diversity"],
        )
        self.assertEqual(
            sourced_run_directory(Path("runs"), protocol, "fold2", "widehead"),
            Path("runs/round49_cv2_unext2_widehead_fold2_widehead_stage2"),
        )
        weights = effective_member_weights(protocol, "round47_split30")
        self.assertEqual(len(weights), 8)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round50_is_fresh_matched_paired_affine_screen(self) -> None:
        protocol = load_protocol(ROUND50_PROTOCOL_PATH)
        recipe = protocol["recipes"]["pairedaffine"]
        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(
            recipe["paired_affine_augmentation"],
            {
                "probability": 0.8,
                "outer_halo": 256,
                "degrees": 45.0,
                "scale_min": 0.72,
                "scale_max": 1.3,
            },
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/candidate"),
            protocol["folds"][0]["val_rois"],
            protocol["marker"],
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--paired-affine-probability") + 1],
            "0.8",
        )
        self.assertEqual(
            protocol["candidates"]["round47_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "pairedaffine": 0.30,
            },
        )
        self.assertNotIn(
            "requires",
            protocol["distribution_gates"]["paired_affine_diversity"],
        )

    def test_round51_is_matched_low_weight_auxiliary_marker_screen(self) -> None:
        protocol = load_protocol(ROUND51_PROTOCOL_PATH)
        control = protocol["recipes"]["auxcontrol"]
        candidate = protocol["recipes"]["auxiliary15"]
        expected_markers = ["CD68", "CD45RO", "HLA-DR", "Vimentin"]
        self.assertEqual(control["training_markers"], expected_markers)
        self.assertEqual(candidate["training_markers"], expected_markers)
        self.assertEqual(control["marker_loss_weights"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(candidate["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])
        self.assertEqual(control["model_arch"], candidate["model_arch"])
        self.assertEqual(control["train_stages"], [1, 2])
        self.assertEqual(candidate["train_stages"], [1, 2])
        self.assertTrue(control["fcmae_encoder_init"])
        self.assertTrue(candidate["fcmae_encoder_init"])

        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/candidate"),
            protocol["folds"][0]["val_rois"],
            protocol["marker"],
            candidate,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--markers") + 1],
            "CD68,CD45RO,HLA-DR,Vimentin",
        )
        self.assertEqual(
            command[command.index("--marker-loss-weights") + 1],
            "0.85,0.05,0.05,0.05",
        )
        self.assertEqual(
            protocol["candidates"]["round47_auxiliary_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "auxiliary15": 0.30,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["auxiliary_diversity"]["requires"],
            "auxiliary_diversity_effect",
        )

    def test_round52_uses_audited_official_fcmae_initialization(self) -> None:
        protocol = load_protocol(ROUND52_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]
        self.assertEqual(
            pretraining["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        self.assertEqual(pretraining["epochs"], 800)
        self.assertEqual(
            protocol["recipes"]["officialfcmae800"]["train_stages"],
            [1, 2],
        )
        command = build_fcmae_command(
            Path("dataset"),
            Path("runs/candidate"),
            protocol["folds"][0]["val_rois"],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--encoder-init-mode") + 1],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION["mode"],
        )
        self.assertEqual(
            command[command.index("--encoder-init-sha256") + 1],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION["sha256"],
        )
        self.assertEqual(
            protocol["candidates"]["round47_official_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "officialfcmae800": 0.30,
            },
        )

        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["encoder_initialization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audited official"):
            validate_protocol(invalid)

    def test_round52_promotion_reuses_screen_namespace_and_fixed_split(self) -> None:
        protocol = load_protocol(ROUND52_PROMOTION_PROTOCOL_PATH)
        self.assertEqual(protocol["name"], "round52_cv2_unext2_official_fcmae")
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(
            [
                fold["name"]
                for fold in selected_protocol_folds(
                    protocol,
                    "fold1,fold3,fold4",
                )
            ],
            ["fold1", "fold3", "fold4"],
        )
        self.assertEqual(
            protocol["candidates"]["round47_official_split30"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "officialfcmae800": 0.30,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["official_diversity_full"],
            {
                "candidate": "round47_official_split30",
                "baseline": "round47w060",
                "minimum_q95_gain": 0.06,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 20,
                "minimum_roi_q95_gain": -0.08,
            },
        )
        self.assertEqual(
            sourced_run_directory(
                Path("runs"),
                protocol,
                "fold0",
                "officialfcmae800",
            ),
            Path(
                "runs/round52_cv2_unext2_official_fcmae_fold0_"
                "officialfcmae800_stage2"
            ),
        )
        weights = effective_member_weights(
            protocol,
            "round47_official_split30",
        )
        self.assertEqual(len(weights), 20)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round53_preregisters_bounded_official_fcmae_weight_surface(self) -> None:
        protocol = load_protocol(ROUND53_PROTOCOL_PATH)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            set(protocol["candidates"]),
            {
                f"t{total}_o{official_share}"
                for total in (50, 60, 70)
                for official_share in (33, 50, 67)
            },
        )
        self.assertEqual(
            protocol["candidates"]["t60_o50"],
            {
                "round35base32": 0.28,
                "round35base48": 0.12,
                "fcmae800": 0.30,
                "officialfcmae800": 0.30,
            },
        )
        self.assertEqual(protocol["baseline_candidate"], "t60_o50")
        self.assertEqual(protocol["primary_candidate"], "t60_o67")
        self.assertEqual(
            protocol["distribution_promotion_order"][0],
            "t60_o67_full",
        )
        self.assertEqual(len(protocol["distribution_gates"]), 8)
        for candidate, weights in protocol["candidates"].items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, msg=candidate)
        effective = effective_member_weights(protocol, "t60_o67")
        self.assertEqual(len(effective), 20)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

    def test_round54_uses_audited_supervised_22k_initialization(self) -> None:
        protocol = load_protocol(ROUND54_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]
        self.assertEqual(
            pretraining["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION,
        )
        self.assertEqual(pretraining["epochs"], 800)
        self.assertEqual(
            protocol["recipes"]["supervised22kfcmae800"]["train_stages"],
            [1, 2],
        )
        command = build_fcmae_command(
            Path("dataset"),
            Path("runs/candidate"),
            protocol["folds"][0]["val_rois"],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--encoder-init-mode") + 1],
            OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION["mode"],
        )
        self.assertEqual(
            command[command.index("--encoder-init-sha256") + 1],
            OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION["sha256"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_o50"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.35,
            },
        )
        self.assertEqual(
            set(protocol["candidates"]["supervised22k_equal_thirds"]),
            {
                "round35base32",
                "round35base48",
                "fcmae800",
                "officialfcmae800",
                "supervised22kfcmae800",
            },
        )
        self.assertAlmostEqual(
            sum(protocol["candidates"]["supervised22k_equal_thirds"].values()),
            1.0,
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["supervised22k_replacement", "supervised22k_diversity"],
        )

        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["encoder_initialization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audited official"):
            validate_protocol(invalid)

    def test_round55_combines_official_fcmae_with_matched_auxiliary_control(
        self,
    ) -> None:
        protocol = load_protocol(ROUND55_PROTOCOL_PATH)
        self.assertEqual(
            protocol["fcmae_pretraining"]["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        control = protocol["recipes"]["officialauxcontrol"]
        candidate = protocol["recipes"]["officialauxiliary15"]
        expected_markers = ["CD68", "CD45RO", "HLA-DR", "Vimentin"]
        self.assertEqual(control["training_markers"], expected_markers)
        self.assertEqual(candidate["training_markers"], expected_markers)
        self.assertEqual(control["marker_loss_weights"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            candidate["marker_loss_weights"],
            [0.85, 0.05, 0.05, 0.05],
        )
        self.assertEqual(control["train_stages"], [1, 2])
        self.assertEqual(candidate["train_stages"], [1, 2])
        self.assertTrue(control["fcmae_encoder_init"])
        self.assertTrue(candidate["fcmae_encoder_init"])
        self.assertEqual(
            protocol["candidates"]["t70_o50"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.35,
            },
        )
        self.assertEqual(
            protocol["candidates"]["t70_official_auxiliary_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialauxiliary15": 0.175,
            },
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["official_auxiliary_replacement", "official_auxiliary_diversity"],
        )
        effective = effective_member_weights(
            protocol,
            "t70_official_auxiliary_split",
        )
        self.assertEqual(len(effective), 10)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

    def test_round55_promotion_reuses_namespace_and_fixed_auxiliary_split(
        self,
    ) -> None:
        screen = load_protocol(ROUND55_PROTOCOL_PATH)
        protocol = load_protocol(ROUND55_PROMOTION_PROTOCOL_PATH)
        self.assertEqual(protocol["name"], screen["name"])
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(protocol["training"], screen["training"])
        for recipe_name in ("officialauxcontrol", "officialauxiliary15"):
            self.assertEqual(
                protocol["recipes"][recipe_name],
                screen["recipes"][recipe_name],
            )
        self.assertEqual(
            [
                fold["name"]
                for fold in selected_protocol_folds(
                    protocol,
                    "fold1,fold3,fold4",
                )
            ],
            ["fold1", "fold3", "fold4"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_official_auxiliary_split"],
            screen["candidates"]["t70_official_auxiliary_split"],
        )
        self.assertEqual(
            protocol["distribution_gates"][
                "official_auxiliary_diversity_full"
            ],
            {
                "candidate": "t70_official_auxiliary_split",
                "baseline": "t70_o50",
                "minimum_q95_gain": 0.04,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 20,
                "minimum_roi_q95_gain": -0.08,
                "requires": "official_auxiliary_diversity_effect_full",
            },
        )
        weights = effective_member_weights(
            protocol,
            "t70_official_auxiliary_split",
        )
        self.assertEqual(len(weights), 25)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round60_is_a_fresh_matched_marker_decoder_screen(self) -> None:
        protocol = load_protocol(ROUND60_PROTOCOL_PATH)
        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(
            protocol["fcmae_pretraining"]["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        control = protocol["recipes"]["markerdecodercontrol"]
        candidate = protocol["recipes"]["markerdecoderauxiliary15"]
        for recipe in (control, candidate):
            self.assertEqual(
                recipe["model_arch"],
                "unext2-fcmae-tiny-marker-decoders",
            )
            self.assertEqual(recipe["train_stages"], [1, 2])
            self.assertTrue(recipe["fcmae_encoder_init"])
            self.assertEqual(recipe["batch_size"], 8)
            self.assertEqual(recipe["gradient_accumulation_steps"], 2)
            self.assertEqual(
                recipe["training_markers"],
                ["CD68", "CD45RO", "HLA-DR", "Vimentin"],
            )
        self.assertEqual(control["marker_loss_weights"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            candidate["marker_loss_weights"],
            [0.85, 0.05, 0.05, 0.05],
        )
        self.assertEqual(
            protocol["baseline_candidate"],
            "t70_official_auxiliary_split",
        )
        self.assertEqual(
            protocol["primary_candidate"],
            "t70_markerdecoder_auxiliary_split",
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["decoder_isolation_deployment", "decoder_isolation_replacement"],
        )
        self.assertEqual(
            protocol["distribution_gates"]["isolated_auxiliary_mechanism"][
                "baseline"
            ],
            "markerdecodercontrol",
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][0]["val_rois"],
            "CD68",
            candidate,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--model-arch") + 1],
            "unext2-fcmae-tiny-marker-decoders",
        )
        self.assertEqual(
            command[command.index("--marker-loss-weights") + 1],
            "0.85,0.05,0.05,0.05",
        )
        self.assertEqual(command[command.index("--batch-size") + 1], "8")
        self.assertEqual(
            command[command.index("--gradient-accumulation-steps") + 1],
            "2",
        )
        effective = effective_member_weights(
            protocol,
            "t70_markerdecoder_auxiliary_split",
        )
        self.assertEqual(len(effective), 10)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

    def test_round61_routes_auxiliary_gradients_only_through_shared_decoder(
        self,
    ) -> None:
        round55 = load_protocol(ROUND55_PROTOCOL_PATH)
        protocol = load_protocol(ROUND61_PROTOCOL_PATH)
        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(protocol["training"], round55["training"])
        self.assertEqual(
            protocol["fcmae_pretraining"]["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        candidate = protocol["recipes"]["decoderonlyauxiliary15"]
        self.assertEqual(candidate["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(candidate["train_stages"], [1, 2])
        self.assertTrue(candidate["fcmae_encoder_init"])
        self.assertEqual(
            candidate["training_markers"],
            ["CD68", "CD45RO", "HLA-DR", "Vimentin"],
        )
        self.assertEqual(
            candidate["marker_loss_weights"],
            [0.85, 0.05, 0.05, 0.05],
        )
        self.assertEqual(candidate["auxiliary_gradient_scope"], "decoder-only")
        trained = [
            name
            for name, recipe in protocol["recipes"].items()
            if recipe["train_stages"]
        ]
        self.assertEqual(trained, ["decoderonlyauxiliary15"])
        self.assertEqual(
            protocol["run_sources"]["officialauxcontrol"],
            {
                "protocol_name": "round55_cv2_unext2_official_auxiliary",
                "recipe_name": "officialauxcontrol",
                "stage": 2,
            },
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][0]["val_rois"],
            "CD68",
            candidate,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--auxiliary-gradient-scope") + 1],
            "decoder-only",
        )
        self.assertEqual(command[command.index("--batch-size") + 1], "16")
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["decoder_only_deployment", "decoder_only_replacement"],
        )
        effective = effective_member_weights(
            protocol,
            "t70_decoderonly_auxiliary_split",
        )
        self.assertEqual(len(effective), 10)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["decoderonlyauxiliary15"][
            "auxiliary_gradient_scope"
        ] = "encoder-only"
        with self.assertRaisesRegex(ValueError, "invalid auxiliary_gradient_scope"):
            validate_protocol(invalid)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["decoderonlyauxiliary15"]["training_markers"] = [
            "CD45RO",
            "CD68",
            "HLA-DR",
            "Vimentin",
        ]
        with self.assertRaisesRegex(ValueError, "protocol marker first"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round61_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("pid0=$!"), 1)
        self.assertEqual(launcher.count("pid1=$!"), 1)
        self.assertIn('wait "$pid0"', launcher)
        self.assertIn('wait "$pid1"', launcher)

    def test_round62_preregisters_fresh_base_capacity_screen(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        protocol = load_protocol(ROUND62_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]
        candidate = protocol["recipes"]["officialbasefcmae800"]

        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(protocol["training"], round52["training"])
        self.assertEqual(pretraining["epochs"], 800)
        self.assertEqual(pretraining["model_size"], "base")
        self.assertEqual(pretraining["batch_size"], 16)
        self.assertEqual(pretraining["gradient_accumulation_steps"], 2)
        self.assertEqual(
            pretraining["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_BASE_INITIALIZATION,
        )
        self.assertEqual(candidate["model_arch"], "unext2-fcmae-base")
        self.assertEqual(candidate["batch_size"], 8)
        self.assertEqual(candidate["gradient_accumulation_steps"], 2)
        self.assertEqual(candidate["train_stages"], [1, 2])

        pretrain_command = build_fcmae_command(
            Path("dataset"),
            Path("runs/fresh-round62"),
            protocol["folds"][0]["val_rois"],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(
            pretrain_command[pretrain_command.index("--model-size") + 1],
            "base",
        )
        self.assertEqual(
            pretrain_command[
                pretrain_command.index("--gradient-accumulation-steps") + 1
            ],
            "2",
        )
        self.assertNotIn("--resume", pretrain_command)

        train_command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/output"),
            protocol["folds"][0]["val_rois"],
            "CD68",
            candidate,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            train_command[train_command.index("--model-arch") + 1],
            "unext2-fcmae-base",
        )
        self.assertEqual(train_command[train_command.index("--batch-size") + 1], "8")
        self.assertEqual(
            train_command[
                train_command.index("--gradient-accumulation-steps") + 1
            ],
            "2",
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["base_replacement", "base_diversity"],
        )
        effective = effective_member_weights(protocol, "t70_base_split")
        self.assertEqual(len(effective), 10)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["model_size"] = "tiny"
        invalid["fcmae_pretraining"][
            "encoder_initialization"
        ] = OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION
        with self.assertRaisesRegex(ValueError, "model_size disagrees"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round62_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(launcher.count("pid0=$!"), 1)
        self.assertEqual(launcher.count("pid1=$!"), 1)
        self.assertIn('wait "$pid0"', launcher)
        self.assertIn('wait "$pid1"', launcher)

    def test_round63_promotes_fixed_base_diversity_to_five_folds(self) -> None:
        screen = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND63_PROMOTION_PROTOCOL_PATH)

        self.assertEqual(protocol["name"], screen["name"])
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(protocol["training"], screen["training"])
        self.assertEqual(
            protocol["fcmae_pretraining"],
            screen["fcmae_pretraining"],
        )
        self.assertEqual(
            [
                fold["name"]
                for fold in selected_protocol_folds(
                    protocol,
                    "fold1,fold3,fold4",
                )
            ],
            ["fold1", "fold3", "fold4"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            screen["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["distribution_gates"]["base_diversity_full"],
            {
                "candidate": "t70_base_split",
                "baseline": "t70_o50",
                "minimum_q95_gain": 0.04,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 20,
                "minimum_roi_q95_gain": -0.08,
            },
        )
        self.assertEqual(
            sourced_run_directory(
                Path("runs"),
                protocol,
                "fold0",
                "officialbasefcmae800",
            ),
            Path(
                "runs/round62_cv2_unext2_base_fold0_"
                "officialbasefcmae800_stage2"
            ),
        )
        effective = effective_member_weights(protocol, "t70_base_split")
        self.assertEqual(len(effective), 25)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round63_cv5_promotion_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            launcher,
        )
        self.assertIn('run_fold fold1 "$gpu0"', launcher)
        self.assertIn('run_fold fold3 "$gpu1"', launcher)
        self.assertIn('run_fold fold4 "$free_gpu"', launcher)
        self.assertIn("wait -n -p completed_pid", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

    def test_round64_calibrates_one_fixed_auxiliary_weight(self) -> None:
        round55 = load_protocol(ROUND55_PROMOTION_PROTOCOL_PATH)
        protocol = load_protocol(ROUND64_PROTOCOL_PATH)

        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(protocol["training"], round55["training"])
        self.assertEqual(
            protocol["candidates"]["t70_auxiliary_half"],
            round55["candidates"]["t70_official_auxiliary_split"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_auxiliary_two_thirds"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.1166666666666667,
                "officialauxiliary15": 0.2333333333333333,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["auxiliary_two_thirds_deployment"],
            {
                "candidate": "t70_auxiliary_two_thirds",
                "baseline": "t70_o50",
                "minimum_q95_gain": 0.04,
                "require_positive_float": True,
                "minimum_positive_folds": 5,
                "minimum_positive_rois": 18,
                "minimum_roi_q95_gain": -0.05,
                "requires": "auxiliary_two_thirds_advance",
            },
        )
        self.assertEqual(
            sourced_run_directory(
                Path("runs"),
                protocol,
                "fold4",
                "officialauxiliary15",
            ),
            Path(
                "runs/round55_cv2_unext2_official_auxiliary_fold4_"
                "officialauxiliary15_stage2"
            ),
        )
        effective = effective_member_weights(
            protocol,
            "t70_auxiliary_two_thirds",
        )
        self.assertEqual(len(effective), 25)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

        launcher = (
            PROJECT_ROOT
            / "tools"
            / "run_round64_auxiliary_weight_calibration.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "0f2b52e0a5cd0fa34df5ee597019b7ce7ae53b86cfae1593d387a60a68772056",
            launcher,
        )
        self.assertNotIn("run_cv_training.py", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

        submission_launcher = (
            PROJECT_ROOT / "tools" / "run_round71_base_submission_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "e3e854ecf193e95fe0b1d9b721a87361e98c3ac3b0c11684f302d22f18eec372",
            submission_launcher,
        )
        self.assertIn(
            "7cd5de3e89be461134056072279efe848a85bde7b98c781945883ab44ffc6bad",
            submission_launcher,
        )
        self.assertIn("t70_base_split", submission_launcher)
        self.assertIn("predict_cv_ensemble.py", submission_launcher)
        self.assertIn("package_submission.py", submission_launcher)
        self.assertIn('report.get("test_images") != 1346', submission_launcher)
        self.assertIn('len(report.get("members", [])) != 25', submission_launcher)

    def test_round65_shares_update_matched_all_train_dapi_encoder(self) -> None:
        protocol = load_protocol(ROUND65_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]
        self.assertEqual(pretraining["data_scope"], "all-train")
        self.assertTrue(pretraining["shared_across_folds"])
        self.assertEqual(pretraining["epochs"], 630)
        self.assertEqual(pretraining["warmup_epochs"], 8)
        self.assertEqual(
            fcmae_pretraining_directory(Path("runs"), protocol, "fold0"),
            fcmae_pretraining_directory(Path("runs"), protocol, "fold2"),
        )
        self.assertEqual(
            fcmae_pretraining_directory(Path("runs"), protocol, "fold0"),
            Path(
                "runs/round65_cv2_unext2_all_dapi_fcmae_shared_"
                "dapi_alltrain_officialfcmae630_pretraining"
            ),
        )
        command = build_fcmae_command(
            Path("dataset"),
            Path("runs/candidate"),
            [],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--data-scope") + 1], "all-train")
        self.assertEqual(command[command.index("--val-rois") + 1], "")
        self.assertEqual(
            protocol["candidates"]["t70_all_dapi_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "alldapiofficialfcmae630": 0.175,
            },
        )
        self.assertEqual(
            fcmae_encoder_checkpoint(
                Path("runs"),
                protocol,
                "fold2",
            ),
            fcmae_pretraining_directory(Path("runs"), protocol, "fold0")
            / "final_encoder.pt",
        )
        launcher = (PROJECT_ROOT / "tools" / "run_round65_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("a10a20a521a2eab8a4462d79ec346e620187531b9e5eee33bfbb677cbf7645e9", launcher)
        self.assertIn("--folds fold0,fold2", launcher)
        self.assertIn("gpu_snapshot", launcher)
        self.assertIn("ROUND65_SERIAL_GPU", launcher)
        self.assertIn("minimum_free_memory_mb", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["data_scope"] = "fold-train"
        with self.assertRaisesRegex(ValueError, "Shared FCMAE"):
            validate_protocol(invalid)

    def test_round66_expands_fixed_all_dapi_diversity_to_complete_oof(self) -> None:
        screen = load_protocol(ROUND65_PROTOCOL_PATH)
        protocol = load_protocol(ROUND66_PROTOCOL_PATH)

        self.assertEqual(protocol["name"], screen["name"])
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(protocol["roi_scope"], "complete")
        self.assertEqual(protocol["training"], screen["training"])
        self.assertEqual(protocol["inference"], screen["inference"])
        self.assertEqual(
            protocol["fcmae_pretraining"],
            screen["fcmae_pretraining"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_all_dapi_split"],
            screen["candidates"]["t70_all_dapi_split"],
        )
        self.assertEqual(
            protocol["distribution_gates"]["all_dapi_diversity_full"],
            {
                "candidate": "t70_all_dapi_split",
                "baseline": "t70_o50",
                "minimum_q95_gain": 0.04,
                "require_positive_float": True,
                "minimum_positive_folds": 4,
                "minimum_positive_rois": 20,
                "minimum_roi_q95_gain": -0.08,
            },
        )
        self.assertEqual(
            fcmae_encoder_checkpoint(Path("runs"), protocol, "fold4"),
            Path(
                "runs/round65_cv2_unext2_all_dapi_fcmae_shared_"
                "dapi_alltrain_officialfcmae630_pretraining/final_encoder.pt"
            ),
        )
        self.assertEqual(
            sourced_run_directory(
                Path("runs"),
                protocol,
                "fold2",
                "alldapiofficialfcmae630",
            ),
            Path(
                "runs/round65_cv2_unext2_all_dapi_fcmae_fold2_"
                "alldapiofficialfcmae630_stage2"
            ),
        )
        effective = effective_member_weights(protocol, "t70_all_dapi_split")
        self.assertEqual(len(effective), 25)
        self.assertAlmostEqual(sum(effective.values()), 1.0)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round66_cv5_promotion_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "7db9be9bf5d34fb4dfb4fb0bf7a8d80a5d5fc7954de9be2caf15994e8d457daf",
            "56a91690be0438569499f903ff0797ad233131c1af28b5c78c3e063477453316",
            "e0a137d30383555ed930bf7ab03f27f101c76f82f3bba9a879bc6777b032a8ed",
            "f4a8c93c616310ce4a981f9b1fe6830ba82bf78ff91bdbe52d471cf1c0164554",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("pending=(fold1 fold3 fold4)", launcher)
        self.assertIn("wait -n -p completed_pid", launcher)
        self.assertIn("minimum_free_memory_mb", launcher)
        self.assertIn("ROUND66_PREFLIGHT_ONLY", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

    def test_round67_composes_two_independent_complete_oof_endpoints(self) -> None:
        protocol = load_protocol(ROUND67_PROTOCOL_PATH)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            protocol["candidates"]["tri_aux23_residual_half"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0583333333333333,
                "officialauxiliary15": 0.2333333333333333,
                "alldapiofficialfcmae630": 0.0583333333333334,
            },
        )
        for gate in protocol["distribution_gates"].values():
            self.assertEqual(gate["minimum_q95_gain"], 0.05)
            self.assertEqual(gate["minimum_positive_folds"], 5)
            self.assertEqual(gate["minimum_positive_rois"], 20)
            self.assertEqual(gate["minimum_roi_q95_gain"], -0.05)
        for candidate in protocol["candidates"]:
            effective = effective_member_weights(protocol, candidate)
            self.assertAlmostEqual(sum(effective.values()), 1.0)
        launcher = (
            PROJECT_ROOT / "tools" / "run_round67_composition_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "e0a137d30383555ed930bf7ab03f27f101c76f82f3bba9a879bc6777b032a8ed",
            launcher,
        )
        self.assertIn(
            "4eee928e2f034fd0b76151359196eb36705b13606cd9b79f3688fd37db243e03",
            launcher,
        )
        self.assertNotIn("run_cv_training.py", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

        submission_launcher = (
            PROJECT_ROOT / "tools" / "run_round68_submission_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2",
            submission_launcher,
        )
        self.assertIn(
            "cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc",
            submission_launcher,
        )
        self.assertIn("tri_auxhalf_residual_half", submission_launcher)
        self.assertIn("predict_cv_ensemble.py", submission_launcher)
        self.assertIn("package_submission.py", submission_launcher)
        self.assertIn('report.get("test_images") != 1346', submission_launcher)
        self.assertIn('len(report.get("members", [])) != 30', submission_launcher)

        base_resume = (
            PROJECT_ROOT / "tools" / "run_round69_base_resume_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "821cd877c82e415c09e159f2d4b9c020b755c92f96e069b4a4fb23a431223a56",
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            "cd861f525cebeac50a84070a8eba33ac2dbc3f2f67f5adaead9b790cf2bd4146",
            "07346f1d343d1a14c1585014ed69200d137e15856925e07028d0fa62b476b6ce",
            "f7ff4aacbaeaf0c278c063423b7b50bb74b3d9c8c3909288add42b50e332c2f8",
            "538cb4fb3a1e8a61b03ae570bca1f4eb44104aa5844595b8fb9ca7f20758e80c",
        ):
            self.assertIn(digest, base_resume)
        self.assertIn("ROUND69_PREFLIGHT_ONLY", base_resume)
        self.assertIn("pending=(fold1 fold3 fold4)", base_resume)
        self.assertIn("kill -0", base_resume)
        self.assertIn("poll_seconds", base_resume)
        self.assertIn("trap write_exit_status EXIT", base_resume)

    def test_round70_interpolates_only_two_promoted_complete_oof_endpoints(self) -> None:
        protocol = load_protocol(ROUND70_PROTOCOL_PATH)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            protocol["candidates"]["blend_base50"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.13125,
                "officialauxiliary15": 0.0875,
                "alldapiofficialfcmae630": 0.04375,
                "officialbasefcmae800": 0.0875,
            },
        )
        for candidate in protocol["candidates"]:
            self.assertAlmostEqual(
                sum(effective_member_weights(protocol, candidate).values()),
                1.0,
            )
        for suffix in ("50", "67", "33"):
            advance = protocol["distribution_gates"][f"blend{suffix}_advance"]
            deployment = protocol["distribution_gates"][f"blend{suffix}_deployment"]
            self.assertEqual(advance["baseline"], "t70_base_split")
            self.assertEqual(advance["minimum_q95_gain"], 0.005)
            self.assertEqual(deployment["baseline"], "t70_o50")
            self.assertEqual(deployment["minimum_q95_gain"], 0.085)
            self.assertEqual(deployment["requires"], f"blend{suffix}_advance")
        self.assertEqual(
            protocol["distribution_promotion_order"],
            [
                "blend50_deployment",
                "blend67_deployment",
                "blend33_deployment",
                "base_retained",
            ],
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round70_base_composition_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2",
            "cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc",
            "cf3b9af9ae5b2da3757107bcec5b8d1aa6b013d0e2181369df6cc9ee9e2b9c53",
            "fb5994f347e440ddf69d4c16f821ef09286264bb2d92e72c4c0e68d5112f9d77",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND70_PREFLIGHT_ONLY", launcher)
        self.assertNotIn("run_cv_training.py", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

    def test_round73_uses_fixed_tenth_rate_fcmae_encoder(self) -> None:
        control = load_protocol(ROUND52_PROTOCOL_PATH)
        protocol = load_protocol(ROUND73_PROTOCOL_PATH)
        recipe = protocol["recipes"]["officialfcmaellrd10"]

        self.assertEqual(protocol["folds"], control["folds"])
        self.assertEqual(protocol["training"], control["training"])
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(recipe["encoder_learning_rate_multiplier"], 0.1)
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(
            protocol["candidates"]["t70_llrd_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialfcmaellrd10": 0.175,
            },
        )
        stage1 = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/stage1"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        stage2 = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/stage2"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            2,
            Path("runs/stage1/best_global_mse.pt"),
            "cuda:0",
        )
        self.assertEqual(
            stage1[stage1.index("--encoder-learning-rate") + 1],
            "2.9999999999999997e-05",
        )
        self.assertEqual(
            stage2[stage2.index("--encoder-learning-rate") + 1],
            "1e-05",
        )
        self.assertEqual(
            fcmae_pretraining_directory(Path("runs"), protocol, "fold0"),
            Path(
                "runs/round73_cv2_unext2_encoder_lr_fold0_"
                "dapi_officialfcmae800_llrd_pretraining"
            ),
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round73_encoder_lr_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9",
            "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND73_PREFLIGHT_ONLY", launcher)
        self.assertIn("round73_encoder_lr_smoke", launcher)
        self.assertIn("decoder / encoder - 10.0", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)
        self.assertIn("trap write_exit_status EXIT", launcher)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["officialfcmaellrd10"][
            "encoder_learning_rate_multiplier"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "must be in"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["officialfcmaellrd10"]["model_arch"] = "unet"
        with self.assertRaisesRegex(ValueError, "requires ConvNeXt"):
            validate_protocol(invalid)

    def test_round74_uses_official_tiny_encoder_drop_path_rate(self) -> None:
        control = load_protocol(ROUND52_PROTOCOL_PATH)
        protocol = load_protocol(ROUND74_PROTOCOL_PATH)
        recipe = protocol["recipes"]["officialfcmaedroppath20"]

        self.assertEqual(protocol["folds"], control["folds"])
        self.assertEqual(protocol["training"], control["training"])
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(recipe["encoder_drop_path_rate"], 0.2)
        self.assertEqual(recipe["train_stages"], [1, 2])
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/stage1"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--encoder-drop-path-rate") + 1],
            "0.2",
        )
        self.assertNotIn("--encoder-learning-rate", command)
        self.assertEqual(
            protocol["candidates"]["t70_droppath_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialfcmaedroppath20": 0.175,
            },
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round74_drop_path_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9",
            "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND74_PREFLIGHT_ONLY", launcher)
        self.assertIn("round74_drop_path_smoke", launcher)
        self.assertIn("encoder_drop_path_rates", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["officialfcmaedroppath20"]["encoder_drop_path_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "must be in"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["officialfcmaedroppath20"]["model_arch"] = "unet"
        with self.assertRaisesRegex(ValueError, "requires UNeXt2"):
            validate_protocol(invalid)

    def test_round75_uses_audited_cytoland_prior_with_matched_training(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        round62 = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND75_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]

        self.assertEqual(protocol["folds"], round52["folds"])
        self.assertEqual(protocol["training"], round52["training"])
        self.assertEqual(
            pretraining["encoder_initialization"],
            CYTOLAND_VSCYTO2D_INITIALIZATION,
        )
        self.assertEqual(pretraining["epochs"], 800)
        self.assertEqual(
            protocol["recipes"]["cytolandfcmae800"]["train_stages"],
            [1, 2],
        )
        command = build_fcmae_command(
            Path("dataset"),
            Path("runs/candidate"),
            protocol["folds"][0]["val_rois"],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--encoder-init-mode") + 1],
            CYTOLAND_VSCYTO2D_INITIALIZATION["mode"],
        )
        self.assertEqual(
            command[command.index("--encoder-init-sha256") + 1],
            CYTOLAND_VSCYTO2D_INITIALIZATION["sha256"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            round62["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_cytoland_split_tiny"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0875,
                "officialbasefcmae800": 0.175,
                "cytolandfcmae800": 0.0875,
            },
        )

        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["encoder_initialization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audited official/public"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round75_cytoland_fcmae_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            CYTOLAND_VSCYTO2D_INITIALIZATION["sha256"],
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND75_PREFLIGHT_ONLY", launcher)
        self.assertIn("round75_cytoland_fcmae_smoke", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)
        self.assertIn("ROUND75_MIN_EVAL_FREE_MEMORY_MB", launcher)

    def test_round76_preregisters_direct_cytoland_decoder_transfer(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        round62 = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND76_PROTOCOL_PATH)

        self.assertNotIn("fcmae_pretraining", protocol)
        self.assertEqual(protocol["folds"], round52["folds"])
        self.assertEqual(protocol["training"], round52["training"])
        encoder = protocol["recipes"]["cytolandencoder"]
        transfer = protocol["recipes"]["cytolandtransfer"]
        for recipe in (encoder, transfer):
            self.assertEqual(
                recipe["cytoland_initialization"],
                CYTOLAND_VSCYTO2D_INITIALIZATION,
            )
            self.assertEqual(recipe["train_stages"], [1, 2])
            self.assertNotIn("fcmae_encoder_init", recipe)
        self.assertEqual(encoder["cytoland_initialization_scope"], "encoder")
        self.assertEqual(
            transfer["cytoland_initialization_scope"], "encoder-decoder"
        )

        for recipe, scope in ((encoder, "encoder"), (transfer, "encoder-decoder")):
            stage1 = build_train_command(
                "python",
                Path("dataset"),
                Path("runs/stage1"),
                list(protocol["folds"][0]["val_rois"]),
                "CD68",
                recipe,
                protocol["training"],
                1,
                None,
                "cuda:0",
            )
            self.assertEqual(
                stage1[stage1.index("--cytoland-init-scope") + 1], scope
            )
            self.assertEqual(
                stage1[stage1.index("--cytoland-init-sha256") + 1],
                CYTOLAND_VSCYTO2D_INITIALIZATION["sha256"],
            )
            stage2 = build_train_command(
                "python",
                Path("dataset"),
                Path("runs/stage2"),
                list(protocol["folds"][0]["val_rois"]),
                "CD68",
                recipe,
                protocol["training"],
                2,
                Path("runs/stage1/best_global_mse.pt"),
                "cuda:0",
            )
            self.assertNotIn("--cytoland-init-checkpoint", stage2)

        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            round62["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["distribution_gates"]["cytoland_transfer_prior"]["requires"],
            "cytoland_decoder_effect",
        )
        self.assertEqual(
            protocol["distribution_gates"]["cytoland_transfer_replacement"][
                "requires"
            ],
            "cytoland_transfer_prior",
        )

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["cytolandtransfer"]["cytoland_initialization"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "audited public"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["cytolandtransfer"][
            "cytoland_initialization_scope"
        ] = "decoder"
        with self.assertRaisesRegex(ValueError, "invalid Cytoland"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round76_cytoland_direct_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            CYTOLAND_VSCYTO2D_INITIALIZATION["sha256"],
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND76_PREFLIGHT_ONLY", launcher)
        self.assertIn("round76_cytoland_${scope}_smoke", launcher)
        self.assertIn("fold0:cytolandencoder", launcher)
        self.assertIn("fold2:cytolandtransfer", launcher)
        self.assertIn("ROUND76_MIN_EVAL_FREE_MEMORY_MB", launcher)
        self.assertIn('config.get("init_checkpoint_report", {})', launcher)

    def test_round77_preregisters_one_fixed_encoder_composition(self) -> None:
        round76 = load_protocol(ROUND76_PROTOCOL_PATH)
        protocol = load_protocol(ROUND77_PROTOCOL_PATH)

        self.assertEqual(protocol["folds"], round76["folds"])
        self.assertNotIn("fcmae_pretraining", protocol)
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_cytoland_encoder_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0875,
                "officialbasefcmae800": 0.175,
                "cytolandencoder": 0.0875,
            },
        )
        self.assertEqual(
            set(protocol["candidates"]),
            {
                "officialfcmae800",
                "cytolandencoder",
                "t70_base_split",
                "t70_base_cytoland_encoder_split",
            },
        )
        gate = protocol["distribution_gates"]["cytoland_encoder_composition"]
        self.assertEqual(gate["minimum_q95_gain"], 0.005)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        self.assertEqual(gate["minimum_roi_q95_gain"], -0.04)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round77_cytoland_encoder_composition.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "1f7223566a654f730212fa23ac1436e20888cb9cf8f4fc9720c449af8e580312",
            "d34690c569d4e6b5a350ea228f37c9034aebbb1373a864deacba27d3ed56ac13",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND77_PREFLIGHT_ONLY", launcher)
        self.assertIn("checked != 16", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round78_preregisters_target_intensity_balanced_l1(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        round62 = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND78_PROTOCOL_PATH)
        recipe = protocol["recipes"]["targetl1s2"]

        self.assertEqual(protocol["folds"], round52["folds"])
        self.assertEqual(protocol["training"], round52["training"])
        self.assertEqual(recipe["target_l1_weight_strength"], 2.0)
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(
            protocol["fcmae_pretraining"]["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/stage1"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--target-l1-weight-strength") + 1], "2.0"
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            round62["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_targetl1_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0875,
                "officialbasefcmae800": 0.175,
                "targetl1s2": 0.0875,
            },
        )

        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["targetl1s2"]["target_l1_weight_strength"] = -1.0
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["officialfcmae800"]["target_l1_weight_strength"] = 2.0
        with self.assertRaisesRegex(ValueError, "requires training stages"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round78_target_l1_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9",
            "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND78_PREFLIGHT_ONLY", launcher)
        self.assertIn("--target-l1-weight-strength 2.0", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)

    def test_round79_preregisters_matched_base_auxiliary_interaction(self) -> None:
        round62 = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND79_PROTOCOL_PATH)
        control = protocol["recipes"]["baseauxcontrol"]
        auxiliary = protocol["recipes"]["baseauxiliary15"]

        self.assertEqual(protocol["folds"], round62["folds"])
        self.assertEqual(protocol["training"], round62["training"])
        self.assertEqual(
            protocol["fcmae_pretraining"]["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_BASE_INITIALIZATION,
        )
        for recipe in (control, auxiliary):
            self.assertEqual(recipe["model_arch"], "unext2-fcmae-base")
            self.assertEqual(recipe["batch_size"], 8)
            self.assertEqual(recipe["gradient_accumulation_steps"], 2)
            self.assertEqual(
                recipe["training_markers"],
                ["CD68", "CD45RO", "HLA-DR", "Vimentin"],
            )
            self.assertTrue(recipe["fcmae_encoder_init"])
            self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(control["marker_loss_weights"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(auxiliary["marker_loss_weights"], [0.85, 0.05, 0.05, 0.05])

        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/stage1"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            auxiliary,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--batch-size") + 1], "8")
        self.assertEqual(
            command[command.index("--gradient-accumulation-steps") + 1], "2"
        )
        self.assertEqual(
            command[command.index("--markers") + 1],
            "CD68,CD45RO,HLA-DR,Vimentin",
        )
        self.assertEqual(
            command[command.index("--marker-loss-weights") + 1],
            "0.85,0.05,0.05,0.05",
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_auxiliary_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialbasefcmae800": 0.0875,
                "baseauxiliary15": 0.0875,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["base_auxiliary_diversity"]["requires"],
            "base_auxiliary_diversity_effect",
        )

        launcher = (
            PROJECT_ROOT / "tools" / "run_round79_base_auxiliary_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            "47c31d8b17a5c3491f187073c72b470a76a45439e1a112569701b02ed972df01",
            "d7a0c27f7e1fa09a5bb17b3eb0221fe21e830ed613bea6bf3bb4767806f70213",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND79_PREFLIGHT_ONLY", launcher)
        self.assertIn("pending=(", launcher)
        self.assertIn("minimum_train_free_memory_mb", launcher)
        self.assertIn("gpu_snapshot \"$gpu\" 5", launcher)

    def test_round80_preregisters_fixed_candidate_output_shift(self) -> None:
        round62 = load_protocol(ROUND62_PROTOCOL_PATH)
        protocol = load_protocol(ROUND80_PROTOCOL_PATH)

        self.assertEqual(protocol["folds"], round62["folds"])
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            round62["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split_shift_xm1"],
            round62["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_base_split_shift_xm1"
            ),
            (0, -1),
        )
        self.assertEqual(
            candidate_output_source_offset(protocol, "t70_base_split"),
            (0, 0),
        )
        gate = protocol["distribution_gates"]["fixed_output_shift"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 8)
        self.assertEqual(gate["minimum_roi_q95_gain"], -0.05)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round80_fixed_output_shift.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe",
            launcher,
        )
        self.assertIn("ROUND80_PREFLIGHT_ONLY", launcher)
        self.assertIn("checked != 10", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round81_expands_fixed_shift_to_complete_five_fold_oof(self) -> None:
        round70 = load_protocol(ROUND70_PROTOCOL_PATH)
        protocol = load_protocol(ROUND81_PROTOCOL_PATH)

        self.assertEqual(protocol["folds"], round70["folds"])
        self.assertEqual(len(protocol["expected_rois"]), 25)
        self.assertEqual(protocol["roi_scope"], "complete")
        self.assertTrue(
            all(not recipe["train_stages"] for recipe in protocol["recipes"].values())
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split"],
            round70["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            protocol["candidates"]["t70_base_split_shift_xm1"],
            round70["candidates"]["t70_base_split"],
        )
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_base_split_shift_xm1"
            ),
            (0, -1),
        )
        gate = protocol["distribution_gates"]["fixed_output_shift_full"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 5)
        self.assertEqual(gate["minimum_positive_rois"], 20)
        self.assertEqual(gate["minimum_roi_q95_gain"], -0.05)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round81_cv5_fixed_output_shift.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "29ba58e2c86db1caf9a87a05943a0c730d3753e9f6e465f61f47c2e78ce2b893",
            "a72b75b738286e4ebd8db6c3f968169b3b12de1cc67f6d67e71406992cf49709",
            "e3e854ecf193e95fe0b1d9b721a87361e98c3ac3b0c11684f302d22f18eec372",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND81_PREFLIGHT_ONLY", launcher)
        self.assertIn("checked != 25", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round82_packages_only_the_fixed_retrospective_candidate(self) -> None:
        launcher = (
            PROJECT_ROOT / "tools" / "run_round82_shift_submission.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "c2418b53d443c026cc9589a95f48438f6e50069d3f64fb9bc8e81ca66243def7",
            "aee7ecd340a1f242db1e7cca0e7afdd87bcfed96ec4d767d2a722f22fe2b96cf",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND82_PREFLIGHT_ONLY", launcher)
        self.assertIn('"roi_floor": False', launcher)
        self.assertIn('len(record["positive_folds"]) != 5', launcher)
        self.assertIn('len(record["positive_rois"]) != 21', launcher)
        self.assertIn("t70_base_split_shift_xm1", launcher)
        self.assertIn('report.get("output_source_offset") != [0, -1]', launcher)
        self.assertIn("checked != 25", launcher)
        self.assertIn("package_submission.py", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round83_preregisters_inverse_training_and_output_offsets(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        protocol = load_protocol(ROUND83_PROTOCOL_PATH)
        recipe = protocol["recipes"]["alignedtarget"]

        self.assertEqual(protocol["folds"], round52["folds"])
        self.assertEqual(protocol["training"], round52["training"])
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(recipe["train_target_context_halo"], 64)
        self.assertEqual(recipe["train_target_source_offset"], [0, 1])
        self.assertEqual(recipe["validation_output_source_offset"], [0, -1])
        self.assertEqual(recipe["train_stages"], [1, 2])

        for stage, init_checkpoint in (
            (1, None),
            (2, Path("runs/stage1/best_global_mse.pt")),
        ):
            command = build_train_command(
                "python",
                Path("dataset"),
                Path(f"runs/stage{stage}"),
                list(protocol["folds"][0]["val_rois"]),
                "CD68",
                recipe,
                protocol["training"],
                stage,
                init_checkpoint,
                "cuda:0",
            )
            self.assertEqual(
                command[command.index("--train-target-source-offset") + 1],
                "0,1",
            )
            self.assertEqual(
                command[command.index("--validation-output-source-offset") + 1],
                "0,-1",
            )

        self.assertEqual(
            set(protocol["candidate_output_source_offsets"]),
            set(protocol["candidates"]),
        )
        self.assertTrue(
            all(
                candidate_output_source_offset(protocol, candidate) == (0, -1)
                for candidate in protocol["candidates"]
            )
        )

        invalid = copy.deepcopy(protocol)
        del invalid["recipes"]["alignedtarget"]["validation_output_source_offset"]
        with self.assertRaisesRegex(ValueError, "configured together"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["alignedtarget"]["validation_output_source_offset"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "must be opposites"):
            validate_protocol(invalid)
        invalid = copy.deepcopy(protocol)
        invalid["recipes"]["alignedtarget"]["train_target_context_halo"] = 0
        with self.assertRaisesRegex(ValueError, "exceeds its target halo"):
            validate_protocol(invalid)

        launcher = (
            PROJECT_ROOT / "tools" / "run_round83_aligned_target_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "29ba58e2c86db1caf9a87a05943a0c730d3753e9f6e465f61f47c2e78ce2b893",
            "a72b75b738286e4ebd8db6c3f968169b3b12de1cc67f6d67e71406992cf49709",
            "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9",
            "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND83_PREFLIGHT_ONLY", launcher)
        self.assertIn("--train-target-source-offset 0,1", launcher)
        self.assertIn("--validation-output-source-offset 0,-1", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)

    def test_round84_preregisters_independent_official_fcmae_seed(self) -> None:
        protocol = load_protocol(ROUND84_PROTOCOL_PATH)
        recipe = protocol["recipes"]["independentseed"]
        self.assertEqual(protocol["training"]["seed"], 2027)
        self.assertTrue(recipe["fcmae_encoder_init"])
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(
            protocol["run_sources"]["independentseed"],
            {
                "protocol_name": "round84_cv2_official_fcmae_independent_seed",
                "recipe_name": "independentseed",
                "stage": 2,
            },
        )
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_base_independent_replace_shift_xm1"
            ),
            (0, -1),
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/independent"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--seed") + 1], "2027")
        launcher = (
            PROJECT_ROOT
            / "tools"
            / "run_round84_independent_seed_pipeline.sh"
        ).read_text(encoding="utf-8")
        for digest in (
            "29ba58e2c86db1caf9a87a05943a0c730d3753e9f6e465f61f47c2e78ce2b893",
            "a72b75b738286e4ebd8db6c3f968169b3b12de1cc67f6d67e71406992cf49709",
            "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9",
            "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7",
        ):
            self.assertIn(digest, launcher)
        self.assertIn("ROUND84_PREFLIGHT_ONLY", launcher)
        self.assertIn("--recipes independentseed", launcher)
        self.assertIn("pending=(fold0 fold2)", launcher)

    def test_round85_preregisters_context128_inference_only_screen(self) -> None:
        protocol = load_protocol(ROUND85_PROTOCOL_PATH)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(protocol["inference"]["context_halo"], 128)
        self.assertTrue(
            all(
                recipe["train_stages"] == []
                and recipe["inference_context_halo"] == 128
                for recipe in protocol["recipes"].values()
            )
        )
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_base_split_context128_shift_xm1"
            ),
            (0, -1),
        )
        self.assertEqual(
            protocol["context128_gate"]["baseline_report_candidate"],
            "t70_base_split_shift_xm1",
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round85_context128_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("c2418b53d443c026cc9589a95f48438f6e50069d3f64fb9bc8e81ca66243def7", launcher)
        self.assertIn("evaluate_context128_gate.py", launcher)
        self.assertIn("ROUND85_PREFLIGHT_ONLY", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round86_preregisters_context128_training_with_fixed_effective_batch(
        self,
    ) -> None:
        protocol = load_protocol(ROUND86_PROTOCOL_PATH)
        recipe = protocol["recipes"]["context128train"]
        self.assertEqual(recipe["context_halo"], 128)
        self.assertEqual(recipe["train_target_context_halo"], 64)
        self.assertEqual(recipe["batch_size"], 8)
        self.assertEqual(recipe["gradient_accumulation_steps"], 2)
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(protocol["training"]["batch_size"], 16)
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_context128_train_replace_shift_xm1"
            ),
            (0, -1),
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/context128"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--context-halo") + 1], "128")
        self.assertEqual(
            command[command.index("--train-target-context-halo") + 1], "64"
        )
        self.assertEqual(
            command[command.index("--gradient-accumulation-steps") + 1], "2"
        )
        launcher = (
            PROJECT_ROOT
            / "tools"
            / "run_round86_context128_training_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f", launcher)
        self.assertIn("--context-halo 128", launcher)
        self.assertIn("--train-target-context-halo 64", launcher)
        self.assertIn("--batch-size 8 --gradient-accumulation-steps 2", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)
        self.assertLess(launcher.index("select_gpu()"), launcher.index('smoke_gpu="$(select_gpu'))

    def test_round87_preregisters_context128_fulltarget_training(self) -> None:
        protocol = load_protocol(ROUND87_PROTOCOL_PATH)
        recipe = protocol["recipes"]["context128fulltarget"]
        self.assertEqual(recipe["context_halo"], 128)
        self.assertEqual(recipe["train_target_context_halo"], 128)
        self.assertEqual(recipe["inference_context_halo"], 128)
        self.assertEqual(recipe["batch_size"], 8)
        self.assertEqual(recipe["gradient_accumulation_steps"], 2)
        self.assertEqual(recipe["train_stages"], [1, 2])
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_context128_fulltarget_replace_shift_xm1"
            ),
            (0, -1),
        )
        command = build_train_command(
            "python",
            Path("dataset"),
            Path("runs/context128fulltarget"),
            list(protocol["folds"][0]["val_rois"]),
            "CD68",
            recipe,
            protocol["training"],
            1,
            None,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--context-halo") + 1], "128")
        self.assertEqual(
            command[command.index("--train-target-context-halo") + 1], "128"
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round87_context128_fulltarget_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f", launcher)
        self.assertIn("--train-target-context-halo 128", launcher)
        self.assertIn("--recipes context128fulltarget", launcher)
        self.assertIn("evaluate_distribution_gate.py", launcher)

    def test_round97_preregisters_full_neighbor_ring_inference_only(self) -> None:
        protocol = load_protocol(ROUND97_PROTOCOL_PATH)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertEqual(protocol["inference"]["context_halo"], 256)
        self.assertEqual(protocol["inference"]["batch_size"], 1)
        self.assertTrue(
            all(
                recipe["train_stages"] == []
                and recipe["inference_context_halo"] == 256
                for recipe in protocol["recipes"].values()
            )
        )
        self.assertEqual(
            candidate_output_source_offset(
                protocol, "t70_base_split_context256_shift_xm1"
            ),
            (0, -1),
        )
        gate = protocol["context_scale_gate"]
        self.assertEqual(gate["baseline_protocol_name"], ROUND85_PROTOCOL_PATH.stem)
        self.assertEqual(
            gate["baseline_report_candidate"],
            "t70_base_split_context128_shift_xm1",
        )
        self.assertEqual(
            (gate["baseline_context_halo"], gate["candidate_context_halo"]),
            (128, 256),
        )
        launcher = (
            PROJECT_ROOT / "tools" / "run_round97_context256_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f",
            launcher,
        )
        self.assertIn("evaluate_context128_gate.py", launcher)
        self.assertIn("ROUND97_PREFLIGHT_ONLY", launcher)
        self.assertNotIn("run_cv_training.py", launcher)

    def test_round56_refines_auxiliary_stage1_with_primary_only_stage2(
        self,
    ) -> None:
        round55 = load_protocol(ROUND55_PROTOCOL_PATH)
        protocol = load_protocol(ROUND56_PROTOCOL_PATH)
        recipe = protocol["recipes"]["officialauxwarmprimary"]

        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(protocol["training"], round55["training"])
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(
            recipe["training_markers"],
            ["CD68", "CD45RO", "HLA-DR", "Vimentin"],
        )
        self.assertEqual(recipe["marker_loss_weights"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(recipe["train_stages"], [2])
        self.assertEqual(
            recipe["init_source"],
            {
                "protocol_name": "round55_cv2_unext2_official_auxiliary",
                "recipe_name": "officialauxiliary15",
                "stage": 1,
            },
        )
        self.assertEqual(
            initialization_checkpoint(
                Path("runs"),
                protocol,
                "fold0",
                "officialauxwarmprimary",
                2,
            ),
            Path(
                "runs/round55_cv2_unext2_official_auxiliary_"
                "fold0_officialauxiliary15_stage1/best_global_mse.pt"
            ),
        )
        self.assertEqual(
            protocol["run_sources"]["officialauxwarmprimary"],
            {
                "protocol_name": "round56_cv2_unext2_auxiliary_warm_primary",
                "recipe_name": "officialauxwarmprimary",
                "stage": 2,
            },
        )
        self.assertEqual(
            protocol["candidates"]["t70_official_auxwarm_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialauxwarmprimary": 0.175,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["warm_diversity_increment"],
            {
                "candidate": "t70_official_auxwarm_split",
                "baseline": "t70_official_auxiliary_split",
                "minimum_q95_gain": 0.01,
                "require_positive_float": True,
                "minimum_positive_folds": 2,
                "minimum_positive_rois": 8,
                "minimum_roi_q95_gain": -0.06,
                "requires": "warm_schedule_mechanism",
            },
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["warm_replacement", "warm_diversity"],
        )
        weights = effective_member_weights(
            protocol,
            "t70_official_auxwarm_split",
        )
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round57_refines_primary_stage1_with_late_auxiliary_stage2(
        self,
    ) -> None:
        round55 = load_protocol(ROUND55_PROTOCOL_PATH)
        round56 = load_protocol(ROUND56_PROTOCOL_PATH)
        protocol = load_protocol(ROUND57_PROTOCOL_PATH)
        recipe = protocol["recipes"]["officialprimarywarmaux15"]

        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(protocol["training"], round55["training"])
        self.assertEqual(protocol["training"], round56["training"])
        self.assertEqual(recipe["model_arch"], "unext2-fcmae-tiny")
        self.assertEqual(
            recipe["training_markers"],
            ["CD68", "CD45RO", "HLA-DR", "Vimentin"],
        )
        self.assertEqual(
            recipe["marker_loss_weights"],
            [0.85, 0.05, 0.05, 0.05],
        )
        self.assertEqual(recipe["train_stages"], [2])
        self.assertEqual(
            recipe["init_source"],
            {
                "protocol_name": "round55_cv2_unext2_official_auxiliary",
                "recipe_name": "officialauxcontrol",
                "stage": 1,
            },
        )
        self.assertEqual(
            initialization_checkpoint(
                Path("runs"),
                protocol,
                "fold2",
                "officialprimarywarmaux15",
                2,
            ),
            Path(
                "runs/round55_cv2_unext2_official_auxiliary_"
                "fold2_officialauxcontrol_stage1/best_global_mse.pt"
            ),
        )
        self.assertEqual(
            protocol["run_sources"]["officialauxwarmprimary"],
            {
                "protocol_name": "round56_cv2_unext2_auxiliary_warm_primary",
                "recipe_name": "officialauxwarmprimary",
                "stage": 2,
            },
        )
        self.assertEqual(
            protocol["candidates"]["t70_official_auxlate_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialprimarywarmaux15": 0.175,
            },
        )
        self.assertEqual(
            protocol["distribution_gates"]["late_schedule_advance"],
            {
                "candidate": "officialprimarywarmaux15",
                "baseline": "officialauxiliary15",
                "minimum_q95_gain": 0.02,
                "require_positive_float": True,
                "minimum_positive_folds": 2,
                "minimum_positive_rois": 7,
                "minimum_roi_q95_gain": -0.10,
                "requires": "late_auxiliary_mechanism",
            },
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["late_replacement", "late_diversity"],
        )
        weights = effective_member_weights(
            protocol,
            "t70_official_auxlate_split",
        )
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_round58_preregisters_fresh_official_fcmae1200_screen(self) -> None:
        round52 = load_protocol(ROUND52_PROTOCOL_PATH)
        protocol = load_protocol(ROUND58_PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]

        self.assertEqual(
            [fold["name"] for fold in protocol["folds"]],
            ["fold0", "fold2"],
        )
        self.assertEqual(pretraining["epochs"], 1200)
        self.assertEqual(
            pretraining["encoder_initialization"],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        )
        for key in (
            "batch_size",
            "workers",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_epochs",
            "weight_decay",
            "mask_ratio",
            "checkpoint_interval",
            "seed",
        ):
            self.assertEqual(
                pretraining[key],
                round52["fcmae_pretraining"][key],
                msg=key,
            )
        self.assertNotEqual(
            pretraining["name"],
            round52["fcmae_pretraining"]["name"],
        )
        self.assertEqual(protocol["training"], round52["training"])
        self.assertEqual(
            protocol["recipes"]["officialfcmae1200"]["train_stages"],
            [1, 2],
        )
        self.assertTrue(
            protocol["recipes"]["officialfcmae1200"]["fcmae_encoder_init"]
        )

        command = build_fcmae_command(
            Path("dataset"),
            Path("runs/fresh-round58"),
            protocol["folds"][0]["val_rois"],
            pretraining,
            "cuda:0",
        )
        self.assertEqual(command[command.index("--epochs") + 1], "1200")
        self.assertNotIn("--resume", command)
        self.assertEqual(
            command[command.index("--encoder-init-mode") + 1],
            OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION["mode"],
        )
        self.assertEqual(
            protocol["run_sources"]["officialfcmae800"],
            {
                "protocol_name": "round52_cv2_unext2_official_fcmae",
                "recipe_name": "officialfcmae800",
                "stage": 2,
            },
        )
        self.assertEqual(
            protocol["candidates"]["t70_official1200_split"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.175,
                "officialfcmae1200": 0.175,
            },
        )
        self.assertNotIn(
            "requires",
            protocol["distribution_gates"]["longer_official_diversity"],
        )
        self.assertEqual(
            protocol["distribution_promotion_order"],
            ["longer_official_replacement", "longer_official_diversity"],
        )
        weights = effective_member_weights(
            protocol,
            "t70_official1200_split",
        )
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
