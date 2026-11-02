from __future__ import annotations

import gc
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from ihc_spatial import (
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
    SPATIAL_WHOLE_ROI_CONDITION_DIMENSION,
    SPATIAL_WHOLE_ROI_CONDITION_MODE,
    spatial_whole_roi_conditioning_specification,
)
from ihc_torch import (
    build_model,
    build_model_from_checkpoint,
    checkpoint_model_specification,
    model_specification,
)
from ihc_unext2 import (
    CYTOLAND_VSCYTO2D_TINY_MODE,
    ConvNeXtV2Block,
    DapiFCMAE,
    OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE,
    OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE,
    OFFICIAL_CONVNEXTV2_TINY_INITIALIZATION_SPECS,
    UNeXt2BaseVirtualStainer,
    UNeXt2CrossMarkerFeedbackVirtualStainer,
    UNeXt2DapiDetailResidualVirtualStainer,
    UNeXt2MarkerDecoderVirtualStainer,
    UNeXt2ShallowDapiSkipVirtualStainer,
    UNeXt2VirtualStainer,
    UNeXt2WideHeadVirtualStainer,
    generate_fcmae_mask,
    cytoland_vscyto2d_decoder_state_mapping,
    cytoland_vscyto2d_encoder_state_mapping,
    initialize_fcmae_encoder_from_cytoland_vscyto2d,
    initialize_unext2_from_cytoland_vscyto2d,
    initialize_fcmae_encoder_from_official_convnextv2,
    initialize_unext2_encoder_from_fcmae,
    official_convnextv2_fcmae_state_mapping,
    official_convnextv2_fcmae_tiny_state_mapping,
    unext2_encoder_specification,
)
from train_dapi_fcmae import (
    cuda_rng_states_for_restore,
    gradient_accumulation_sample_scale,
    learning_rate_for_epoch,
)

import train_unet


class UNeXt2ForwardTests(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()

    def test_fcmae_mask_and_reconstruction_contract(self) -> None:
        inputs = torch.rand(2, 1, 32, 48)
        mask = generate_fcmae_mask(inputs, mask_ratio=0.5)
        self.assertEqual(tuple(mask.shape), (2, 1, 2, 3))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.flatten(1).sum(dim=1).tolist(), [3, 3])

        model = DapiFCMAE()
        reconstruction, full_mask = model(inputs, mask_ratio=0.5)
        self.assertEqual(tuple(reconstruction.shape), tuple(inputs.shape))
        self.assertEqual(tuple(full_mask.shape), tuple(inputs.shape))
        self.assertEqual(full_mask.dtype, torch.bool)
        self.assertGreaterEqual(float(reconstruction.detach().min()), 0.0)
        self.assertLessEqual(float(reconstruction.detach().max()), 1.0)
        ((reconstruction - inputs).square() * full_mask).mean().backward()
        self.assertIsNotNone(model.encoder.stem[0].weight.grad)

    def test_virtual_stainer_condition_and_factory_contract(self) -> None:
        model = build_model(
            "unext2-fcmae-tiny",
            out_channels=1,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(model, UNeXt2VirtualStainer)
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        output = model(inputs, condition)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, "requires a condition"):
            model(inputs)
        with self.assertRaisesRegex(ValueError, "condition shape"):
            model(inputs, condition[:, :-1])

    def test_texture_condition_and_factory_contract(self) -> None:
        model = build_model(
            "unext2-fcmae-tiny",
            out_channels=1,
            condition_mode=SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
        )
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION)
        output = model(inputs, condition)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        self.assertEqual(model.condition_dimension, SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION)

    def test_base_factory_uses_official_encoder_shape(self) -> None:
        model = build_model(
            "unext2-fcmae-base",
            out_channels=1,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(model, UNeXt2BaseVirtualStainer)
        self.assertEqual(model.model_size, "base")
        self.assertEqual(model.encoder.block_counts, (3, 3, 27, 3))
        self.assertEqual(model.encoder.dimensions, (128, 256, 512, 1024))
        self.assertEqual(
            model_specification("unext2-fcmae-base")["architecture"],
            "unext2-fcmae-base",
        )
        self.assertEqual(
            model.decoder.stages[0].projection[1].out_channels,
            512,
        )

    def test_differential_lr_groups_cover_unext2_once(self) -> None:
        model = build_model(
            "unext2-fcmae-tiny",
            out_channels=1,
            condition_mode="spatial-thumbnail-film",
        )
        optimizer, report = train_unet.build_optimizer(
            model,
            "unext2-fcmae-tiny",
            learning_rate=3e-4,
            encoder_learning_rate=3e-5,
            weight_decay=1e-4,
        )
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["decoder_head", "encoder"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [3e-4, 3e-5],
        )
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertEqual(
            sum(item["parameters"] for item in report),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_encoder_drop_path_is_linear_and_eval_identity(self) -> None:
        torch.manual_seed(2050)
        control = build_model("unext2-fcmae-tiny", out_channels=1)
        torch.manual_seed(2050)
        candidate = build_model("unext2-fcmae-tiny", out_channels=1)
        original_keys = tuple(candidate.state_dict())
        rates = candidate.configure_encoder_drop_path(0.2)
        self.assertEqual(len(rates), 18)
        self.assertEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[-1], 0.2)
        self.assertTrue(all(left <= right for left, right in zip(rates, rates[1:])))
        self.assertEqual(tuple(candidate.state_dict()), original_keys)
        decoder_rates = [
            block.drop_path_rate
            for stage in candidate.decoder.stages
            for block in stage.blocks
        ]
        self.assertEqual(decoder_rates, [0.0] * len(decoder_rates))
        inputs = torch.rand(2, 1, 32, 32)
        control.eval()
        candidate.eval()
        torch.testing.assert_close(candidate(inputs), control(inputs), rtol=0, atol=0)

    def test_encoder_drop_path_changes_training_residuals(self) -> None:
        block = ConvNeXtV2Block(4, drop_path_rate=0.5)
        inputs = torch.rand(16, 4, 8, 8)
        block.eval()
        evaluation = block(inputs)
        block.train()
        torch.manual_seed(2051)
        training = block(inputs)
        self.assertFalse(torch.equal(training, evaluation))
        with self.assertRaisesRegex(ValueError, "drop path rate"):
            ConvNeXtV2Block(4, drop_path_rate=1.0)

    def test_wide_head_factory_preserves_encoder_and_expands_decoder(self) -> None:
        torch.manual_seed(2049)
        baseline = UNeXt2VirtualStainer(condition_mode="spatial-thumbnail-film")
        torch.manual_seed(2049)
        candidate = build_model(
            "unext2-fcmae-tiny-widehead",
            out_channels=1,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(candidate, UNeXt2WideHeadVirtualStainer)
        for key, value in baseline.encoder.state_dict().items():
            torch.testing.assert_close(
                candidate.encoder.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )
        for key, value in baseline.condition_film.state_dict().items():
            torch.testing.assert_close(
                candidate.condition_film.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )
        self.assertEqual(candidate.decoder.stages[-1].projection[1].out_channels, 48)
        self.assertGreater(
            sum(parameter.numel() for parameter in candidate.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        output = candidate(inputs, condition)
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))
        output.mean().backward()
        self.assertGreater(
            float(candidate.output_head.projection[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_marker_decoders_preserve_primary_initialization_and_isolate_gradients(
        self,
    ) -> None:
        torch.manual_seed(2060)
        baseline = UNeXt2VirtualStainer(condition_mode="spatial-thumbnail-film")
        torch.manual_seed(2060)
        candidate = build_model(
            "unext2-fcmae-tiny-marker-decoders",
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(candidate, UNeXt2MarkerDecoderVirtualStainer)
        self.assertEqual(len(candidate.auxiliary_decoders), 3)
        for module_name in ("encoder", "decoder", "condition_film"):
            baseline_state = getattr(baseline, module_name).state_dict()
            candidate_state = getattr(candidate, module_name).state_dict()
            for key, value in baseline_state.items():
                torch.testing.assert_close(
                    candidate_state[key],
                    value,
                    rtol=0,
                    atol=0,
                )
        del baseline

        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        output = candidate(inputs, condition)
        self.assertEqual(tuple(output.shape), (1, 4, 32, 32))
        output[:, :1].mean().backward()
        self.assertIsNotNone(candidate.decoder.stages[0].projection[1].weight.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                or int(torch.count_nonzero(parameter.grad)) == 0
                for parameter in candidate.auxiliary_decoders.parameters()
            )
        )

    def test_decoder_only_auxiliary_gradients_preserve_values_and_stop_at_decoder(
        self,
    ) -> None:
        model = UNeXt2VirtualStainer(
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.rand(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)

        with torch.no_grad():
            baseline = model(inputs, condition)
            routed = model.forward_with_decoder_only_auxiliary_gradients(
                inputs,
                condition,
            )
        torch.testing.assert_close(routed, baseline, rtol=0, atol=0)

        model.zero_grad(set_to_none=True)
        routed = model.forward_with_decoder_only_auxiliary_gradients(
            inputs,
            condition,
        )
        routed[:, 1:].mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or int(torch.count_nonzero(parameter.grad)) == 0
                for parameter in model.encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                or int(torch.count_nonzero(parameter.grad)) == 0
                for parameter in model.condition_film.parameters()
            )
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.decoder.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertEqual(float(model.output_bias.grad[0]), 0.0)
        self.assertGreater(float(model.output_bias.grad[1:].abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        routed = model.forward_with_decoder_only_auxiliary_gradients(
            inputs,
            condition,
        )
        routed[:, :1].mean().backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.encoder.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_invalid_spatial_size_is_rejected(self) -> None:
        model = UNeXt2VirtualStainer()
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            model(torch.rand(1, 1, 31, 32))

    def test_whole_roi_branch_preserves_shared_initialization_and_zero_output(self) -> None:
        torch.manual_seed(2048)
        baseline = UNeXt2VirtualStainer(
            condition_mode="spatial-thumbnail-film"
        ).eval()
        torch.manual_seed(2048)
        candidate = UNeXt2VirtualStainer(
            condition_mode=SPATIAL_WHOLE_ROI_CONDITION_MODE
        ).eval()

        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        for key, value in baseline_state.items():
            self.assertIn(key, candidate_state)
            torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)

        inputs = torch.rand(1, 1, 32, 32)
        baseline_condition = torch.rand(
            1,
            SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
        )
        candidate_condition = torch.cat(
            (
                baseline_condition,
                torch.rand(
                    1,
                    SPATIAL_WHOLE_ROI_CONDITION_DIMENSION
                    - SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
                ),
            ),
            dim=1,
        )
        with torch.no_grad():
            baseline_output = baseline(inputs, baseline_condition)
            candidate_output = candidate(inputs, candidate_condition)
        torch.testing.assert_close(candidate_output, baseline_output, rtol=0, atol=0)

        candidate.train()
        candidate(inputs, candidate_condition).mean().backward()
        self.assertGreater(
            float(candidate.condition_film.whole_roi_output.weight.grad.abs().sum()),
            0.0,
        )


class UNeXt2CheckpointTests(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()

    def test_fold_local_fcmae_encoder_transfer_is_strict_and_recorded(self) -> None:
        source = DapiFCMAE()
        source_state = source.encoder.state_dict()
        first_key = next(iter(source_state))
        source_state[first_key] = torch.full_like(source_state[first_key], 0.125)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "final_encoder.pt"
            torch.save(
                {
                    "version": 1,
                    "task": "dapi-fcmae-pretraining",
                    "epoch": 100,
                    "val_rois": ["ROI001"],
                    "encoder_spec": unext2_encoder_specification(),
                    "encoder": source_state,
                },
                checkpoint_path,
            )
            target = UNeXt2VirtualStainer()
            report = initialize_unext2_encoder_from_fcmae(
                target,
                checkpoint_path,
                torch.device("cpu"),
            )
        torch.testing.assert_close(
            target.encoder.state_dict()[first_key],
            source_state[first_key],
        )
        self.assertEqual(report["mode"], "fold-local-dapi-fcmae")
        self.assertEqual(report["pretraining_epoch"], 100)
        self.assertEqual(len(report["checkpoint_sha256"]), 64)

    def test_official_fcmae_mapping_covers_every_tensor_except_input_conv(self) -> None:
        encoder_state = DapiFCMAE().encoder.state_dict()
        mapping = official_convnextv2_fcmae_tiny_state_mapping()
        preserved = {"stem.0.weight", "stem.0.bias"}
        self.assertEqual(len(mapping), 194)
        self.assertEqual(set(mapping.values()), set(encoder_state) - preserved)
        loaded_parameters = sum(
            encoder_state[target_key].numel() for target_key in mapping.values()
        )
        total_parameters = sum(tensor.numel() for tensor in encoder_state.values())
        self.assertEqual(loaded_parameters, 27_860_256)
        self.assertEqual(total_parameters, 27_860_736)
        self.assertGreater(loaded_parameters / total_parameters, 0.99998)

    def test_official_base_mapping_covers_every_tensor_except_input_conv(self) -> None:
        encoder_state = DapiFCMAE("base").encoder.state_dict()
        mapping = official_convnextv2_fcmae_state_mapping("base")
        preserved = {"stem.0.weight", "stem.0.bias"}
        self.assertEqual(len(mapping), 374)
        self.assertEqual(set(mapping.values()), set(encoder_state) - preserved)
        loaded_parameters = sum(
            encoder_state[target_key].numel() for target_key in mapping.values()
        )
        total_parameters = sum(tensor.numel() for tensor in encoder_state.values())
        self.assertEqual(loaded_parameters, 87_684_480)
        self.assertEqual(total_parameters, 87_685_120)
        self.assertGreater(loaded_parameters / total_parameters, 0.99999)
        self.assertNotIn(
            OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE,
            OFFICIAL_CONVNEXTV2_TINY_INITIALIZATION_SPECS,
        )

    def test_official_fcmae_checkpoint_hash_and_schema_are_strict(self) -> None:
        encoder = DapiFCMAE().encoder
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "official.pt"
            torch.save({"model": {}}, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                initialize_fcmae_encoder_from_official_convnextv2(
                    encoder,
                    checkpoint_path,
                    "0" * 64,
                )
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "state keys"):
                initialize_fcmae_encoder_from_official_convnextv2(
                    encoder,
                    checkpoint_path,
                    checkpoint_sha256,
                )

            with self.assertRaisesRegex(ValueError, "state keys"):
                initialize_fcmae_encoder_from_official_convnextv2(
                    encoder,
                    checkpoint_path,
                    checkpoint_sha256,
                    OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE,
                )

        supervised_spec = OFFICIAL_CONVNEXTV2_TINY_INITIALIZATION_SPECS[
            OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE
        ]
        self.assertEqual(
            set(supervised_spec["additional_skipped_source_tensors"]),
            {"norm.weight", "norm.bias", "head.weight", "head.bias"},
        )

    def test_cytoland_mapping_loads_stem_and_preserves_only_depthwise_biases(
        self,
    ) -> None:
        encoder_state = DapiFCMAE().encoder.state_dict()
        mapping = cytoland_vscyto2d_encoder_state_mapping()
        preserved = {
            f"stages.{stage}.blocks.{block}.depthwise.bias"
            for stage, block_count in enumerate((3, 3, 9, 3))
            for block in range(block_count)
        }
        self.assertEqual(len(mapping), 178)
        self.assertEqual(set(mapping.values()), set(encoder_state) - preserved)
        self.assertIn("stem.0.weight", mapping.values())
        loaded_parameters = sum(
            encoder_state[target_key].numel() for target_key in mapping.values()
        )
        total_parameters = sum(tensor.numel() for tensor in encoder_state.values())
        self.assertEqual(loaded_parameters, 27_854_112)
        self.assertEqual(total_parameters, 27_860_736)
        self.assertGreater(loaded_parameters / total_parameters, 0.9997)

    def test_cytoland_checkpoint_hash_and_encoder_schema_are_strict(self) -> None:
        encoder = DapiFCMAE().encoder
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "cytoland.ckpt"
            torch.save({"state_dict": {}}, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                initialize_fcmae_encoder_from_cytoland_vscyto2d(
                    encoder,
                    checkpoint_path,
                    "0" * 64,
                    CYTOLAND_VSCYTO2D_TINY_MODE,
                )
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "encoder state keys"):
                initialize_fcmae_encoder_from_cytoland_vscyto2d(
                    encoder,
                    checkpoint_path,
                    checkpoint_sha256,
                    CYTOLAND_VSCYTO2D_TINY_MODE,
                )

    def test_cytoland_supervised_transfer_loads_only_fixed_compatible_stages(
        self,
    ) -> None:
        candidate = UNeXt2VirtualStainer(
            out_channels=1,
            condition_mode="spatial-thumbnail-film",
        )
        source_two_output = UNeXt2VirtualStainer(out_channels=2)
        source_state: dict[str, torch.Tensor] = {}
        for source_key, target_key in cytoland_vscyto2d_encoder_state_mapping().items():
            source_state[source_key] = candidate.encoder.state_dict()[target_key].clone()
        source_state["model.encoder.stem.conv3d.weight"] = torch.zeros(96, 1, 1, 2, 2)
        source_state["model.encoder.stem.conv3d.bias"] = torch.zeros(96)

        for stages, decoder in (
            ((0, 1), candidate.decoder),
            ((2,), source_two_output.decoder),
        ):
            decoder_state = decoder.state_dict()
            for source_key, target_key in cytoland_vscyto2d_decoder_state_mapping(
                stages
            ).items():
                tensor = decoder_state[target_key].clone()
                if source_key.endswith(("mlp.fc1.weight", "mlp.fc2.weight")):
                    tensor = tensor[..., None, None]
                source_state[source_key] = tensor

        self.assertEqual(len(source_state), 252)
        preserved_stage2 = {
            key: value.clone()
            for key, value in candidate.decoder.state_dict().items()
            if key.startswith("stages.2.")
        }
        preserved_condition = {
            key: value.clone()
            for key, value in candidate.condition_film.state_dict().items()
        }
        preserved_bias = candidate.output_bias.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "cytoland.ckpt"
            torch.save({"state_dict": source_state}, checkpoint_path)
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            report = initialize_unext2_from_cytoland_vscyto2d(
                candidate,
                checkpoint_path,
                checkpoint_sha256,
                CYTOLAND_VSCYTO2D_TINY_MODE,
                "encoder-decoder",
            )

        self.assertEqual(report["state_tensors_loaded"], 178)
        self.assertEqual(report["decoder_state_tensors_loaded"], 48)
        self.assertEqual(report["decoder_parameters_loaded"], 3_302_784)
        self.assertEqual(len(report["decoder_reshaped_source_tensors"]), 8)
        current_decoder = candidate.decoder.state_dict()
        for key, value in preserved_stage2.items():
            torch.testing.assert_close(current_decoder[key], value)
        for key, value in preserved_condition.items():
            torch.testing.assert_close(candidate.condition_film.state_dict()[key], value)
        torch.testing.assert_close(candidate.output_bias, preserved_bias)

        encoder_only = UNeXt2VirtualStainer(out_channels=1)
        decoder_before = {
            key: value.clone() for key, value in encoder_only.decoder.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "cytoland.ckpt"
            torch.save({"state_dict": source_state}, checkpoint_path)
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            report = initialize_unext2_from_cytoland_vscyto2d(
                encoder_only,
                checkpoint_path,
                checkpoint_sha256,
                CYTOLAND_VSCYTO2D_TINY_MODE,
                "encoder",
            )
        self.assertEqual(report["decoder_state_tensors_loaded"], 0)
        for key, value in decoder_before.items():
            torch.testing.assert_close(encoder_only.decoder.state_dict()[key], value)

    def test_downstream_checkpoint_roundtrip_never_needs_pretraining_file(self) -> None:
        source = UNeXt2VirtualStainer(condition_mode="none")
        specification = model_specification("unext2-fcmae-tiny")
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": "unext2-fcmae-tiny",
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2VirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key in ("encoder.stem.0.weight", "decoder.stages.0.projection.1.weight"):
            torch.testing.assert_close(
                restored.state_dict()[key],
                source.state_dict()[key],
            )

    def test_base_downstream_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-base"
        source = UNeXt2BaseVirtualStainer(condition_mode="none")
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2BaseVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key in (
            "encoder.stem.0.weight",
            "encoder.stages.2.blocks.26.project.weight",
            "decoder.stages.0.projection.1.weight",
        ):
            torch.testing.assert_close(
                restored.state_dict()[key],
                source.state_dict()[key],
                rtol=0,
                atol=0,
            )

    def test_wide_head_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-tiny-widehead"
        source = UNeXt2WideHeadVirtualStainer(condition_mode="none")
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2WideHeadVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )

    def test_marker_decoder_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-tiny-marker-decoders"
        source = UNeXt2MarkerDecoderVirtualStainer(
            out_channels=4,
            condition_mode="none",
        )
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68", "CD45RO", "HLA-DR", "Vimentin"),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2MarkerDecoderVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )

    def test_cross_marker_feedback_starts_as_exact_control_and_trains(self) -> None:
        torch.manual_seed(2070)
        control = UNeXt2VirtualStainer(
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        torch.manual_seed(2070)
        candidate = build_model(
            "unext2-fcmae-tiny-cross-marker-feedback",
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(candidate, UNeXt2CrossMarkerFeedbackVirtualStainer)
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        control.eval()
        candidate.eval()
        torch.testing.assert_close(
            candidate(inputs, condition),
            control(inputs, condition),
            rtol=0,
            atol=0,
        )
        self.assertEqual(candidate.feedback_head[-1].weight.detach().abs().sum(), 0)
        candidate.train()
        candidate(inputs, condition).mean().backward()
        self.assertGreater(
            float(candidate.feedback_head[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_cross_marker_feedback_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-tiny-cross-marker-feedback"
        source = UNeXt2CrossMarkerFeedbackVirtualStainer(
            out_channels=4,
            condition_mode="none",
        )
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68", "CD45RO", "HLA-DR", "Vimentin"),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2CrossMarkerFeedbackVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )

    def test_dapi_detail_residual_starts_as_exact_control_and_trains(self) -> None:
        torch.manual_seed(2071)
        control = UNeXt2VirtualStainer(
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        torch.manual_seed(2071)
        candidate = build_model(
            "unext2-fcmae-tiny-dapi-detail-residual",
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(candidate, UNeXt2DapiDetailResidualVirtualStainer)
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        control.eval()
        candidate.eval()
        torch.testing.assert_close(
            candidate(inputs, condition),
            control(inputs, condition),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate._high_pass(inputs).mean(dim=(-2, -1)),
            torch.zeros(1, 1),
            atol=0.05,
            rtol=0,
        )
        self.assertEqual(candidate.detail_head[-1].weight.detach().abs().sum(), 0)
        candidate.train()
        candidate(inputs, condition).mean().backward()
        self.assertGreater(
            float(candidate.detail_head[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_dapi_detail_residual_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-tiny-dapi-detail-residual"
        source = UNeXt2DapiDetailResidualVirtualStainer(
            out_channels=4,
            condition_mode="none",
        )
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68", "CD45RO", "HLA-DR", "Vimentin"),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2DapiDetailResidualVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )

    def test_shallow_dapi_skip_starts_as_exact_control_and_trains(self) -> None:
        torch.manual_seed(2072)
        control = UNeXt2VirtualStainer(
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        torch.manual_seed(2072)
        candidate = build_model(
            "unext2-fcmae-tiny-shallow-dapi-skip",
            out_channels=4,
            condition_mode="spatial-thumbnail-film",
        )
        self.assertIsInstance(candidate, UNeXt2ShallowDapiSkipVirtualStainer)
        inputs = torch.rand(1, 1, 32, 32)
        condition = torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION)
        control.eval()
        candidate.eval()
        torch.testing.assert_close(
            candidate(inputs, condition),
            control(inputs, condition),
            rtol=0,
            atol=0,
        )
        self.assertEqual(candidate.shallow_projection.weight.detach().abs().sum(), 0)
        candidate.train()
        candidate(inputs, condition).mean().backward()
        self.assertGreater(
            float(candidate.shallow_projection.weight.grad.abs().sum()),
            0.0,
        )

    def test_shallow_dapi_skip_checkpoint_roundtrip_is_strict(self) -> None:
        architecture = "unext2-fcmae-tiny-shallow-dapi-skip"
        source = UNeXt2ShallowDapiSkipVirtualStainer(
            out_channels=4,
            condition_mode="none",
        )
        specification = model_specification(architecture)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68", "CD45RO", "HLA-DR", "Vimentin"),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": architecture,
                "model_spec": specification,
                "condition_mode": "none",
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertIsInstance(restored, UNeXt2ShallowDapiSkipVirtualStainer)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )

    def test_whole_roi_checkpoint_roundtrip_is_strict(self) -> None:
        source = UNeXt2VirtualStainer(
            condition_mode=SPATIAL_WHOLE_ROI_CONDITION_MODE
        )
        specification = model_specification("unext2-fcmae-tiny")
        conditioning = spatial_whole_roi_conditioning_specification()
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": conditioning,
            "args": {
                "model_arch": "unext2-fcmae-tiny",
                "model_spec": specification,
                "condition_mode": SPATIAL_WHOLE_ROI_CONDITION_MODE,
                "conditioning_spec": conditioning,
            },
        }
        restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertEqual(restored.condition_mode, SPATIAL_WHOLE_ROI_CONDITION_MODE)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                restored.state_dict()[key],
                value,
                rtol=0,
                atol=0,
            )


class FCMAEScheduleTests(unittest.TestCase):
    def test_gradient_accumulation_preserves_sample_weighting(self) -> None:
        self.assertEqual(
            [
                gradient_accumulation_sample_scale(index, 4, size, 2, 10)
                for index, size in enumerate((4, 4, 2))
            ],
            [0.5, 0.5, 1.0],
        )

    def test_cuda_rng_states_are_validated_and_restored_from_cpu(self) -> None:
        source = torch.arange(16, dtype=torch.uint8)
        restored = cuda_rng_states_for_restore([source])
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].device.type, "cpu")
        torch.testing.assert_close(restored[0], source, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "ByteTensors"):
            cuda_rng_states_for_restore([source.float()])

    def test_warmup_cosine_endpoints(self) -> None:
        self.assertAlmostEqual(
            learning_rate_for_epoch(1, 100, 10, 2e-4, 1e-6),
            2e-5,
        )
        self.assertAlmostEqual(
            learning_rate_for_epoch(10, 100, 10, 2e-4, 1e-6),
            2e-4,
        )
        self.assertAlmostEqual(
            learning_rate_for_epoch(100, 100, 10, 2e-4, 1e-6),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
