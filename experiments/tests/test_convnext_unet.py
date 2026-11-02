from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ConvNeXt_Tiny_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import ihc_convnext_unet
import train_unet
from ihc_convnext_unet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ConvNeXtTinyUNet,
    initialize_convnext_tiny_encoder_imagenet1k_v1,
)
from ihc_torch import (
    build_model,
    build_model_from_checkpoint,
    checkpoint_model_specification,
    model_specification,
)


class ConvNeXtTinyUNetTests(unittest.TestCase):
    def test_output_contract_normalization_and_shallow_skip(self) -> None:
        model = ConvNeXtTinyUNet(out_channels=2).eval()
        source = torch.rand(1, 1, 65, 73)
        encoder_input: dict[str, torch.Tensor] = {}

        def capture_encoder_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            encoder_input["value"] = inputs[0].detach().clone()

        hook = model.encoder[0].register_forward_pre_hook(capture_encoder_input)

        with torch.no_grad():
            result = model(source)
        hook.remove()

        self.assertEqual(tuple(result.shape), (1, 2, 65, 73))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        torch.testing.assert_close(
            model.imagenet_mean.flatten(), torch.tensor(IMAGENET_MEAN)
        )
        torch.testing.assert_close(
            model.imagenet_std.flatten(), torch.tensor(IMAGENET_STD)
        )
        expected_encoder_input = (
            source.repeat(1, 3, 1, 1) - model.imagenet_mean
        ) / model.imagenet_std
        torch.testing.assert_close(encoder_input["value"], expected_encoder_input)
        self.assertEqual(model.shallow_skip.block[0].in_channels, 1)
        torch.testing.assert_close(
            model.head.bias, torch.full_like(model.head.bias, -2.0)
        )

    def test_rejects_invalid_input_and_conditioning(self) -> None:
        model = ConvNeXtTinyUNet()
        with self.assertRaisesRegex(ValueError, "does not support conditioning"):
            model(torch.rand(1, 1, 32, 32), torch.zeros(1, 1))
        with self.assertRaisesRegex(ValueError, "expects input shape"):
            model(torch.rand(1, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, ">= 32"):
            model(torch.rand(1, 1, 31, 32))

    def test_backward_reaches_encoder_shallow_skip_and_head(self) -> None:
        model = ConvNeXtTinyUNet()
        model(torch.rand(1, 1, 32, 32)).mean().backward()

        checked = (
            model.encoder[0][0].weight,
            model.shallow_skip.block[0].weight,
            model.head.weight,
        )
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_explicit_encoder_initialization_is_offline_patchable(self) -> None:
        model = ConvNeXtTinyUNet()
        encoder_state = model.encoder.state_dict()
        changed_key = next(iter(encoder_state))
        changed_value = torch.full_like(encoder_state[changed_key], 0.125)
        full_state = {
            f"features.{key}": (changed_value if key == changed_key else value)
            for key, value in encoder_state.items()
        }
        provenance = {
            "version": 1,
            "mode": "torchvision-imagenet",
            "weights_enum": (
                "torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
            ),
            "weights_url": "https://download.pytorch.org/example.pth",
            "cache_file": "cache/example.pth",
            "cache_file_sha256": "a" * 64,
        }
        with mock.patch.object(
            ihc_convnext_unet,
            "_load_convnext_tiny_imagenet1k_v1",
            return_value=(full_state, provenance),
        ):
            report = initialize_convnext_tiny_encoder_imagenet1k_v1(model)

        torch.testing.assert_close(model.encoder.state_dict()[changed_key], changed_value)
        self.assertEqual(report["cache_file_sha256"], "a" * 64)
        self.assertEqual(report["state_tensors_loaded"], len(encoder_state))

    def test_weight_loader_records_full_cached_file_sha256_without_network(self) -> None:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        cache_name = Path(weights.url).name
        payload = b"offline-convnext-test-weights"
        with tempfile.TemporaryDirectory() as temporary:
            hub_dir = Path(temporary)
            cache_file = hub_dir / "checkpoints" / cache_name
            cache_file.parent.mkdir()
            cache_file.write_bytes(payload)
            with (
                mock.patch.object(
                    weights,
                    "get_state_dict",
                    return_value={"features.test": torch.ones(1)},
                ) as get_state_dict,
                mock.patch.object(
                    ihc_convnext_unet.torch.hub, "get_dir", return_value=str(hub_dir)
                ),
            ):
                state, report = ihc_convnext_unet._load_convnext_tiny_imagenet1k_v1()

        get_state_dict.assert_called_once_with(progress=True, check_hash=True)
        self.assertIn("features.test", state)
        self.assertEqual(report["weights_url"], weights.url)
        self.assertEqual(
            report["cache_file_sha256"], hashlib.sha256(payload).hexdigest()
        )
        self.assertEqual(len(report["cache_file_sha256"]), 64)


class ConvNeXtFactoryAndOptimizerTests(unittest.TestCase):
    def test_encoder_init_is_rejected_for_checkpoint_initialized_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = [
                "train_unet.py",
                str(root / "missing-dataset"),
                "--output-dir",
                str(root / "run"),
                "--model-arch",
                "convnext-tiny-unet",
                "--encoder-init",
                "imagenet1k-v1",
                "--init-checkpoint",
                str(root / "stage1.pt"),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(train_unet, "resolve_data_root") as resolve_data_root,
            ):
                with self.assertRaisesRegex(ValueError, "fresh stage1"):
                    train_unet.main()
            resolve_data_root.assert_not_called()

    def test_checkpoint_factory_never_requests_torchvision_weights(self) -> None:
        specification = model_specification("convnext-tiny-unet")
        checkpoint = {
            "model": {},
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": "convnext-tiny-unet",
                "model_spec": specification,
                "condition_mode": "none",
                "encoder_init": "imagenet1k-v1",
            },
        }
        original_constructor = ihc_convnext_unet.convnext_tiny
        with (
            mock.patch.object(
                ihc_convnext_unet,
                "_load_convnext_tiny_imagenet1k_v1",
                side_effect=AssertionError("weight loader must not run"),
            ),
            mock.patch.object(
                ihc_convnext_unet,
                "convnext_tiny",
                wraps=original_constructor,
            ) as constructor,
            mock.patch.object(ConvNeXtTinyUNet, "load_state_dict") as load_state_dict,
        ):
            loaded = build_model_from_checkpoint(checkpoint, torch.device("cpu"))

        self.assertIsInstance(loaded, ConvNeXtTinyUNet)
        constructor.assert_called_once_with(weights=None)
        load_state_dict.assert_called_once_with({}, strict=True)
        self.assertEqual(checkpoint_model_specification(checkpoint), specification)

    def test_checkpoint_factory_strict_roundtrip_is_offline(self) -> None:
        specification = model_specification("convnext-tiny-unet")
        source = ConvNeXtTinyUNet(out_channels=1)
        checkpoint = {
            "model": source.state_dict(),
            "markers": ("CD68",),
            "model_spec": specification,
            "conditioning": {"version": 1, "mode": "none"},
            "args": {
                "model_arch": "convnext-tiny-unet",
                "model_spec": specification,
                "condition_mode": "none",
                "encoder_init": "imagenet1k-v1",
            },
        }

        with mock.patch.object(
            ihc_convnext_unet,
            "_load_convnext_tiny_imagenet1k_v1",
            side_effect=AssertionError("checkpoint restore must stay offline"),
        ):
            restored = build_model_from_checkpoint(checkpoint, torch.device("cpu"))

        for key in ("encoder.0.0.weight", "head.bias", "imagenet_mean"):
            torch.testing.assert_close(
                restored.state_dict()[key], source.state_dict()[key]
            )

    def test_differential_lr_groups_cover_every_parameter_and_cosine_together(self) -> None:
        model = build_model("convnext-tiny-unet", out_channels=1)
        optimizer, report = train_unet.build_optimizer(
            model,
            "convnext-tiny-unet",
            learning_rate=3e-4,
            encoder_learning_rate=3e-5,
            weight_decay=1e-4,
        )

        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["decoder_head", "encoder"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [3e-4, 3e-5]
        )
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped), len(list(model.parameters())))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )
        self.assertEqual(
            sum(item["parameters"] for item in report),
            sum(parameter.numel() for parameter in model.parameters()),
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=4)
        optimizer.step()
        scheduler.step()
        decoder_lr, encoder_lr = scheduler.get_last_lr()
        self.assertAlmostEqual(decoder_lr / encoder_lr, 10.0)

    def test_legacy_optimizer_remains_a_single_parameter_group(self) -> None:
        model = build_model("unet", out_channels=1, base_channels=4)
        optimizer, report = train_unet.build_optimizer(
            model,
            "unet",
            learning_rate=3e-4,
            encoder_learning_rate=None,
            weight_decay=1e-4,
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertNotIn("group_name", optimizer.param_groups[0])
        self.assertEqual(report[0]["name"], "all")


if __name__ == "__main__":
    unittest.main()
