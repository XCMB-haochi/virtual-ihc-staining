from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from average_checkpoints import average_checkpoints
from ihc_torch import UNet, build_unet_from_checkpoint


def checkpoint_for(model: nn.Module, epoch: int, value: float) -> dict[str, object]:
    state = OrderedDict(
        (name, torch.full_like(tensor, value))
        for name, tensor in model.state_dict().items()
    )
    args = {
        "dataset": Path("dataset"),
        "base_channels": 8,
        "block_type": "residual",
        "markers": "CD68",
        "seed": 2026,
        "output_dir": "runs/same_run",
    }
    return {
        "epoch": epoch,
        "model": state,
        "best_score": 70.0 + epoch,
        "markers": ("CD68",),
        "base_channels": 8,
        "block_type": "residual",
        "args": args,
        "selection_psnr_aggregation": "mean-image",
    }


class AverageCheckpointTests(unittest.TestCase):
    def save_inputs(
        self, root: Path, first: dict[str, object], second: dict[str, object]
    ) -> tuple[Path, Path]:
        first_path = root / "epoch_001.pt"
        second_path = root / "epoch_002.pt"
        torch.save(first, first_path)
        torch.save(second, second_path)
        return first_path, second_path

    def test_averages_in_float64_and_writes_compact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = UNet(out_channels=1, base_channels=8, block_type="residual")
            paths = self.save_inputs(
                root, checkpoint_for(model, 1, 0.1), checkpoint_for(model, 2, 0.2)
            )
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            output = root / "soup.pt"

            report = average_checkpoints(paths, [1.0, 2.0], output)
            result = torch.load(output, map_location="cpu", weights_only=False)

            self.assertEqual(
                set(result),
                {
                    "epoch",
                    "model",
                    "best_score",
                    "markers",
                    "base_channels",
                    "block_type",
                    "args",
                    "selection_psnr_aggregation",
                    "soup",
                },
            )
            self.assertEqual(result["epoch"], 2)
            self.assertIsNone(result["best_score"])
            self.assertEqual(result["args"]["dataset"], Path("dataset"))
            self.assertEqual(result["soup"]["buffer_policy"], "error")
            self.assertEqual(
                [item["sha256"] for item in result["soup"]["inputs"]], hashes
            )
            self.assertAlmostEqual(result["soup"]["inputs"][0]["weight"], 1 / 3)
            self.assertAlmostEqual(result["soup"]["inputs"][1]["weight"], 2 / 3)
            expected = torch.tensor(0.1, dtype=torch.float64) / 3 + torch.tensor(
                0.2, dtype=torch.float64
            ) * 2 / 3
            for tensor in result["model"].values():
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(
                    torch.equal(
                        tensor,
                        torch.full_like(tensor, expected.to(torch.float32)),
                    )
                )
            reloaded = UNet(out_channels=1, base_channels=8, block_type="residual")
            reloaded.load_state_dict(result["model"], strict=True)
            self.assertEqual(
                report["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )

    def test_rejects_different_args_and_duplicate_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = UNet(out_channels=1, base_channels=8)
            first = checkpoint_for(model, 1, 0.1)
            second = checkpoint_for(model, 2, 0.2)
            second["args"] = {**second["args"], "learning_rate": 1e-4}
            paths = self.save_inputs(root, first, second)
            with self.assertRaisesRegex(ValueError, "not from the same trajectory"):
                average_checkpoints(paths, [1, 1], root / "args.pt")

            second = checkpoint_for(model, 1, 0.2)
            paths = self.save_inputs(root, first, second)
            with self.assertRaisesRegex(ValueError, "unique epochs"):
                average_checkpoints(paths, [1, 1], root / "epochs.pt")

    def test_conditioned_soup_preserves_spec_and_rejects_different_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = UNet(
                out_channels=1,
                base_channels=8,
                condition_mode="roi-mean-film",
            )
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
            first = checkpoint_for(model, 1, 0.1)
            second = checkpoint_for(model, 2, 0.2)
            for checkpoint in (first, second):
                checkpoint["conditioning"] = spec
                checkpoint["args"] = {
                    **checkpoint["args"],
                    "condition_mode": "roi-mean-film",
                    "conditioning_spec": spec,
                }
            paths = self.save_inputs(root, first, second)
            output = root / "conditioned.pt"

            average_checkpoints(paths, [1, 1], output)
            result = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(result["conditioning"], spec)
            build_unet_from_checkpoint(result, torch.device("cpu"))

            incompatible = {
                **spec,
                "normalization": {
                    **spec["normalization"],
                    "center": 31.0,
                },
            }
            second["conditioning"] = incompatible
            second["args"] = {
                **second["args"],
                "conditioning_spec": incompatible,
            }
            torch.save(second, paths[1])
            with self.assertRaisesRegex(ValueError, "architecture metadata differs"):
                average_checkpoints(paths, [1, 1], root / "incompatible.pt")

    def test_rejects_invalid_weights_state_and_input_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = UNet(out_channels=1, base_channels=8)
            first = checkpoint_for(model, 1, 0.1)
            second = checkpoint_for(model, 2, 0.2)
            paths = self.save_inputs(root, first, second)
            for weights in ([1.0], [1.0, 0.0], [1.0, float("nan")]):
                with self.assertRaises(ValueError):
                    average_checkpoints(paths, weights, root / "invalid.pt")
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                average_checkpoints(paths, [1, 1], paths[0])

            second["model"].pop(next(iter(second["model"])))
            torch.save(second, paths[1])
            with self.assertRaisesRegex(ValueError, "model keys differ"):
                average_checkpoints(paths, [1, 1], root / "state.pt")

    def test_buffer_policy_is_explicit_and_latest_uses_highest_epoch(self) -> None:
        class BufferedModel(nn.Module):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.register_buffer("counter", torch.zeros(1, dtype=torch.int64))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = BufferedModel()
            first = checkpoint_for(model, 3, 1.0)
            second = checkpoint_for(model, 7, 3.0)
            first["model"]["counter"] = torch.tensor([11], dtype=torch.int64)
            second["model"]["counter"] = torch.tensor([29], dtype=torch.int64)
            paths = self.save_inputs(root, first, second)

            def build_buffered(
                checkpoint: dict[str, object], _device: torch.device
            ) -> BufferedModel:
                buffered = BufferedModel()
                buffered.load_state_dict(checkpoint["model"], strict=True)
                return buffered

            with mock.patch(
                "average_checkpoints.build_model_from_checkpoint",
                side_effect=build_buffered,
            ):
                with self.assertRaisesRegex(ValueError, "contains buffers"):
                    average_checkpoints(paths, [1, 1], root / "error.pt")
                average_checkpoints(
                    paths, [1, 3], root / "latest.pt", buffer_policy="latest"
                )
                average_checkpoints(
                    paths, [1, 3], root / "average.pt", buffer_policy="average"
                )

            result = torch.load(
                root / "latest.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(result["model"]["counter"].item(), 29)
            self.assertAlmostEqual(result["model"]["weight"].item(), 2.5)
            averaged = torch.load(
                root / "average.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(averaged["model"]["counter"].item(), 29)
            self.assertAlmostEqual(averaged["model"]["weight"].item(), 2.5)

    def test_failed_save_preserves_existing_output_and_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = UNet(out_channels=1, base_channels=8)
            paths = self.save_inputs(
                root, checkpoint_for(model, 1, 0.1), checkpoint_for(model, 2, 0.2)
            )
            output = root / "soup.pt"
            output.write_bytes(b"existing")

            with mock.patch("average_checkpoints.torch.save", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    average_checkpoints(paths, [1, 1], output)

            self.assertEqual(output.read_bytes(), b"existing")
            self.assertEqual(list(root.glob(".soup.pt.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
