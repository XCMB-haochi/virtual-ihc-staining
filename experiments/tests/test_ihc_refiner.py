from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from ihc_refiner import (
    CachedOOFDataset,
    CrossFitRefiner,
    build_refiner_features,
    build_refiner_from_checkpoint,
    predict_refiner_with_tta,
    refiner_feature_specification,
)
from refiner_protocol import load_refiner_protocol, validate_refiner_protocol
from train_refiner import prepare_cuda_runtime


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round37_cv5_crossfit_refiner.json"


class RefinerProtocolTests(unittest.TestCase):
    def test_round37_protocol_is_valid_and_matches_round35(self) -> None:
        protocol = load_refiner_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["baseline_weights"]["thumbnailbase32"], 0.7)
        self.assertEqual(
            protocol["baseline_weights"]["thumbnailbase48continue"], 0.3
        )
        self.assertEqual(protocol["training"]["workers"], 4)
        self.assertEqual(protocol["training"]["selection_tta"], "d4")

    def test_protocol_rejects_weight_or_architecture_drift(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        invalid_weight = copy.deepcopy(protocol)
        invalid_weight["baseline_weights"]["thumbnailbase32"] = 0.6
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_refiner_protocol(invalid_weight)

        invalid_width = copy.deepcopy(protocol)
        invalid_width["model"]["width"] = 47
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            validate_refiner_protocol(invalid_width)


class RefinerModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe_order = ("base32", "base48")
        self.weights = {"base32": 0.7, "base48": 0.3}
        self.dapi = torch.rand(2, 1, 32, 32)
        self.predictions = {
            "base32": torch.rand(2, 1, 32, 32),
            "base48": torch.rand(2, 1, 32, 32),
        }
        self.features = build_refiner_features(
            self.dapi, self.predictions, self.recipe_order, self.weights
        )

    def test_feature_order_and_weighted_baseline(self) -> None:
        specification = refiner_feature_specification(self.recipe_order)
        self.assertEqual(tuple(self.features.shape), (2, 5, 32, 32))
        expected = self.predictions["base32"] * 0.7 + self.predictions["base48"] * 0.3
        torch.testing.assert_close(
            self.features[:, specification["baseline_channel"] : 4],
            expected,
        )
        torch.testing.assert_close(
            self.features[:, 4:5],
            self.predictions["base48"] - self.predictions["base32"],
        )

    def test_zero_initialized_model_is_exact_baseline_and_trains(self) -> None:
        model = CrossFitRefiner(
            input_channels=5,
            baseline_channel=3,
            width=8,
            dilations=(1, 2, 1),
            residual_scale=0.25,
        )
        result = model(self.features)
        self.assertTrue(torch.equal(result, self.features[:, 3:4]))
        result.mean().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)

    def test_d4_tta_and_checkpoint_roundtrip(self) -> None:
        model = CrossFitRefiner(
            input_channels=5,
            baseline_channel=3,
            width=8,
            dilations=(1,),
            residual_scale=0.2,
        )
        direct = model(self.features)
        d4 = predict_refiner_with_tta(model, self.features, "d4")
        torch.testing.assert_close(d4, direct, rtol=0, atol=2e-7)
        restored = build_refiner_from_checkpoint(
            {"model_spec": model.specification(), "model": model.state_dict()},
            torch.device("cpu"),
        )
        torch.testing.assert_close(restored(self.features), direct)

    def test_cuda_peak_reset_uses_current_visible_device(self) -> None:
        device = torch.device("cuda:0")
        with (
            patch("train_refiner.torch.cuda.set_device") as set_device,
            patch("train_refiner.torch.cuda.reset_peak_memory_stats") as reset,
        ):
            prepare_cuda_runtime(device)
        set_device.assert_called_once_with(device)
        reset.assert_called_once_with()


class CachedOOFDatasetTests(unittest.TestCase):
    def create_fold(
        self,
        root: Path,
        fold: str,
        roi: str,
        base32: float,
        base48: float,
    ) -> dict[str, object]:
        directory = root / fold
        directory.mkdir()
        name = f"{roi}_00_00.jpg"
        np.save(directory / "dapi.npy", np.full((1, 256, 256), 40, np.uint8))
        np.save(directory / "target.npy", np.full((1, 256, 256), 50, np.uint8))
        np.save(
            directory / "prediction_base32.npy",
            np.full((1, 256, 256), base32, np.float32),
        )
        np.save(
            directory / "prediction_base48.npy",
            np.full((1, 256, 256), base48, np.float32),
        )
        manifest = {
            "version": 1,
            "fold": fold,
            "val_rois": [roi],
            "images": 1,
            "names": [name],
            "recipe_order": ["base32", "base48"],
            "baseline_weights": {"base32": 0.7, "base48": 0.3},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {"name": fold, "val_rois": [roi], "images": 1}

    def test_dataset_reads_only_selected_folds_and_builds_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folds = [
                self.create_fold(root, "fold0", "ROI000", 20.0, 40.0),
                self.create_fold(root, "fold1", "ROI001", 60.0, 80.0),
            ]
            root_manifest = {
                "version": 1,
                "recipe_order": ["base32", "base48"],
                "baseline_weights": {"base32": 0.7, "base48": 0.3},
                "folds": folds,
            }
            (root / "manifest.json").write_text(
                json.dumps(root_manifest), encoding="utf-8"
            )
            dataset = CachedOOFDataset(
                root,
                ["fold1"],
                ["base32", "base48"],
                {"base32": 0.7, "base48": 0.3},
            )
            self.assertEqual(len(dataset), 1)
            features, target, name = dataset[0]
            self.assertEqual(name, "ROI001_00_00.jpg")
            self.assertEqual(tuple(features.shape), (5, 256, 256))
            self.assertEqual(tuple(target.shape), (1, 256, 256))
            self.assertAlmostEqual(float(features[3, 0, 0]), 66.0 / 255.0)
            with self.assertRaisesRegex(ValueError, "unique folds"):
                CachedOOFDataset(
                    root,
                    ["fold0", "fold0"],
                    ["base32", "base48"],
                    {"base32": 0.7, "base48": 0.3},
                )


if __name__ == "__main__":
    unittest.main()
