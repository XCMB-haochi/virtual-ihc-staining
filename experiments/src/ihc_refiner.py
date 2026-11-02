from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ihc_data import parse_patch_name
from ihc_torch import apply_d4_transform, invert_d4_transform


class RefinerResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.conv1(inputs)))
        residual = self.norm2(self.conv2(residual))
        return self.activation(inputs + residual)


class CrossFitRefiner(nn.Module):
    def __init__(
        self,
        input_channels: int,
        baseline_channel: int,
        width: int = 48,
        dilations: Sequence[int] = (1, 2, 4, 8, 4, 2, 1),
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if not 0 <= baseline_channel < input_channels:
            raise ValueError("baseline_channel is outside the input tensor.")
        if width <= 0 or width % 8:
            raise ValueError("width must be positive and divisible by eight.")
        if not dilations or any(type(value) is not int or value <= 0 for value in dilations):
            raise ValueError("dilations must contain positive integers.")
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be in (0, 1].")
        self.input_channels = input_channels
        self.baseline_channel = baseline_channel
        self.width = width
        self.dilations = tuple(dilations)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(RefinerResidualBlock(width, dilation) for dilation in self.dilations)
        )
        self.head = nn.Conv2d(width, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def specification(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "baseline_channel": self.baseline_channel,
            "width": self.width,
            "dilations": list(self.dilations),
            "residual_scale": self.residual_scale,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected NCHW input with {self.input_channels} channels, got "
                f"{tuple(features.shape)}."
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Refiner inputs must be finite.")
        baseline = features[:, self.baseline_channel : self.baseline_channel + 1]
        residual = self.residual_scale * torch.tanh(
            self.head(self.blocks(self.stem(features)))
        )
        return torch.clamp(baseline + residual, 0.0, 1.0)


def build_refiner_features(
    dapi: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    recipe_order: Sequence[str],
    weights: dict[str, float],
) -> torch.Tensor:
    if len(recipe_order) != 2 or set(recipe_order) != set(weights):
        raise ValueError("Refiner features require two ordered, weighted recipes.")
    if dapi.ndim not in {3, 4} or dapi.shape[-3] != 1:
        raise ValueError("DAPI must have shape CHW or NCHW with one channel.")
    ordered = []
    for name in recipe_order:
        if name not in predictions:
            raise ValueError(f"Missing refiner prediction: {name}.")
        prediction = predictions[name]
        if prediction.shape != dapi.shape:
            raise ValueError(
                f"Prediction {name} shape {tuple(prediction.shape)} does not match "
                f"DAPI {tuple(dapi.shape)}."
            )
        ordered.append(prediction)
    baseline = sum(
        ordered[index] * float(weights[name])
        for index, name in enumerate(recipe_order)
    )
    disagreement = ordered[1] - ordered[0]
    return torch.cat((dapi, *ordered, baseline, disagreement), dim=-3)


def refiner_feature_specification(recipe_order: Sequence[str]) -> dict[str, object]:
    if len(recipe_order) != 2:
        raise ValueError("Refiner feature specification requires two recipes.")
    return {
        "channels": [
            "dapi",
            f"prediction:{recipe_order[0]}",
            f"prediction:{recipe_order[1]}",
            "weighted_baseline",
            f"difference:{recipe_order[1]}-{recipe_order[0]}",
        ],
        "baseline_channel": 3,
    }


@torch.no_grad()
def predict_refiner_with_tta(
    model: CrossFitRefiner,
    features: torch.Tensor,
    tta: str,
) -> torch.Tensor:
    if tta == "none":
        return model(features)
    if tta != "d4":
        raise ValueError(f"Unsupported refiner TTA: {tta}.")
    result = None
    for transform_index in range(8):
        transformed = apply_d4_transform(features, transform_index)
        prediction = invert_d4_transform(model(transformed), transform_index)
        result = prediction if result is None else result + prediction
    if result is None:
        raise RuntimeError("D4 refiner prediction produced no transforms.")
    return result / 8.0


def build_refiner_from_checkpoint(
    checkpoint: dict[str, object], device: torch.device
) -> CrossFitRefiner:
    specification = checkpoint.get("model_spec")
    if not isinstance(specification, dict):
        raise ValueError("Refiner checkpoint has no model_spec.")
    model = CrossFitRefiner(
        input_channels=int(specification["input_channels"]),
        baseline_channel=int(specification["baseline_channel"]),
        width=int(specification["width"]),
        dilations=tuple(int(value) for value in specification["dilations"]),
        residual_scale=float(specification["residual_scale"]),
    ).to(device)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("Refiner checkpoint has no model state.")
    model.load_state_dict(state, strict=True)
    return model


class CachedOOFDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        fold_names: Sequence[str],
        recipe_order: Sequence[str],
        weights: dict[str, float],
        *,
        augment: bool = False,
        preload: bool = True,
    ) -> None:
        self.cache_root = cache_root.resolve()
        self.recipe_order = tuple(recipe_order)
        self.weights = dict(weights)
        self.augment = augment
        root_manifest_path = self.cache_root / "manifest.json"
        if not root_manifest_path.is_file():
            raise FileNotFoundError(f"Missing OOF cache manifest: {root_manifest_path}")
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        available = {item["name"]: item for item in root_manifest.get("folds", [])}
        if not fold_names or len(set(fold_names)) != len(fold_names):
            raise ValueError("fold_names must contain unique folds.")
        missing = sorted(set(fold_names) - set(available))
        if missing:
            raise ValueError(f"OOF cache is missing folds: {missing}.")
        if tuple(root_manifest.get("recipe_order", ())) != self.recipe_order:
            raise ValueError("OOF cache recipe order does not match the refiner protocol.")
        if root_manifest.get("baseline_weights") != self.weights:
            raise ValueError("OOF cache weights do not match the refiner protocol.")

        self.folds: list[dict[str, object]] = []
        self.offsets: list[int] = []
        total = 0
        all_names: set[str] = set()
        load_mode = None if preload else "r"
        for fold_name in fold_names:
            fold_dir = self.cache_root / fold_name
            manifest_path = fold_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing fold cache manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = manifest.get("names")
            if not isinstance(names, list) or not names:
                raise ValueError(f"Fold {fold_name} has no cached names.")
            if len(names) != len(set(names)) or all_names.intersection(names):
                raise ValueError("OOF cache names must be unique across folds.")
            all_names.update(names)
            expected_rois = set(manifest.get("val_rois", []))
            root_entry = available[fold_name]
            if (
                expected_rois != set(root_entry.get("val_rois", []))
                or len(names) != int(root_entry.get("images", -1))
            ):
                raise ValueError(
                    f"Fold {fold_name} manifest disagrees with the root cache manifest."
                )
            actual_rois = {parse_patch_name(name).roi for name in names}
            if actual_rois != expected_rois:
                raise ValueError(f"Fold {fold_name} names do not match its ROI manifest.")
            dapi = np.load(fold_dir / "dapi.npy", mmap_mode=load_mode)
            target = np.load(fold_dir / "target.npy", mmap_mode=load_mode)
            predictions = {
                name: np.load(
                    fold_dir / f"prediction_{name}.npy", mmap_mode=load_mode
                )
                for name in self.recipe_order
            }
            expected_shape = (len(names), 256, 256)
            if dapi.shape != expected_shape or target.shape != expected_shape:
                raise ValueError(f"Fold {fold_name} has invalid DAPI/target shapes.")
            if dapi.dtype != np.uint8 or target.dtype != np.uint8:
                raise ValueError(f"Fold {fold_name} DAPI/target caches must be uint8.")
            for name, prediction in predictions.items():
                if prediction.shape != expected_shape or prediction.dtype != np.float32:
                    raise ValueError(
                        f"Fold {fold_name} prediction {name} must be float32 "
                        f"with shape {expected_shape}."
                    )
            self.folds.append(
                {
                    "name": fold_name,
                    "names": names,
                    "dapi": dapi,
                    "target": target,
                    "predictions": predictions,
                }
            )
            total += len(names)
            self.offsets.append(total)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        fold_index = bisect.bisect_right(self.offsets, index)
        previous = 0 if fold_index == 0 else self.offsets[fold_index - 1]
        local_index = index - previous
        fold = self.folds[fold_index]
        dapi = torch.from_numpy(
            np.array(fold["dapi"][local_index], copy=True)
        ).float()[None] / 255.0
        predictions = {
            name: torch.from_numpy(
                np.array(fold["predictions"][name][local_index], copy=True)
            ).float()[None]
            / 255.0
            for name in self.recipe_order
        }
        target = torch.from_numpy(
            np.array(fold["target"][local_index], copy=True)
        ).float()[None] / 255.0
        features = build_refiner_features(
            dapi, predictions, self.recipe_order, self.weights
        )
        if self.augment:
            transform_index = int(torch.randint(0, 8, ()).item())
            features = apply_d4_transform(features, transform_index)
            target = apply_d4_transform(target, transform_index)
        return features, target, str(fold["names"][local_index])
