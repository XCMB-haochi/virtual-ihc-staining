from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ihc_unext2 import DapiFCMAE, unext2_encoder_specification


MULTIMARKER_FCMAE_MARKERS = (
    "DAPI",
    "CD68",
    "CD45RO",
    "HLA-DR",
    "Vimentin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_multimarker_fcmae_teacher(
    checkpoint_path: Path,
    expected_sha256: str,
    expected_val_rois: set[str],
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint_path = checkpoint_path.resolve()
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Feature-guidance teacher SHA-256 mismatch: {actual_sha256}."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Feature-guidance teacher checkpoint must be a mapping.")
    if checkpoint.get("version") != 1 or checkpoint.get("task") != "dapi-fcmae-pretraining":
        raise ValueError("Feature-guidance teacher is not an FCMAE encoder artifact.")
    if checkpoint.get("encoder_spec") != unext2_encoder_specification("tiny"):
        raise ValueError("Feature-guidance teacher must use the Tiny encoder.")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Feature-guidance teacher lacks configuration metadata.")
    markers = tuple(config.get("pretraining_markers", ()))
    if markers != MULTIMARKER_FCMAE_MARKERS:
        raise ValueError(
            "Feature-guidance teacher must be the fixed five-modality FCMAE encoder."
        )
    val_rois = set(checkpoint.get("val_rois", ()))
    if val_rois != expected_val_rois:
        raise ValueError(
            "Feature-guidance teacher validation exclusions do not match the fold."
        )
    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping):
        raise ValueError("Feature-guidance teacher lacks encoder parameters.")

    # Constructing an immediately overwritten frozen teacher must not advance
    # the student's CPU random stream relative to a no-guidance control.
    with torch.random.fork_rng(devices=[]):
        teacher = DapiFCMAE("tiny").encoder
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval().requires_grad_(False)
    report = {
        "version": 1,
        "mode": "frozen-fold-local-multimarker-fcmae-perceptual",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_sha256,
        "pretraining_epoch": int(checkpoint["epoch"]),
        "pretraining_markers": list(markers),
        "excluded_val_rois": sorted(val_rois),
        "parameters": sum(parameter.numel() for parameter in teacher.parameters()),
    }
    return teacher, report


def center_crop_square(inputs: torch.Tensor, side: int) -> torch.Tensor:
    if inputs.ndim != 4 or side <= 0:
        raise ValueError("Feature-guidance inputs must be BCHW with a positive crop.")
    height, width = inputs.shape[-2:]
    if side > height or side > width:
        raise ValueError(
            f"Feature-guidance crop {side} exceeds input shape {(height, width)}."
        )
    top = (height - side) // 2
    left = (width - side) // 2
    return inputs[..., top : top + side, left : left + side]


class FrozenFCMAEPerceptualLoss(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        *,
        stage: int = 4,
        crop_size: int = 256,
        batch_samples: int = 4,
    ) -> None:
        super().__init__()
        if not 1 <= stage <= 4:
            raise ValueError("Feature-guidance stage must be from 1 to 4.")
        if crop_size <= 0 or crop_size % 16:
            raise ValueError("Feature-guidance crop size must be divisible by 16.")
        if batch_samples <= 0:
            raise ValueError("Feature-guidance batch samples must be positive.")
        self.teacher = teacher.eval().requires_grad_(False)
        self.stage = int(stage)
        self.crop_size = int(crop_size)
        self.batch_samples = int(batch_samples)

    def train(self, mode: bool = True) -> FrozenFCMAEPerceptualLoss:
        super().train(False)
        self.teacher.eval()
        return self

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError(
                "Feature-guidance prediction and target must have equal BCHW shapes."
            )
        if prediction.shape[1] != 1:
            raise ValueError("Feature-guidance loss accepts one selected marker channel.")
        samples = min(self.batch_samples, prediction.shape[0])
        prediction = center_crop_square(prediction[:samples], self.crop_size)
        target = center_crop_square(target[:samples], self.crop_size)
        self.teacher.eval()
        with torch.no_grad():
            target_features, _ = self.teacher(target)
            reference = F.normalize(
                target_features[self.stage - 1].float(), dim=1, eps=1e-6
            )
        prediction_features, _ = self.teacher(prediction)
        predicted = F.normalize(
            prediction_features[self.stage - 1].float(), dim=1, eps=1e-6
        )
        return (1.0 - (predicted * reference).sum(dim=1)).mean()


def relational_distance_loss(student: torch.Tensor, teacher: torch.Tensor, *, robust_scale: bool = False) -> torch.Tensor:
    """RKD pair loss normalized by mean distance, or lower median when requested."""
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError("Relational features must be two batch-aligned matrices.")
    if student.shape[0] < 3:
        # With two samples normalization makes the single distance always one.
        return student.sum() * 0.0
    distances = torch.pdist(student.float())
    with torch.no_grad():
        reference = torch.pdist(teacher.detach().float())
        scale = reference.median() if robust_scale else reference.mean()
        reference = reference / scale.clamp_min(1e-6)
    scale = distances.median() if robust_scale else distances.mean()
    distances = distances / scale.clamp_min(1e-6)
    return F.smooth_l1_loss(distances, reference)


class FrozenFCMAERelationalLoss(FrozenFCMAEPerceptualLoss):
    """Transfer pooled CD68 relations directly to the DAPI encoder, without a head."""
    def __init__(self, teacher: nn.Module, *, robust_scale: bool = False, **kwargs) -> None:
        super().__init__(teacher, **kwargs)
        self.robust_scale = bool(robust_scale)
        if self.stage != 4 or self.batch_samples < 3:
            raise ValueError("Encoder relational guidance requires stage4 and at least 3 samples.")

    def forward(self, encoder_features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 4 or target.shape[1] != 1 or encoder_features.shape[0] != target.shape[0]:
            raise ValueError("Relational guidance requires batch-aligned encoder features and one marker.")
        samples = min(self.batch_samples, target.shape[0])
        student = center_crop_square(encoder_features[:samples], self.crop_size // 16)
        student = student.float().mean(dim=(-2, -1))
        target = center_crop_square(target[:samples], self.crop_size)
        self.teacher.eval()
        with torch.no_grad():
            features, _ = self.teacher(target)
            reference = features[3].float().mean(dim=(-2, -1))
        return relational_distance_loss(student, reference, robust_scale=self.robust_scale)


def build_feature_guidance(teacher: nn.Module, mode: str, **kwargs) -> FrozenFCMAEPerceptualLoss:
    if mode == "perceptual":
        return FrozenFCMAEPerceptualLoss(teacher, **kwargs)
    if mode in {"encoder-relational", "encoder-relational-robust"}:
        return FrozenFCMAERelationalLoss(
            teacher, robust_scale=mode == "encoder-relational-robust", **kwargs
        )
    raise ValueError(f"Unknown feature-guidance mode: {mode}")
