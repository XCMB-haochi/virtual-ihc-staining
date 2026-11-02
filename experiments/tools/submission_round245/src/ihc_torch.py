from __future__ import annotations
import math,random
from pathlib import Path
from typing import Mapping,Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from ihc_data import *
from ihc_spatial import *
from ihc_unext2 import *


def shift_array_edge(
    array: np.ndarray,
    source_offset: tuple[int, int],
) -> np.ndarray:
    """Translate spatial content by sampling a fixed source offset."""

    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("Shifted array must have at least two spatial dimensions.")
    row_offset, col_offset = source_offset
    if type(row_offset) is not int or type(col_offset) is not int:
        raise ValueError("Array source offsets must be integers.")
    padding = max(abs(row_offset), abs(col_offset))
    if padding == 0:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[-2] = (padding, padding)
    pad_width[-1] = (padding, padding)
    padded = np.pad(array, pad_width, mode="edge")
    height, width = array.shape[-2:]
    return padded[
        ...,
        padding + row_offset : padding + row_offset + height,
        padding + col_offset : padding + col_offset + width,
    ]


def batch_condition_tensor(
    names: Sequence[str],
    condition_by_roi: Mapping[str, float | Sequence[float] | np.ndarray] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if condition_by_roi is None:
        return None
    keys = [
        name if name in condition_by_roi else parse_patch_name(name).roi
        for name in names
    ]
    missing = sorted({key for key in keys if key not in condition_by_roi})
    if missing:
        raise ValueError(f"Conditioning values are missing for: {missing}")
    rows = [np.atleast_1d(np.asarray(condition_by_roi[key], dtype=np.float32)) for key in keys]
    dimensions = {row.shape for row in rows}
    if len(dimensions) != 1 or any(row.ndim != 1 for row in rows):
        raise ValueError(f"Conditioning values have inconsistent shapes: {dimensions}")
    values = np.stack(rows)
    if not np.isfinite(values).all():
        raise ValueError("Conditioning values must be finite.")
    return torch.as_tensor(values, device=device, dtype=torch.float32)










def _ssim_uniform_impl(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> torch.Tensor:
    """Match scikit-image's default uniform-window SSIM on normalized images."""
    if window_size <= 1 or window_size % 2 != 1:
        raise ValueError("window_size must be an odd integer greater than one.")
    channels = prediction.shape[1]
    kernel = torch.full(
        (channels, 1, window_size, window_size),
        1.0 / window_size**2,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_prediction = F.conv2d(prediction, kernel, groups=channels)
    mu_target = F.conv2d(target, kernel, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    covariance_normalization = window_size**2 / (window_size**2 - 1)
    variance_prediction = covariance_normalization * (
        F.conv2d(prediction.square(), kernel, groups=channels) - mu_prediction_sq
    )
    variance_target = covariance_normalization * (
        F.conv2d(target.square(), kernel, groups=channels) - mu_target_sq
    )
    covariance = covariance_normalization * (
        F.conv2d(prediction * target, kernel, groups=channels) - mu_product
    )

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_product + c1) * (2 * covariance + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        variance_prediction + variance_target + c2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean(dim=(-2, -1))


def ssim_uniform_per_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> torch.Tensor:
    with torch.amp.autocast(prediction.device.type, enabled=False):
        return _ssim_uniform_impl(prediction.float(), target.float(), window_size)


def apply_d4_transform(inputs: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Apply one indexed D4 transform while preserving the last two axes."""
    if type(transform_index) is not int or not 0 <= transform_index < 8:
        raise ValueError("D4 transform index must be an integer from 0 to 7.")
    rotations = transform_index % 4
    transformed = torch.rot90(inputs, rotations, dims=(-2, -1))
    if transform_index >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def invert_d4_transform(inputs: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Invert the corresponding indexed D4 transform."""
    if type(transform_index) is not int or not 0 <= transform_index < 8:
        raise ValueError("D4 transform index must be an integer from 0 to 7.")
    restored = inputs
    if transform_index >= 4:
        restored = torch.flip(restored, dims=(-1,))
    return torch.rot90(restored, -(transform_index % 4), dims=(-2, -1))


def predict_with_tta(
    model: nn.Module,
    inputs: torch.Tensor,
    mode: str = "none",
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "none":
        return model(inputs) if condition is None else model(inputs, condition)
    if mode != "d4":
        raise ValueError(f"Unsupported TTA mode: {mode}")

    predictions = []
    for rotations in range(4):
        for transform_index in (rotations, rotations + 4):
            transformed = apply_d4_transform(inputs, transform_index)
            transformed_condition = model_condition_d4_transform(
                model,
                condition,
                transform_index,
            )
            transformed_prediction = (
                model(transformed)
                if transformed_condition is None
                else model(transformed, transformed_condition)
            )
            predictions.append(
                invert_d4_transform(transformed_prediction, transform_index)
            )
    return torch.stack(predictions).mean(dim=0)


def center_crop_like(
    inputs: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    target_height, target_width = reference.shape[-2:]
    height, width = inputs.shape[-2:]
    if height < target_height or width < target_width:
        raise ValueError(
            f"Cannot crop tensor with spatial shape {(height, width)} to "
            f"{(target_height, target_width)}."
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return inputs[..., top : top + target_height, left : left + target_width]


class CompetitionLoss(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.7,
        l1_weight: float = 0.2,
        mse_weight: float = 0.1,
        pyramid_weight: float = 0.0,
        target_l1_weight_strength: float = 0.0,
        ssim_mode: str = "uniform7",
        channel_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        total = ssim_weight + l1_weight + mse_weight
        if not np.isclose(total, 1.0):
            raise ValueError(f"Loss weights must sum to 1, got {total}")
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        if ssim_mode not in {"uniform7"}:
            raise ValueError(f"Unsupported SSIM mode: {ssim_mode}")
        self.ssim_mode = ssim_mode
        if pyramid_weight < 0:
            raise ValueError("Pyramid weight must be non-negative.")
        self.pyramid_weight = pyramid_weight
        if not np.isfinite(target_l1_weight_strength) or target_l1_weight_strength < 0:
            raise ValueError("Target L1 weight strength must be finite and non-negative.")
        self.target_l1_weight_strength = float(target_l1_weight_strength)
        normalized_channel_weights = None
        if channel_weights is not None:
            normalized_channel_weights = torch.as_tensor(
                channel_weights,
                dtype=torch.float32,
            ).detach().clone()
            if (
                normalized_channel_weights.ndim != 1
                or not normalized_channel_weights.numel()
            ):
                raise ValueError(
                    "channel_weights must be a non-empty one-dimensional sequence."
                )
            if not bool(
                torch.isfinite(normalized_channel_weights).all()
            ) or not bool((normalized_channel_weights >= 0).all()):
                raise ValueError("channel_weights must be finite and non-negative.")
            total_channel_weight = normalized_channel_weights.sum()
            if not bool(torch.isfinite(total_channel_weight)) or not bool(
                total_channel_weight > 0
            ):
                raise ValueError("channel_weights must have a finite positive sum.")
            normalized_channel_weights = (
                normalized_channel_weights / total_channel_weight
            )
        self.register_buffer("channel_weights", normalized_channel_weights)

    def _l1_per_sample_channel(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        absolute_error = (prediction - target).abs()
        if self.target_l1_weight_strength == 0.0:
            return absolute_error.mean(dim=(-2, -1))
        weights = 1.0 + self.target_l1_weight_strength * target.detach()
        weights = weights / weights.mean(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        return (absolute_error * weights).mean(dim=(-2, -1))

    @staticmethod
    def _gaussian_downsample(inputs: torch.Tensor) -> torch.Tensor:
        coordinates = torch.arange(-1, 2, device=inputs.device, dtype=inputs.dtype)
        kernel_1d = torch.exp(-(coordinates**2) / 2.0)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel.expand(inputs.shape[1], 1, 3, 3).contiguous()
        padded = F.pad(inputs, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, kernel, stride=2, groups=inputs.shape[1])

    def per_sample(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return one hybrid loss per image without changing scalar reduction."""
        ssim_function = ssim_uniform_per_channel
        if self.channel_weights is None:
            ssim_loss = 1.0 - ssim_function(prediction, target).mean(dim=1)
            l1_loss = self._l1_per_sample_channel(prediction, target).mean(dim=1)
            mse_loss = (prediction - target).square().mean(dim=(1, 2, 3))
            per_sample = (
                self.ssim_weight * ssim_loss
                + self.l1_weight * l1_loss
                + self.mse_weight * mse_loss
            )
            if self.pyramid_weight > 0:
                prediction_low = self._gaussian_downsample(prediction)
                target_low = self._gaussian_downsample(target)
                pyramid_loss = (prediction_low - target_low).abs().mean(
                    dim=(1, 2, 3)
                )
                per_sample = per_sample + self.pyramid_weight * pyramid_loss
            return per_sample

        if self.channel_weights.shape[0] != prediction.shape[1]:
            raise ValueError(
                "channel_weights length must match the prediction channels, got "
                f"{self.channel_weights.shape[0]} weights for "
                f"{prediction.shape[1]} channels."
            )
        channel_weights = self.channel_weights.to(
            device=prediction.device,
            dtype=torch.float32,
        )
        ssim_loss = 1.0 - ssim_function(prediction, target)
        l1_loss = self._l1_per_sample_channel(prediction, target)
        mse_loss = (prediction - target).square().mean(dim=(-2, -1))
        per_sample_channel = (
            self.ssim_weight * ssim_loss
            + self.l1_weight * l1_loss
            + self.mse_weight * mse_loss
        )
        if self.pyramid_weight > 0:
            prediction_low = self._gaussian_downsample(prediction)
            target_low = self._gaussian_downsample(target)
            pyramid_loss = (prediction_low - target_low).abs().mean(dim=(-2, -1))
            per_sample_channel = (
                per_sample_channel + self.pyramid_weight * pyramid_loss
            )
        return (per_sample_channel * channel_weights[None, :]).sum(dim=1)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ssim_function = ssim_uniform_per_channel
        if self.channel_weights is not None:
            per_sample = self.per_sample(prediction, target)
            if sample_weights is None:
                return per_sample.mean()
            if (
                sample_weights.ndim != 1
                or sample_weights.shape[0] != prediction.shape[0]
            ):
                raise ValueError(
                    "sample_weights must have shape (batch_size,), got "
                    f"{tuple(sample_weights.shape)} for batch {prediction.shape[0]}."
                )
            sample_weights = sample_weights.to(
                device=prediction.device,
                dtype=torch.float32,
            )
            if not bool(torch.isfinite(sample_weights).all()) or not bool(
                (sample_weights > 0).all()
            ):
                raise ValueError("sample_weights must be finite and positive.")
            return (per_sample * sample_weights).mean()

        if sample_weights is not None:
            if (
                sample_weights.ndim != 1
                or sample_weights.shape[0] != prediction.shape[0]
            ):
                raise ValueError(
                    "sample_weights must have shape (batch_size,), got "
                    f"{tuple(sample_weights.shape)} for batch {prediction.shape[0]}."
                )
            sample_weights = sample_weights.to(
                device=prediction.device,
                dtype=torch.float32,
            )
            if not bool(torch.isfinite(sample_weights).all()) or not bool(
                (sample_weights > 0).all()
            ):
                raise ValueError("sample_weights must be finite and positive.")
            per_sample = self.per_sample(prediction, target)
            return (per_sample * sample_weights).mean()

        if self.target_l1_weight_strength > 0.0:
            return self.per_sample(prediction, target).mean()

        ssim_loss = 1.0 - ssim_function(prediction, target).mean()
        l1_loss = F.l1_loss(prediction, target)
        mse_loss = F.mse_loss(prediction, target)
        loss = (
            self.ssim_weight * ssim_loss
            + self.l1_weight * l1_loss
            + self.mse_weight * mse_loss
        )
        if self.pyramid_weight > 0:
            prediction_low = self._gaussian_downsample(prediction)
            target_low = self._gaussian_downsample(target)
            loss = loss + self.pyramid_weight * F.l1_loss(
                prediction_low, target_low
            )
        return loss


def save_grayscale_rgb(array: np.ndarray, path: Path, quality: int = 75) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").convert("RGB").save(
        path, format="JPEG", quality=quality
    )


def model_condition_d4_transform(model,condition,transform_index):
    # The submitted FiLM branch internally averages D4 thumbnail embeddings.
    return condition

def checkpoint_conditioning(checkpoint):
    spec=checkpoint.get('conditioning',checkpoint.get('args',{}).get('conditioning_spec'))
    if spec!=spatial_thumbnail_conditioning_specification():raise ValueError('Condition metadata mismatch')
    return spec

def build_model_from_checkpoint(checkpoint,device):
    args=checkpoint['args'];arch=args['model_arch']
    cls={'unext2-fcmae-tiny':UNeXt2VirtualStainer,'unext2-fcmae-tiny-local-condition':UNeXt2LocalConditionVirtualStainer}.get(arch)
    if cls is None:raise ValueError('Unsupported checkpoint architecture')
    checkpoint_conditioning(checkpoint)
    model=cls(out_channels=len(checkpoint['markers']),condition_mode='spatial-thumbnail-film').to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model

class PairedPatchDataset(Dataset):
    """Original paired D4 augmentation and complete-ROI halo loading, fixed submitted modes."""
    def __init__(self,data_root,names,markers=MARKERS,augment=False,context_halo=0,target_context_halo=0,distillation_target_dir=None):
        self.data_root=Path(data_root);self.names=list(names);self.markers=tuple(markers)
        self.augment=augment;self.context_halo=context_halo;self.target_context_halo=target_context_halo
        self.distillation_target_dir=Path(distillation_target_dir) if distillation_target_dir else None
        if not 0<=target_context_halo<=context_halo:raise ValueError('Invalid halo')
    def __len__(self):return len(self.names)
    def __getitem__(self,index):
        name=self.names[index]
        source=load_context_gray(self.data_root/'train/DAPI',name,self.context_halo)[None,...]
        targets=np.stack([load_context_gray(self.data_root/'train'/m,name,self.target_context_halo) for m in self.markers])
        teacher=load_gray(self.distillation_target_dir/name)[None,...] if self.distillation_target_dir else None
        if self.augment:
            if random.random()<.5:
                source=source[...,::-1];targets=targets[...,::-1]
                if teacher is not None:teacher=teacher[...,::-1]
            if random.random()<.5:
                source=source[...,::-1,:];targets=targets[...,::-1,:]
                if teacher is not None:teacher=teacher[...,::-1,:]
            rotations=random.randrange(4)
            if rotations:
                source=np.rot90(source,rotations,axes=(-2,-1));targets=np.rot90(targets,rotations,axes=(-2,-1))
                if teacher is not None:teacher=np.rot90(teacher,rotations,axes=(-2,-1))
        source=torch.from_numpy(np.ascontiguousarray(source)/255.).float()
        targets=torch.from_numpy(np.ascontiguousarray(targets)/255.).float()
        if teacher is None:return source,targets,name
        return source,targets,torch.from_numpy(np.ascontiguousarray(teacher)/255.).float(),name
