from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms import functional as vision_functional

from ihc_convnext_unet import ConvNeXtTinyUNet
from ihc_dinov2 import DinoV2SmallUNet, dinov2_model_specification
from ihc_dino_residual import (
    UNeXt2DinoResidualVirtualStainer,
    dinov2_residual_model_specification,
)
from ihc_dino_fusion import (
    UNeXt2DinoBottleneckFusionVirtualStainer,
    dinov2_fusion_model_specification,
)
from ihc_data import MARKERS, load_context_gray, load_gray, parse_patch_name
from ihc_dualbranch import (
    LOCAL_BASE_CHANNELS,
    SWIN_DEPTHS,
    SWIN_DIMS,
    SWIN_HEADS,
    SWIN_WINDOW_SIZE,
    DualBranchSwinSmall,
)
from ihc_nafnet import NAFNet
from ihc_restormer import (
    RESTORMER_SMALL_BLOCKS,
    RESTORMER_SMALL_DIM,
    RESTORMER_SMALL_FFN_EXPANSION,
    RESTORMER_SMALL_HEADS,
    RESTORMER_SMALL_REFINEMENT_BLOCKS,
    Restormer,
)
from ihc_unext2 import (
    UNeXt2LocalConditionVirtualStainer,
    unext2_local_condition_model_specification,
    UNeXt2SignalAuxVirtualStainer,
    unext2_signal_aux_model_specification,
    UNeXt2BaseVirtualStainer,
    UNeXt2CrossMarkerFeedbackVirtualStainer,
    UNeXt2DapiDetailResidualVirtualStainer,
    UNeXt2MarkerDecoderVirtualStainer,
    UNeXt2ShallowDapiSkipVirtualStainer,
    UNeXt2VirtualStainer,
    UNeXt2WideHeadVirtualStainer,
    unext2_cross_marker_feedback_model_specification,
    unext2_dapi_detail_residual_model_specification,
    unext2_marker_decoders_model_specification,
    unext2_model_specification,
    unext2_shallow_dapi_skip_model_specification,
    unext2_widehead_model_specification,
)
from ihc_spatial import (
    SPATIAL_CONDITION_DIMENSION,
    SPATIAL_CONDITION_MODE,
    SPATIAL_FILM_HIDDEN_FEATURES,
    SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
    SPATIAL_THUMBNAIL_ALIGNED_DILATIONS,
    SPATIAL_THUMBNAIL_ALIGNED_FEATURES,
    SPATIAL_THUMBNAIL_CHANNELS,
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
    SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
    SPATIAL_THUMBNAIL_SIDE,
    SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_TEXTURE_SIDE,
    SPATIAL_WHOLE_ROI_CONDITION_MODE,
    validate_spatial_conditioning_specification,
    validate_spatial_thumbnail_aligned_conditioning_specification,
    validate_spatial_thumbnail_conditioning_specification,
    validate_spatial_thumbnail_texture_conditioning_specification,
    validate_spatial_thumbnail_multiscale_conditioning_specification,
    validate_spatial_thumbnail_ring_conditioning_specification,
    validate_spatial_whole_roi_conditioning_specification,
)


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


def shift_tensor_edge(
    tensor: torch.Tensor,
    source_offset: tuple[int, int],
) -> torch.Tensor:
    """Torch equivalent of shift_array_edge using replicate padding."""

    if tensor.ndim < 2:
        raise ValueError("Shifted tensor must have at least two spatial dimensions.")
    row_offset, col_offset = source_offset
    if type(row_offset) is not int or type(col_offset) is not int:
        raise ValueError("Tensor source offsets must be integers.")
    padding = max(abs(row_offset), abs(col_offset))
    if padding == 0:
        return tensor
    padded = F.pad(tensor, (padding, padding, padding, padding), mode="replicate")
    height, width = tensor.shape[-2:]
    return padded[
        ...,
        padding + row_offset : padding + row_offset + height,
        padding + col_offset : padding + col_offset + width,
    ]


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        names: Sequence[str],
        markers: Sequence[str] = MARKERS,
        augment: bool = False,
        context_halo: int = 0,
        target_context_halo: int = 0,
        target_source_offset: tuple[int, int] = (0, 0),
        return_augmentation: bool = False,
        source_photometric_augmentation: bool = False,
        paired_affine_probability: float = 0.0,
        paired_affine_outer_halo: int = 0,
        paired_affine_degrees: float = 0.0,
        paired_affine_scale_min: float = 1.0,
        paired_affine_scale_max: float = 1.0,
        distillation_target_dir: Path | None = None,
    ) -> None:
        if target_context_halo < 0 or target_context_halo > context_halo:
            raise ValueError(
                "target_context_halo must be between 0 and context_halo, got "
                f"{target_context_halo} for context_halo={context_halo}."
            )
        self.data_root = data_root
        self.names = list(names)
        self.markers = tuple(markers)
        self.augment = augment
        self.context_halo = context_halo
        self.target_context_halo = target_context_halo
        if (
            not isinstance(target_source_offset, tuple)
            or len(target_source_offset) != 2
            or any(
                type(value) is not int or not -8 <= value <= 8
                for value in target_source_offset
            )
        ):
            raise ValueError(
                "target_source_offset must contain two integers from -8 to 8."
            )
        self.target_source_offset = target_source_offset
        self.return_augmentation = return_augmentation
        self.source_photometric_augmentation = source_photometric_augmentation
        if not 0.0 <= paired_affine_probability <= 1.0:
            raise ValueError("paired_affine_probability must be in [0, 1].")
        if paired_affine_probability > 0.0:
            if not augment:
                raise ValueError("Paired affine augmentation requires augment=True.")
            if return_augmentation:
                raise ValueError(
                    "Paired affine augmentation is incompatible with aligned D4 "
                    "augmentation metadata."
                )
            if (
                type(paired_affine_outer_halo) is not int
                or paired_affine_outer_halo < context_halo
                or paired_affine_outer_halo < target_context_halo
            ):
                raise ValueError(
                    "paired_affine_outer_halo must cover both requested context halos."
                )
            if not 0.0 < paired_affine_degrees <= 45.0:
                raise ValueError("paired_affine_degrees must be in (0, 45].")
            if not (
                0.0 < paired_affine_scale_min <= paired_affine_scale_max
            ):
                raise ValueError("Paired affine scale bounds are invalid.")
        self.paired_affine_probability = paired_affine_probability
        self.paired_affine_outer_halo = paired_affine_outer_halo
        self.paired_affine_degrees = paired_affine_degrees
        self.paired_affine_scale_min = paired_affine_scale_min
        self.paired_affine_scale_max = paired_affine_scale_max
        self.distillation_target_dir = (
            distillation_target_dir.resolve()
            if distillation_target_dir is not None
            else None
        )
        if self.distillation_target_dir is not None:
            if not self.distillation_target_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing distillation target directory: {self.distillation_target_dir}"
                )
            if return_augmentation or paired_affine_probability > 0.0:
                raise ValueError(
                    "Distillation targets do not support aligned-metadata or affine augmentation."
                )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(
        self,
        index: int,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, str]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
    ):
        name = self.names[index]
        load_halo = (
            self.paired_affine_outer_halo
            if self.paired_affine_probability > 0.0
            else self.context_halo
        )
        source = load_context_gray(
            self.data_root / "train" / "DAPI", name, load_halo
        )[None, ...]
        targets = np.stack(
            [
                load_context_gray(
                    self.data_root / "train" / marker,
                    name,
                    (
                        self.paired_affine_outer_halo
                        if self.paired_affine_probability > 0.0
                        else self.target_context_halo
                    ),
                )
                for marker in self.markers
            ]
        )
        targets = shift_array_edge(targets, self.target_source_offset)
        distillation_target = (
            load_gray(self.distillation_target_dir / name)[None, ...]
            if self.distillation_target_dir is not None
            else None
        )

        horizontal_flip = False
        vertical_flip = False
        rotations = 0
        if self.augment:
            horizontal_flip = random.random() < 0.5
            if horizontal_flip:
                source = source[..., ::-1]
                targets = targets[..., ::-1]
                if distillation_target is not None:
                    distillation_target = distillation_target[..., ::-1]
            vertical_flip = random.random() < 0.5
            if vertical_flip:
                source = source[..., ::-1, :]
                targets = targets[..., ::-1, :]
                if distillation_target is not None:
                    distillation_target = distillation_target[..., ::-1, :]
            rotations = random.randrange(4)
            if rotations:
                source = np.rot90(source, rotations, axes=(-2, -1))
                targets = np.rot90(targets, rotations, axes=(-2, -1))
                if distillation_target is not None:
                    distillation_target = np.rot90(
                        distillation_target, rotations, axes=(-2, -1)
                    )

        # Perturb only DAPI intensity to model modest acquisition-domain shifts.
        if self.source_photometric_augmentation and random.random() < 0.5:
            source_float = source.astype(np.float32) / 255.0
            gamma = random.uniform(0.9, 1.1)
            gain = random.uniform(0.9, 1.1)
            offset = random.uniform(-0.02, 0.02)
            source = np.clip(
                np.power(source_float, gamma) * gain + offset,
                0.0,
                1.0,
            ) * 255.0

        if self.paired_affine_probability > 0.0:
            paired_tensor = torch.from_numpy(
                np.ascontiguousarray(np.concatenate((source, targets), axis=0))
                / 255.0
            ).float()
            if torch.rand(()).item() < self.paired_affine_probability:
                angle, translations, scale, shear = RandomAffine.get_params(
                    degrees=(-self.paired_affine_degrees, self.paired_affine_degrees),
                    translate=None,
                    scale_ranges=(
                        self.paired_affine_scale_min,
                        self.paired_affine_scale_max,
                    ),
                    shears=None,
                    img_size=list(paired_tensor.shape[-2:]),
                )
                paired_tensor = vision_functional.affine(
                    paired_tensor,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            source_tensor = crop_context_halo(
                paired_tensor[:1],
                load_halo,
                self.context_halo,
            )
            target_tensor = crop_context_halo(
                paired_tensor[1:],
                load_halo,
                self.target_context_halo,
            )
        else:
            source_tensor = torch.from_numpy(
                np.ascontiguousarray(source) / 255.0
            ).float()
            target_tensor = torch.from_numpy(
                np.ascontiguousarray(targets) / 255.0
            ).float()
        if self.return_augmentation:
            augmentation = torch.tensor(
                (int(horizontal_flip), int(vertical_flip), rotations),
                dtype=torch.int64,
            )
            return source_tensor, target_tensor, augmentation, name
        if distillation_target is not None:
            distillation_tensor = torch.from_numpy(
                np.ascontiguousarray(distillation_target) / 255.0
            ).float()
            return source_tensor, target_tensor, distillation_tensor, name
        return source_tensor, target_tensor, name


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs) + self.skip(inputs)


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        growth = max(out_channels // 4, 8)
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(3):
            groups = min(8, growth)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, growth, 3, padding=1, bias=False),
                    nn.GroupNorm(groups, growth),
                    nn.SiLU(inplace=True),
                )
            )
            current_channels += growth
        self.project = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [inputs]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.project(torch.cat(features, dim=1)) + self.skip(inputs)


class MixStyle(nn.Module):
    """Mix detached instance statistics during training without adding state."""

    def __init__(
        self,
        probability: float = 0.0,
        alpha: float = 0.1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.configure(probability, alpha)
        self.epsilon = float(epsilon)

    def configure(self, probability: float, alpha: float) -> None:
        probability = float(probability)
        alpha = float(alpha)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0, 1].")
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive.")
        self.probability = probability
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            or self.probability == 0.0
            or inputs.shape[0] < 2
            or random.random() >= self.probability
        ):
            return inputs
        with torch.amp.autocast(inputs.device.type, enabled=False):
            features = inputs.float()
            mean = features.mean(dim=(-2, -1), keepdim=True).detach()
            standard_deviation = (
                features.var(dim=(-2, -1), keepdim=True, unbiased=False)
                + self.epsilon
            ).sqrt().detach()
            normalized = (features - mean) / standard_deviation
            concentration = torch.full(
                (inputs.shape[0],),
                self.alpha,
                device=inputs.device,
                dtype=torch.float32,
            )
            mixing = torch.distributions.Beta(
                concentration,
                concentration,
            ).sample()[:, None, None, None]
            permutation = torch.randperm(inputs.shape[0], device=inputs.device)
            mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
            mixed_standard_deviation = (
                mixing * standard_deviation
                + (1.0 - mixing) * standard_deviation[permutation]
            )
            mixed = normalized * mixed_standard_deviation + mixed_mean
        return mixed.to(dtype=inputs.dtype)


class FeatureFiLM(nn.Module):
    """Zero-initialized scalar conditioning without changing base-model RNG state."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels * 2, 1))
        self.bias = nn.Parameter(torch.zeros(channels * 2))

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        affine = F.linear(condition.to(dtype=self.weight.dtype), self.weight, self.bias)
        scale, shift = affine.chunk(2, dim=1)
        return features * (1.0 + scale[..., None, None]) + shift[..., None, None]


class SpatialFeatureFiLM(nn.Module):
    """Zero-output MLP FiLM for fixed DAPI spatial-statistics features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input = nn.Linear(
            SPATIAL_CONDITION_DIMENSION,
            SPATIAL_FILM_HIDDEN_FEATURES,
        )
        self.output = nn.Linear(SPATIAL_FILM_HIDDEN_FEATURES, channels * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = condition.to(dtype=self.input.weight.dtype)
        affine = self.output(F.silu(self.input(condition)))
        scale, shift = affine.chunk(2, dim=1)
        return features * (1.0 + scale[..., None, None]) + shift[..., None, None]


class SpatialThumbnailFeatureFiLM(nn.Module):
    """Preserve spatial-statistics FiLM and add a D4-invariant thumbnail branch."""

    def __init__(
        self,
        channels: int,
        *,
        thumbnail_channels: int = SPATIAL_THUMBNAIL_CHANNELS,
        thumbnail_side: int = SPATIAL_THUMBNAIL_SIDE,
    ) -> None:
        super().__init__()
        self.thumbnail_channels = int(thumbnail_channels)
        self.thumbnail_side = int(thumbnail_side)
        self.input = nn.Linear(
            SPATIAL_CONDITION_DIMENSION,
            SPATIAL_FILM_HIDDEN_FEATURES,
        )
        self.output = nn.Linear(SPATIAL_FILM_HIDDEN_FEATURES, channels * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        encoder_channels = (16, 32, 64, SPATIAL_THUMBNAIL_EMBEDDING_FEATURES)
        layers: list[nn.Module] = []
        input_channels = self.thumbnail_channels
        for output_channels in encoder_channels:
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(min(8, output_channels), output_channels),
                    nn.SiLU(inplace=True),
                )
            )
            input_channels = output_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.thumbnail_encoder = nn.Sequential(*layers)
        self.thumbnail_output = nn.Linear(
            SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
            channels * 2,
        )
        nn.init.zeros_(self.thumbnail_output.weight)
        nn.init.zeros_(self.thumbnail_output.bias)

    def thumbnail_embedding(self, thumbnail: torch.Tensor) -> torch.Tensor:
        transformed = []
        for transform_index in range(8):
            candidate = torch.rot90(
                thumbnail,
                transform_index % 4,
                dims=(-2, -1),
            )
            if transform_index >= 4:
                candidate = torch.flip(candidate, dims=(-1,))
            transformed.append(candidate)
        batch_size = thumbnail.shape[0]
        encoded = self.thumbnail_encoder(torch.cat(transformed, dim=0)).flatten(1)
        return encoded.reshape(8, batch_size, -1).mean(dim=0)

    def condition_embeddings(
        self,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = condition.to(dtype=self.input.weight.dtype)
        stats = condition[:, :SPATIAL_CONDITION_DIMENSION]
        thumbnail = condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
            -1,
            self.thumbnail_channels,
            self.thumbnail_side,
            self.thumbnail_side,
        )
        return F.silu(self.input(stats)), self.thumbnail_embedding(thumbnail)

    @staticmethod
    def apply_affine(
        features: torch.Tensor,
        affine: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = affine.chunk(2, dim=1)
        return features * (1.0 + scale[..., None, None]) + shift[..., None, None]

    def apply_bottleneck(
        self,
        features: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        stats_embedding, thumbnail_embedding = embeddings
        affine = self.output(stats_embedding) + self.thumbnail_output(
            thumbnail_embedding
        )
        return self.apply_affine(features, affine)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_bottleneck(
            features,
            self.condition_embeddings(condition),
        )


class SpatialThumbnailTextureFeatureFiLM(SpatialThumbnailFeatureFiLM):
    """D4-invariant FiLM over mean, contrast, gradient, and presence maps."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            channels,
            thumbnail_channels=SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
            thumbnail_side=SPATIAL_THUMBNAIL_TEXTURE_SIDE,
        )


class SpatialThumbnailRingFeatureFiLM(SpatialThumbnailFeatureFiLM):
    """Retain center/near/far thumbnail structure with a zero residual."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        self.ring_projection = nn.Linear(
            3 * SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
            SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
        )
        nn.init.zeros_(self.ring_projection.weight)
        nn.init.zeros_(self.ring_projection.bias)

    @staticmethod
    def ring_features(feature_map: torch.Tensor) -> torch.Tensor:
        expected_side = 5
        if feature_map.ndim != 4 or feature_map.shape[-2:] != (
            expected_side,
            expected_side,
        ):
            raise ValueError(
                "Thumbnail feature map must have shape (batch, channels, 5, 5), "
                f"got {tuple(feature_map.shape)}."
            )
        center = feature_map[:, :, 2, 2]
        inner_square = feature_map[:, :, 1:4, 1:4]
        inner_sum = inner_square.sum(dim=(-2, -1))
        inner_ring = (inner_sum - center) / 8.0
        outer_ring = (
            feature_map.sum(dim=(-2, -1)) - inner_sum
        ) / 16.0
        return torch.cat((center, inner_ring, outer_ring), dim=1)

    def thumbnail_embedding(self, thumbnail: torch.Tensor) -> torch.Tensor:
        transformed = []
        for transform_index in range(8):
            candidate = torch.rot90(
                thumbnail,
                transform_index % 4,
                dims=(-2, -1),
            )
            if transform_index >= 4:
                candidate = torch.flip(candidate, dims=(-1,))
            transformed.append(candidate)
        feature_map = torch.cat(transformed, dim=0)
        for layer in tuple(self.thumbnail_encoder.children())[:-1]:
            feature_map = layer(feature_map)
        global_embedding = self.thumbnail_encoder[-1](feature_map).flatten(1)
        encoded = global_embedding + self.ring_projection(
            self.ring_features(feature_map)
        )
        batch_size = thumbnail.shape[0]
        return encoded.reshape(8, batch_size, -1).mean(dim=0)


class SpatialThumbnailMultiScaleFeatureFiLM(SpatialThumbnailFeatureFiLM):
    """Extend thumbnail conditioning into zero-initialized decoder FiLM."""

    def __init__(
        self,
        bottleneck_channels: int,
        decoder_channels: Sequence[int],
    ) -> None:
        super().__init__(bottleneck_channels)
        condition_features = (
            SPATIAL_FILM_HIDDEN_FEATURES
            + SPATIAL_THUMBNAIL_EMBEDDING_FEATURES
        )
        self.decoder_outputs = nn.ModuleList(
            nn.Linear(condition_features, channels * 2)
            for channels in decoder_channels
        )
        for output in self.decoder_outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward_with_state(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.condition_embeddings(condition)
        return self.apply_bottleneck(features, embeddings), embeddings

    def apply_decoder(
        self,
        features: torch.Tensor,
        embeddings: tuple[torch.Tensor, torch.Tensor],
        decoder_index: int,
    ) -> torch.Tensor:
        if not 0 <= decoder_index < len(self.decoder_outputs):
            raise ValueError(f"Invalid decoder FiLM index: {decoder_index}.")
        combined = torch.cat(embeddings, dim=1)
        return self.apply_affine(
            features,
            self.decoder_outputs[decoder_index](combined),
        )


class SpatialThumbnailAlignedFeatureFusion(SpatialThumbnailFeatureFiLM):
    """Add a jointly transformed low-resolution context map at the bottleneck."""

    def __init__(self, bottleneck_channels: int) -> None:
        super().__init__(bottleneck_channels)
        layers: list[nn.Module] = []
        input_channels = SPATIAL_THUMBNAIL_CHANNELS
        for dilation in SPATIAL_THUMBNAIL_ALIGNED_DILATIONS:
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        SPATIAL_THUMBNAIL_ALIGNED_FEATURES,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.GroupNorm(8, SPATIAL_THUMBNAIL_ALIGNED_FEATURES),
                    nn.SiLU(inplace=True),
                )
            )
            input_channels = SPATIAL_THUMBNAIL_ALIGNED_FEATURES
        self.aligned_encoder = nn.Sequential(*layers)
        self.aligned_projection = nn.Conv2d(
            SPATIAL_THUMBNAIL_ALIGNED_FEATURES,
            bottleneck_channels,
            1,
        )
        nn.init.zeros_(self.aligned_projection.weight)
        nn.init.zeros_(self.aligned_projection.bias)

    @staticmethod
    def thumbnail_from_condition(condition: torch.Tensor) -> torch.Tensor:
        return condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
            -1,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )

    @staticmethod
    def center_crop_feature_map(
        feature_map: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        target_height, target_width = spatial_shape
        height, width = feature_map.shape[-2:]
        if height < target_height or width < target_width:
            raise ValueError(
                "Aligned context feature map is smaller than the U-Net bottleneck: "
                f"{(height, width)} < {(target_height, target_width)}."
            )
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return feature_map[
            ...,
            top : top + target_height,
            left : left + target_width,
        ]

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self.apply_bottleneck(
            features,
            self.condition_embeddings(condition),
        )
        aligned = self.aligned_encoder(self.thumbnail_from_condition(condition))
        aligned = self.center_crop_feature_map(aligned, features.shape[-2:])
        return conditioned + self.aligned_projection(aligned)


class UNet(nn.Module):
    def __init__(
        self,
        out_channels: int = 4,
        base_channels: int = 32,
        block_type: str = "residual",
        condition_mode: str = "none",
    ) -> None:
        super().__init__()
        block_types = {"residual": ConvBlock, "dense": DenseConvBlock}
        if block_type not in block_types:
            raise ValueError(f"Unsupported U-Net block type: {block_type}")
        block = block_types[block_type]
        self.block_type = block_type
        if condition_mode not in {
            "none",
            "roi-mean-film",
            SPATIAL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
            SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
            SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
        }:
            raise ValueError(f"Unsupported U-Net condition mode: {condition_mode}")
        self.condition_mode = condition_mode
        channels = [base_channels * (2**index) for index in range(5)]
        self.encoders = nn.ModuleList(
            [
                block(1, channels[0]),
                block(channels[0], channels[1]),
                block(channels[1], channels[2]),
                block(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.mixstyles = nn.ModuleList((MixStyle(), MixStyle()))
        self.bottleneck = block(channels[3], channels[4])
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2),
                nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2),
                nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2),
                nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                block(channels[3] * 2, channels[3]),
                block(channels[2] * 2, channels[2]),
                block(channels[1] * 2, channels[1]),
                block(channels[0] * 2, channels[0]),
            ]
        )
        self.head = nn.Conv2d(channels[0], out_channels, 1)
        nn.init.constant_(self.head.bias, -2.0)
        if condition_mode == "roi-mean-film":
            self.condition_film: nn.Module | None = FeatureFiLM(channels[4])
            self.condition_dimension = 1
        elif condition_mode == SPATIAL_CONDITION_MODE:
            self.condition_film = SpatialFeatureFiLM(channels[4])
            self.condition_dimension = SPATIAL_CONDITION_DIMENSION
        elif condition_mode == SPATIAL_THUMBNAIL_CONDITION_MODE:
            self.condition_film = SpatialThumbnailFeatureFiLM(channels[4])
            self.condition_dimension = SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        elif condition_mode == SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE:
            self.condition_film = SpatialThumbnailTextureFeatureFiLM(channels[4])
            self.condition_dimension = SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION
        elif condition_mode == SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE:
            self.condition_film = SpatialThumbnailMultiScaleFeatureFiLM(
                channels[4],
                tuple(reversed(channels[:4])),
            )
            self.condition_dimension = SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        elif condition_mode == SPATIAL_THUMBNAIL_RING_CONDITION_MODE:
            self.condition_film = SpatialThumbnailRingFeatureFiLM(channels[4])
            self.condition_dimension = SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        elif condition_mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE:
            self.condition_film = SpatialThumbnailAlignedFeatureFusion(channels[4])
            self.condition_dimension = SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        else:
            self.condition_film = None
            self.condition_dimension = 0

    def configure_mixstyle(self, probability: float, alpha: float) -> None:
        for module in self.mixstyles:
            module.configure(probability, alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_mode == "none":
            if condition is not None:
                raise ValueError("Unconditioned U-Net received a condition tensor.")
        else:
            if condition is None:
                raise ValueError("Conditioned U-Net requires a condition tensor.")
            if condition.ndim == 1:
                condition = condition[:, None]
            if condition.shape != (inputs.shape[0], self.condition_dimension):
                raise ValueError(
                    "condition has an invalid shape; expected "
                    f"(batch_size, {self.condition_dimension}), got "
                    f"{tuple(condition.shape)}"
                )
            condition = condition.to(device=inputs.device, dtype=torch.float32)
            if not bool(torch.isfinite(condition).all()):
                raise ValueError("condition must contain only finite values.")

        skips = []
        current = inputs
        for encoder_index, encoder in enumerate(self.encoders):
            current = encoder(current)
            skips.append(current)
            current = self.pool(current)
            if encoder_index < len(self.mixstyles):
                current = self.mixstyles[encoder_index](current)
        current = self.bottleneck(current)
        multiscale_state: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.condition_film is not None and condition is not None:
            if isinstance(
                self.condition_film,
                SpatialThumbnailMultiScaleFeatureFiLM,
            ):
                current, multiscale_state = self.condition_film.forward_with_state(
                    current,
                    condition,
                )
            else:
                current = self.condition_film(current, condition)
        for decoder_index, (upconv, decoder, skip) in enumerate(
            zip(self.upconvs, self.decoders, reversed(skips))
        ):
            current = upconv(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = F.interpolate(
                    current, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            current = decoder(torch.cat((current, skip), dim=1))
            if multiscale_state is not None:
                if not isinstance(
                    self.condition_film,
                    SpatialThumbnailMultiScaleFeatureFiLM,
                ):
                    raise RuntimeError("Multiscale condition state has no FiLM module.")
                current = self.condition_film.apply_decoder(
                    current,
                    multiscale_state,
                    decoder_index,
                )
        return torch.sigmoid(self.head(current))


def checkpoint_conditioning(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    raw = checkpoint.get("conditioning")
    if raw is None:
        conditioning: dict[str, object] = {"version": 1, "mode": "none"}
    else:
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint conditioning metadata must be a dictionary.")
        if raw.get("version") != 1:
            raise ValueError(
                f"Unsupported checkpoint conditioning version: {raw.get('version')}"
            )
        mode = raw.get("mode")
        if mode == "none":
            conditioning = {"version": 1, "mode": "none"}
        elif mode == SPATIAL_CONDITION_MODE:
            conditioning = validate_spatial_conditioning_specification(raw)
        elif mode == SPATIAL_THUMBNAIL_CONDITION_MODE:
            conditioning = validate_spatial_thumbnail_conditioning_specification(raw)
        elif mode == SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE:
            conditioning = validate_spatial_thumbnail_texture_conditioning_specification(
                raw
            )
        elif mode == SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE:
            conditioning = (
                validate_spatial_thumbnail_multiscale_conditioning_specification(
                    raw
                )
            )
        elif mode == SPATIAL_THUMBNAIL_RING_CONDITION_MODE:
            conditioning = validate_spatial_thumbnail_ring_conditioning_specification(
                raw
            )
        elif mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE:
            conditioning = (
                validate_spatial_thumbnail_aligned_conditioning_specification(raw)
            )
        elif mode == SPATIAL_WHOLE_ROI_CONDITION_MODE:
            conditioning = validate_spatial_whole_roi_conditioning_specification(raw)
        else:
            if mode != "roi-mean-film":
                raise ValueError(f"Unsupported checkpoint conditioning mode: {mode}")
            if (
                raw.get("feature") != "dapi_roi_mean"
                or raw.get("injection") != "bottleneck"
            ):
                raise ValueError("Unsupported ROI mean conditioning specification.")
            normalization = raw.get("normalization")
            if not isinstance(normalization, dict):
                raise ValueError("Checkpoint conditioning normalization is missing.")
            expected_normalization = {
                "fit_scope": "training_rois_only",
                "roi_weighting": "equal",
                "units": "uint8_0_255",
            }
            for key, expected in expected_normalization.items():
                if normalization.get(key) != expected:
                    raise ValueError(
                        f"Unsupported conditioning normalization {key}: "
                        f"{normalization.get(key)!r}"
                    )
            try:
                center = float(normalization["center"])
                scale = float(normalization["scale"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Checkpoint conditioning normalization is incomplete."
                ) from error
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-6:
                raise ValueError(
                    f"Checkpoint conditioning normalization is invalid: {center}, {scale}"
                )
            conditioning = raw

    args = checkpoint.get("args")
    if isinstance(args, Mapping):
        if args.get("condition_mode", "none") != conditioning["mode"]:
            raise ValueError(
                "Checkpoint conditioning mode disagrees with its args metadata."
            )
        if "conditioning_spec" in args and args["conditioning_spec"] != conditioning:
            raise ValueError(
                "Checkpoint conditioning specification disagrees with its args metadata."
            )
    return conditioning


def model_specification(
    architecture: str,
    base_channels: int = 32,
    block_type: str = "residual",
) -> dict[str, object]:
    if architecture == "unet":
        if base_channels <= 0:
            raise ValueError("U-Net base_channels must be positive.")
        if block_type not in {"residual", "dense"}:
            raise ValueError(f"Unsupported U-Net block type: {block_type}")
        return {
            "version": 1,
            "architecture": "unet",
            "base_channels": int(base_channels),
            "block_type": block_type,
        }
    if architecture == "nafnet-width32":
        return {
            "version": 1,
            "architecture": "nafnet-width32",
            "width": 32,
            "enc_blk_nums": [1, 1, 1, 28],
            "middle_blk_num": 1,
            "dec_blk_nums": [1, 1, 1, 1],
        }
    if architecture == "convnext-tiny-unet":
        return {
            "version": 1,
            "architecture": "convnext-tiny-unet",
            "encoder": "torchvision-convnext-tiny",
            "encoder_channels": [96, 192, 384, 768],
            "decoder_channels": [384, 192, 96, 64, 32],
            "input_channels": 1,
            "encoder_input_replication": 3,
            "encoder_normalization": "imagenet1k-v1",
            "full_resolution_skip": "shallow-dapi",
            "decoder_normalization": "groupnorm",
            "decoder_activation": "silu",
            "output_activation": "sigmoid",
        }
    if architecture == "dinov2-small-unet":
        return dinov2_model_specification()
    if architecture == "unext2-fcmae-tiny-dinov2-residual":
        return dinov2_residual_model_specification()
    if architecture == "unext2-fcmae-tiny-dinov2-fusion":
        return dinov2_fusion_model_specification()
    if architecture == "unext2-fcmae-tiny":
        return unext2_model_specification()
    if architecture == "unext2-fcmae-tiny-signal-aux":
        return unext2_signal_aux_model_specification()
    if architecture == "unext2-fcmae-tiny-local-condition":
        return unext2_local_condition_model_specification()
    if architecture == "unext2-fcmae-base":
        return unext2_model_specification("base")
    if architecture == "unext2-fcmae-tiny-widehead":
        return unext2_widehead_model_specification()
    if architecture == "unext2-fcmae-tiny-marker-decoders":
        return unext2_marker_decoders_model_specification()
    if architecture == "unext2-fcmae-tiny-cross-marker-feedback":
        return unext2_cross_marker_feedback_model_specification()
    if architecture == "unext2-fcmae-tiny-dapi-detail-residual":
        return unext2_dapi_detail_residual_model_specification()
    if architecture == "unext2-fcmae-tiny-shallow-dapi-skip":
        return unext2_shallow_dapi_skip_model_specification()
    if architecture == "dualbranch-swin-s":
        if base_channels != LOCAL_BASE_CHANNELS:
            raise ValueError(
                "DualBranch-Swin-S has fixed local base_channels=32."
            )
        if block_type != "residual":
            raise ValueError(
                "DualBranch-Swin-S has a fixed residual local block type."
            )
        return {
            "version": 1,
            "architecture": "dualbranch-swin-s",
            "local_architecture": "residual-unet",
            "local_base_channels": LOCAL_BASE_CHANNELS,
            "patch_size": 4,
            "swin_dims": list(SWIN_DIMS),
            "swin_depths": list(SWIN_DEPTHS),
            "swin_heads": list(SWIN_HEADS),
            "swin_window_size": SWIN_WINDOW_SIZE,
            "swin_mlp_ratio": 4.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "stochastic_depth": 0.0,
            "fusion_scales": [4, 8, 16],
            "fusion_channels": [128, 256, 512],
            "fusion_mode": "zero-gated-add",
            "pretraining": "none",
            "output_activation": "sigmoid",
        }
    if architecture == "restormer-small":
        if base_channels != 32:
            raise ValueError("Restormer-small has a fixed base_channels=32.")
        if block_type != "residual":
            raise ValueError("Restormer-small has a fixed residual block selector.")
        return {
            "version": 1,
            "architecture": "restormer-small",
            "input_channels": 1,
            "dim": RESTORMER_SMALL_DIM,
            "num_blocks": list(RESTORMER_SMALL_BLOCKS),
            "num_refinement_blocks": RESTORMER_SMALL_REFINEMENT_BLOCKS,
            "heads": list(RESTORMER_SMALL_HEADS),
            "ffn_expansion_factor": RESTORMER_SMALL_FFN_EXPANSION,
            "layer_norm": "bias-free",
            "global_input_residual": False,
            "output_activation": "sigmoid",
        }
    raise ValueError(f"Unsupported model architecture: {architecture}")


def checkpoint_model_specification(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    raw = checkpoint.get("model_spec")
    if raw is None:
        specification = model_specification(
            "unet",
            base_channels=int(checkpoint["base_channels"]),
            block_type=str(checkpoint.get("block_type", "residual")),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("Checkpoint model_spec must be a mapping.")
        if raw.get("version") != 1:
            raise ValueError(
                f"Unsupported checkpoint model specification version: {raw.get('version')}"
            )
        architecture = raw.get("architecture")
        if architecture == "unet":
            try:
                specification = model_specification(
                    "unet",
                    base_channels=int(raw["base_channels"]),
                    block_type=str(raw["block_type"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid U-Net checkpoint model specification.") from error
        elif architecture == "nafnet-width32":
            specification = model_specification("nafnet-width32")
            if dict(raw) != specification:
                raise ValueError("Unsupported NAFNet checkpoint model specification.")
        elif architecture == "convnext-tiny-unet":
            specification = model_specification("convnext-tiny-unet")
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported ConvNeXt-Tiny U-Net checkpoint model specification."
                )
        elif architecture == "dinov2-small-unet":
            specification = dinov2_model_specification()
            if dict(raw) != specification:
                raise ValueError("Unsupported DINOv2 checkpoint model specification.")
        elif architecture == "unext2-fcmae-tiny-dinov2-residual":
            specification = dinov2_residual_model_specification()
            if dict(raw) != specification:
                raise ValueError("Unsupported DINOv2 residual checkpoint model specification.")
        elif architecture == "unext2-fcmae-tiny-dinov2-fusion":
            specification = dinov2_fusion_model_specification()
            if dict(raw) != specification:
                raise ValueError("Unsupported DINOv2 fusion checkpoint model specification.")
        elif architecture == "unext2-fcmae-tiny-local-condition":
            specification=unext2_local_condition_model_specification()
            if dict(raw)!=specification:
                raise ValueError('Unsupported local-condition specification')
        elif architecture == "unext2-fcmae-tiny-signal-aux":
            specification = unext2_signal_aux_model_specification()
            if dict(raw) != specification:
                raise ValueError('Unsupported signal auxiliary model specification')
        elif architecture == "unext2-fcmae-tiny":
            specification = model_specification("unext2-fcmae-tiny")
            if dict(raw) != specification:
                raise ValueError("Unsupported UNeXt2 checkpoint model specification.")
        elif architecture == "unext2-fcmae-base":
            specification = model_specification("unext2-fcmae-base")
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported Base UNeXt2 checkpoint model specification."
                )
        elif architecture == "unext2-fcmae-tiny-widehead":
            specification = model_specification("unext2-fcmae-tiny-widehead")
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported wide-head UNeXt2 checkpoint model specification."
                )
        elif architecture == "unext2-fcmae-tiny-marker-decoders":
            specification = model_specification(
                "unext2-fcmae-tiny-marker-decoders"
            )
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported marker-decoder UNeXt2 checkpoint model specification."
                )
        elif architecture == "unext2-fcmae-tiny-cross-marker-feedback":
            specification = model_specification(
                "unext2-fcmae-tiny-cross-marker-feedback"
            )
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported cross-marker feedback UNeXt2 checkpoint model specification."
                )
        elif architecture == "unext2-fcmae-tiny-dapi-detail-residual":
            specification = model_specification(
                "unext2-fcmae-tiny-dapi-detail-residual"
            )
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported DAPI detail residual UNeXt2 checkpoint model specification."
                )
        elif architecture == "unext2-fcmae-tiny-shallow-dapi-skip":
            specification = model_specification(
                "unext2-fcmae-tiny-shallow-dapi-skip"
            )
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported shallow DAPI skip UNeXt2 checkpoint model specification."
                )
        elif architecture == "dualbranch-swin-s":
            specification = model_specification("dualbranch-swin-s")
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported DualBranch-Swin-S checkpoint model specification."
                )
        elif architecture == "restormer-small":
            specification = model_specification("restormer-small")
            if dict(raw) != specification:
                raise ValueError(
                    "Unsupported Restormer-small checkpoint model specification."
                )
        else:
            raise ValueError(f"Unsupported checkpoint model architecture: {architecture}")

    args = checkpoint.get("args")
    if isinstance(args, Mapping):
        if args.get("model_arch", "unet") != specification["architecture"]:
            raise ValueError(
                "Checkpoint model architecture disagrees with its args metadata."
            )
        if "model_spec" in args and args["model_spec"] != specification:
            raise ValueError(
                "Checkpoint model specification disagrees with its args metadata."
            )
    return specification


def build_model(
    architecture: str,
    out_channels: int,
    base_channels: int = 32,
    block_type: str = "residual",
    condition_mode: str = "none",
) -> nn.Module:
    if architecture == "unet":
        return UNet(
            out_channels=out_channels,
            base_channels=base_channels,
            block_type=block_type,
            condition_mode=condition_mode,
        )
    if architecture == "nafnet-width32":
        if condition_mode != "none":
            raise ValueError("NAFNet does not support ROI conditioning.")
        return NAFNet(out_channels=out_channels)
    if architecture == "convnext-tiny-unet":
        if condition_mode != "none":
            raise ValueError("ConvNeXt-Tiny U-Net does not support ROI conditioning.")
        return ConvNeXtTinyUNet(out_channels=out_channels)
    if architecture == "dinov2-small-unet":
        return DinoV2SmallUNet(out_channels=out_channels, condition_mode=condition_mode)
    if architecture == "unext2-fcmae-tiny-signal-aux":
        return UNeXt2SignalAuxVirtualStainer(out_channels=out_channels, condition_mode=condition_mode)
    if architecture == "unext2-fcmae-tiny-local-condition":
        return UNeXt2LocalConditionVirtualStainer(out_channels=out_channels,condition_mode=condition_mode)
    if architecture == "unext2-fcmae-tiny-dinov2-residual":
        return UNeXt2DinoResidualVirtualStainer(out_channels=out_channels, condition_mode=condition_mode)
    if architecture == "unext2-fcmae-tiny-dinov2-fusion":
        return UNeXt2DinoBottleneckFusionVirtualStainer(out_channels=out_channels, condition_mode=condition_mode)
    if architecture == "unext2-fcmae-tiny":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2VirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-base":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "Base UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2BaseVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-tiny-widehead":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "Wide-head UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2WideHeadVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-tiny-marker-decoders":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "Marker-decoder UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2MarkerDecoderVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-tiny-cross-marker-feedback":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "Cross-marker feedback UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2CrossMarkerFeedbackVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-tiny-dapi-detail-residual":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "DAPI detail residual UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2DapiDetailResidualVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "unext2-fcmae-tiny-shallow-dapi-skip":
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "Shallow DAPI skip UNeXt2 received an unsupported conditioning mode."
            )
        return UNeXt2ShallowDapiSkipVirtualStainer(
            out_channels=out_channels,
            condition_mode=condition_mode,
        )
    if architecture == "dualbranch-swin-s":
        if condition_mode != "none":
            raise ValueError("DualBranch-Swin-S does not support ROI conditioning.")
        model_specification(
            architecture,
            base_channels=base_channels,
            block_type=block_type,
        )
        return DualBranchSwinSmall(out_channels=out_channels)
    if architecture == "restormer-small":
        if condition_mode != "none":
            raise ValueError("Restormer-small does not support ROI conditioning.")
        model_specification(
            architecture,
            base_channels=base_channels,
            block_type=block_type,
        )
        return Restormer(out_channels=out_channels)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def build_model_from_checkpoint(
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> nn.Module:
    markers = tuple(checkpoint.get("markers", MARKERS))
    conditioning = checkpoint_conditioning(checkpoint)
    specification = checkpoint_model_specification(checkpoint)
    model = build_model(
        architecture=str(specification["architecture"]),
        out_channels=len(markers),
        base_channels=int(specification.get("base_channels", 32)),
        block_type=str(specification.get("block_type", "residual")),
        condition_mode=str(conditioning["mode"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def build_unet_from_checkpoint(
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> UNet:
    """Load a legacy U-Net checkpoint; generic callers should use build_model_from_checkpoint."""

    if checkpoint_model_specification(checkpoint)["architecture"] != "unet":
        raise ValueError("Checkpoint does not contain a U-Net model.")
    model = build_model_from_checkpoint(checkpoint, device)
    if not isinstance(model, UNet):
        raise TypeError("Expected build_model_from_checkpoint to return UNet.")
    return model


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


def _replace_condition_thumbnail(
    condition: torch.Tensor,
    thumbnail: torch.Tensor,
) -> torch.Tensor:
    if condition.ndim != 2 or condition.shape[1] != SPATIAL_THUMBNAIL_CONDITION_DIMENSION:
        raise ValueError(
            "Spatial-thumbnail condition has an invalid shape: "
            f"{tuple(condition.shape)}."
        )
    expected_shape = (
        condition.shape[0],
        SPATIAL_THUMBNAIL_CHANNELS,
        SPATIAL_THUMBNAIL_SIDE,
        SPATIAL_THUMBNAIL_SIDE,
    )
    if thumbnail.shape != expected_shape:
        raise ValueError(
            "Spatial-thumbnail tensor has an invalid shape: "
            f"{tuple(thumbnail.shape)} != {expected_shape}."
        )
    return torch.cat(
        (
            condition[:, :SPATIAL_CONDITION_DIMENSION],
            thumbnail.flatten(1),
        ),
        dim=1,
    )


def aligned_condition_from_augmentation(
    condition: torch.Tensor,
    augmentation: torch.Tensor,
) -> torch.Tensor:
    """Apply the dataset's sampled flips and rotation to each thumbnail."""
    if augmentation.ndim != 2 or augmentation.shape != (condition.shape[0], 3):
        raise ValueError(
            "Augmentation metadata must have shape (batch, 3), got "
            f"{tuple(augmentation.shape)}."
        )
    augmentation = augmentation.to(device="cpu", dtype=torch.int64)
    if bool(
        ((augmentation[:, :2] < 0) | (augmentation[:, :2] > 1)).any()
        or ((augmentation[:, 2] < 0) | (augmentation[:, 2] > 3)).any()
    ):
        raise ValueError("Augmentation metadata contains an invalid transform.")
    thumbnail = condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
        -1,
        SPATIAL_THUMBNAIL_CHANNELS,
        SPATIAL_THUMBNAIL_SIDE,
        SPATIAL_THUMBNAIL_SIDE,
    )
    transformed: list[torch.Tensor] = []
    for index, row in enumerate(augmentation.tolist()):
        candidate = thumbnail[index]
        horizontal_flip, vertical_flip, rotations = row
        if horizontal_flip:
            candidate = torch.flip(candidate, dims=(-1,))
        if vertical_flip:
            candidate = torch.flip(candidate, dims=(-2,))
        if rotations:
            candidate = torch.rot90(candidate, rotations, dims=(-2, -1))
        transformed.append(candidate)
    return _replace_condition_thumbnail(condition, torch.stack(transformed))


def aligned_condition_d4_transform(
    condition: torch.Tensor,
    transform_index: int,
) -> torch.Tensor:
    thumbnail = condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
        -1,
        SPATIAL_THUMBNAIL_CHANNELS,
        SPATIAL_THUMBNAIL_SIDE,
        SPATIAL_THUMBNAIL_SIDE,
    )
    return _replace_condition_thumbnail(
        condition,
        apply_d4_transform(thumbnail, transform_index),
    )


def model_condition_d4_transform(
    model: nn.Module,
    condition: torch.Tensor | None,
    transform_index: int,
) -> torch.Tensor | None:
    if condition is None or getattr(model, "condition_mode", None) != (
        SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE
    ):
        return condition
    return aligned_condition_d4_transform(condition, transform_index)


def _gaussian_window(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_gaussian_impl(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    valid_window: bool = False,
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = _gaussian_window(
        channels, window_size, sigma, prediction.device, prediction.dtype
    )
    padding = window_size // 2
    if not valid_window:
        prediction = F.pad(prediction, (padding,) * 4, mode="reflect")
        target = F.pad(target, (padding,) * 4, mode="reflect")

    mu_prediction = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    variance_prediction = (
        F.conv2d(prediction.square(), window, groups=channels) - mu_prediction_sq
    )
    variance_target = F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    covariance = F.conv2d(prediction * target, window, groups=channels) - mu_product

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_product + c1) * (2 * covariance + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        variance_prediction + variance_target + c2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean(dim=(-2, -1))


def ssim_per_channel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    with torch.amp.autocast(prediction.device.type, enabled=False):
        return _ssim_gaussian_impl(
            prediction.float(), target.float(), window_size, sigma
        )


def ssim_gaussian_valid_per_channel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian11 sigma1.5 population SSIM over complete windows only."""
    if prediction.shape != target.shape or min(prediction.shape[-2:]) < 11:
        raise ValueError("Gaussian valid SSIM requires matching images at least11x11.")
    with torch.amp.autocast(prediction.device.type, enabled=False):
        return _ssim_gaussian_impl(prediction.float(), target.float(), valid_window=True)


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


def psnr_per_channel(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().mean(dim=(-2, -1)).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def competition_score(ssim: float, psnr: float, psnr_max: float = 50.0) -> float:
    """Return the 0-100 score confirmed by the organizer for the preliminary round."""
    normalized_psnr = float(np.clip(float(psnr) / psnr_max, 0.0, 1.0))
    return 100.0 * (0.7 * float(ssim) + 0.3 * normalized_psnr)


def psnr_from_global_error(
    squared_error_sum: float,
    pixel_count: int,
    data_range: float = 255.0,
) -> float:
    """Compute PSNR after pooling squared error across all evaluated pixels."""
    if pixel_count <= 0:
        raise ValueError("pixel_count must be positive.")
    if squared_error_sum < 0:
        raise ValueError("squared_error_sum must be non-negative.")
    mse = max(float(squared_error_sum) / pixel_count, 1e-10)
    return float(10.0 * np.log10(float(data_range) ** 2 / mse))


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


def predict_with_stride_phase_tta(
    model: nn.Module,
    outer_inputs: torch.Tensor,
    target_spatial_shape: Sequence[int],
    model_context_halo: int,
    mode: str = "d4",
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average four aligned one-pixel input phases around a fixed target crop."""

    if len(target_spatial_shape) != 2:
        raise ValueError("Target spatial shape must contain height and width.")
    target_height, target_width = (int(value) for value in target_spatial_shape)
    if target_height <= 0 or target_width <= 0 or model_context_halo < 0:
        raise ValueError("Target shape and model context halo must be non-negative.")
    model_height = target_height + 2 * model_context_halo
    model_width = target_width + 2 * model_context_halo
    expected_outer = (model_height + 2, model_width + 2)
    if tuple(outer_inputs.shape[-2:]) != expected_outer:
        raise ValueError(
            "Stride-phase TTA requires exactly one outer pixel per side: "
            f"expected {expected_outer}, got {tuple(outer_inputs.shape[-2:])}."
        )

    aligned_predictions = []
    for row_phase, col_phase in ((0, 0), (0, 1), (1, 0), (1, 1)):
        row_window = 1 + row_phase
        col_window = 1 + col_phase
        window = outer_inputs[
            ...,
            row_window : row_window + model_height,
            col_window : col_window + model_width,
        ]
        prediction = predict_with_tta(model, window, mode, condition)
        row_start = model_context_halo - row_phase
        col_start = model_context_halo - col_phase
        aligned_predictions.append(
            prediction[
                ...,
                row_start : row_start + target_height,
                col_start : col_start + target_width,
            ]
        )
    return torch.stack(aligned_predictions).mean(dim=0)


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


def crop_context_halo(
    inputs: torch.Tensor,
    outer_halo: int,
    inner_halo: int,
) -> torch.Tensor:
    """Crop a tensor loaded with outer_halo to the same center at inner_halo."""
    if type(outer_halo) is not int or type(inner_halo) is not int:
        raise ValueError("Context halos must be integers.")
    if inner_halo < 0 or outer_halo < inner_halo:
        raise ValueError(
            "Context halos must satisfy 0 <= inner_halo <= outer_halo, got "
            f"inner_halo={inner_halo}, outer_halo={outer_halo}."
        )
    crop = outer_halo - inner_halo
    if crop == 0:
        return inputs
    height, width = inputs.shape[-2:]
    if height <= 2 * crop or width <= 2 * crop:
        raise ValueError(
            f"Cannot remove {crop} pixels per side from spatial shape "
            f"{(height, width)}."
        )
    return inputs[..., crop:-crop, crop:-crop]


class CompetitionLoss(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.7,
        l1_weight: float = 0.2,
        mse_weight: float = 0.1,
        pyramid_weight: float = 0.0,
        target_l1_weight_strength: float = 0.0,
        ssim_mode: str = "gaussian11",
        channel_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        total = ssim_weight + l1_weight + mse_weight
        if not np.isclose(total, 1.0):
            raise ValueError(f"Loss weights must sum to 1, got {total}")
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        if ssim_mode not in {"gaussian11", "uniform7", "gaussian11_valid"}:
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
        ssim_function = {"gaussian11": ssim_per_channel,
                         "uniform7": ssim_uniform_per_channel,
                         "gaussian11_valid": ssim_gaussian_valid_per_channel}[self.ssim_mode]
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
        ssim_function = {"gaussian11": ssim_per_channel,
                         "uniform7": ssim_uniform_per_channel,
                         "gaussian11_valid": ssim_gaussian_valid_per_channel}[self.ssim_mode]
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


class ExactCompetitionLoss(nn.Module):
    """Optimize the organizer's mean-image SSIM/PSNR score directly."""

    def __init__(self, ssim_mode: str = "uniform7") -> None:
        super().__init__()
        if ssim_mode not in {"gaussian11", "uniform7"}:
            raise ValueError(f"Unsupported SSIM mode: {ssim_mode}")
        self.ssim_mode = ssim_mode

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_function = (
            ssim_per_channel
            if self.ssim_mode == "gaussian11"
            else ssim_uniform_per_channel
        )
        with torch.amp.autocast(prediction.device.type, enabled=False):
            prediction_float = prediction.float()
            target_float = target.float()
            ssim = ssim_function(prediction_float, target_float).mean()
            psnr = psnr_per_channel(prediction_float, target_float).clamp(max=50.0)
            normalized_score = 0.7 * ssim + 0.3 * (psnr / 50.0).mean()
        return 1.0 - normalized_score


def save_grayscale_rgb(array: np.ndarray, path: Path, quality: int = 75) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").convert("RGB").save(
        path, format="JPEG", quality=quality
    )
