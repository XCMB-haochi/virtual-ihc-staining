from __future__ import annotations

"""UNeXt2-style ConvNeXt V2 models for 2D virtual staining.

The architecture follows the UNeXt2 and dense FCMAE design published by the
Mehta Lab VisCy project (BSD-3-Clause) and the official ConvNeXt V2 FCMAE
example (MIT). This implementation is project-local and uses only PyTorch.

Sources:
https://github.com/mehta-lab/VisCy
https://github.com/facebookresearch/ConvNeXt-V2
"""

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ihc_spatial import (
    SPATIAL_CONDITION_DIMENSION,
    SPATIAL_FILM_HIDDEN_FEATURES,
    SPATIAL_THUMBNAIL_CHANNELS,
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
    SPATIAL_THUMBNAIL_SIDE,
    SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_TEXTURE_SIDE,
    SPATIAL_WHOLE_ROI_CHANNELS,
    SPATIAL_WHOLE_ROI_CONDITION_DIMENSION,
    SPATIAL_WHOLE_ROI_CONDITION_MODE,
    SPATIAL_WHOLE_ROI_EMBEDDING_FEATURES,
    SPATIAL_WHOLE_ROI_SIDE,
)


UNEXT2_ENCODER_BLOCKS = (3, 3, 9, 3)
UNEXT2_ENCODER_DIMS = (96, 192, 384, 768)
UNEXT2_BASE_ENCODER_BLOCKS = (3, 3, 27, 3)
UNEXT2_BASE_ENCODER_DIMS = (128, 256, 512, 1024)
UNEXT2_STEM_STRIDE = 2
UNEXT2_TOTAL_STRIDE = 16
UNEXT2_DECODER_BLOCKS = 2
UNEXT2_WIDE_HEAD_FEATURES = 48
UNEXT2_WIDE_HEAD_EXPANSION = 16
FCMAE_DECODER_BLOCKS = 1
FCMAE_MASK_RATIO = 0.5
OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE = (
    "official-convnextv2-fcmae-imagenet1k-tiny"
)
OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256 = (
    "b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef"
)
OFFICIAL_CONVNEXTV2_FCMAE_TINY_SOURCE = (
    "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/"
    "convnextv2_tiny_1k_224_fcmae.pt"
)
OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE = (
    "official-convnextv2-fcmae-imagenet1k-base"
)
OFFICIAL_CONVNEXTV2_FCMAE_BASE_SHA256 = (
    "cd59ca7d4db45a8d7bc2128dc447c0a5aee209d2b4920e79e1da4196afca6405"
)
OFFICIAL_CONVNEXTV2_FCMAE_BASE_SOURCE = (
    "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/"
    "convnextv2_base_1k_224_fcmae.pt"
)
OFFICIAL_CONVNEXTV2_FCMAE_LICENSE = "CC-BY-NC-4.0"
OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE = (
    "official-convnextv2-supervised-imagenet22k-to-1k-tiny"
)
OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_SHA256 = (
    "853d431aa9363f1b058e3c343d4bf2fca5fe2a4196621c381ddbcd4828290a96"
)
OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_SOURCE = (
    "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/"
    "convnextv2_tiny_22k_224_ema.pt"
)
CYTOLAND_VSCYTO2D_TINY_MODE = "biohub-cytoland-vscyto2d-unext2-tiny"
CYTOLAND_VSCYTO2D_TINY_SHA256 = (
    "1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe"
)
CYTOLAND_VSCYTO2D_TINY_SOURCE = (
    "https://huggingface.co/biohub/VSCyto2D/resolve/main/"
    "epoch%3D399-step%3D23200.ckpt"
)
CYTOLAND_VSCYTO2D_LICENSE = "BSD-3-Clause"
OFFICIAL_CONVNEXTV2_INITIALIZATION_SPECS = {
    OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE: {
        "sha256": OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256,
        "source_url": OFFICIAL_CONVNEXTV2_FCMAE_TINY_SOURCE,
        "license": OFFICIAL_CONVNEXTV2_FCMAE_LICENSE,
        "source_training": "self-supervised-fcmae-imagenet1k",
        "model_size": "tiny",
        "additional_skipped_source_tensors": (),
    },
    OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE: {
        "sha256": OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_SHA256,
        "source_url": OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_SOURCE,
        "license": OFFICIAL_CONVNEXTV2_FCMAE_LICENSE,
        "source_training": "supervised-imagenet22k-to-1k-ema",
        "model_size": "tiny",
        "additional_skipped_source_tensors": (
            "norm.weight",
            "norm.bias",
            "head.weight",
            "head.bias",
        ),
    },
    OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE: {
        "sha256": OFFICIAL_CONVNEXTV2_FCMAE_BASE_SHA256,
        "source_url": OFFICIAL_CONVNEXTV2_FCMAE_BASE_SOURCE,
        "license": OFFICIAL_CONVNEXTV2_FCMAE_LICENSE,
        "source_training": "self-supervised-fcmae-imagenet1k",
        "model_size": "base",
        "additional_skipped_source_tensors": (),
    },
}
# Backward-compatible public name used by existing Tiny audit tests.
OFFICIAL_CONVNEXTV2_TINY_INITIALIZATION_SPECS = {
    mode: specification
    for mode, specification in OFFICIAL_CONVNEXTV2_INITIALIZATION_SPECS.items()
    if specification["model_size"] == "tiny"
}
CYTOLAND_VSCYTO2D_INITIALIZATION_SPEC = {
    "sha256": CYTOLAND_VSCYTO2D_TINY_SHA256,
    "source_url": CYTOLAND_VSCYTO2D_TINY_SOURCE,
    "license": CYTOLAND_VSCYTO2D_LICENSE,
    "source_training": (
        "supervised-label-free-microscopy-to-nucleus-and-membrane-fluorescence"
    ),
    "model_size": "tiny",
}


def unext2_encoder_shape(model_size: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shapes = {
        "tiny": (UNEXT2_ENCODER_BLOCKS, UNEXT2_ENCODER_DIMS),
        "base": (UNEXT2_BASE_ENCODER_BLOCKS, UNEXT2_BASE_ENCODER_DIMS),
    }
    try:
        return shapes[model_size]
    except KeyError as error:
        raise ValueError(f"Unsupported UNeXt2 model size: {model_size!r}.") from error


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs.permute(0, 2, 3, 1)
        current = F.layer_norm(
            current,
            (self.weight.numel(),),
            self.weight,
            self.bias,
            self.eps,
        )
        return current.permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    """Channel-last ConvNeXt V2 global response normalization."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(inputs, ord=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return inputs + self.gamma * (inputs * normalized) + self.beta


def _resize_boolean_mask(
    mask: torch.Tensor,
    spatial_shape: Sequence[int],
) -> torch.Tensor:
    target_height, target_width = (int(value) for value in spatial_shape)
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.dtype != torch.bool:
        raise ValueError("Mask must be a boolean tensor with shape (batch, 1, H, W).")
    height, width = mask.shape[-2:]
    if target_height % height or target_width % width:
        raise ValueError(
            "Target feature shape must be divisible by the FCMAE mask shape: "
            f"{(target_height, target_width)} vs {(height, width)}."
        )
    if (height, width) == (target_height, target_width):
        return mask
    return mask.repeat_interleave(target_height // height, dim=-2).repeat_interleave(
        target_width // width,
        dim=-1,
    )


def generate_fcmae_mask(
    inputs: torch.Tensor,
    mask_ratio: float = FCMAE_MASK_RATIO,
    total_stride: int = UNEXT2_TOTAL_STRIDE,
) -> torch.Tensor:
    if inputs.ndim != 4:
        raise ValueError("FCMAE inputs must have shape (batch, channels, H, W).")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("FCMAE mask_ratio must be strictly between 0 and 1.")
    height, width = inputs.shape[-2:]
    if height % total_stride or width % total_stride:
        raise ValueError(
            f"FCMAE input shape {(height, width)} must be divisible by {total_stride}."
        )
    mask_height = height // total_stride
    mask_width = width // total_stride
    positions = mask_height * mask_width
    masked_positions = int(positions * mask_ratio)
    noise = torch.rand(inputs.shape[0], positions, device=inputs.device)
    order = noise.argsort(dim=1)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    mask.scatter_(1, order[:, :masked_positions], True)
    return mask.reshape(inputs.shape[0], 1, mask_height, mask_width)


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels: int, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("ConvNeXt-V2 drop path rate must be in [0, 1).")
        self.drop_path_rate = float(drop_path_rate)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, channels * 4)
        self.grn = GlobalResponseNorm(channels * 4)
        self.project = nn.Linear(channels * 4, channels)
        self.apply(_initialize_module)

    def forward(
        self,
        inputs: torch.Tensor,
        unmasked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if unmasked is not None:
            spatial_mask = _resize_boolean_mask(unmasked, inputs.shape[-2:])
            shortcut = inputs * spatial_mask
        else:
            spatial_mask = None
            shortcut = inputs
        current = self.depthwise(shortcut)
        if spatial_mask is not None:
            current = current * spatial_mask
        current = current.permute(0, 2, 3, 1)
        current = self.norm(current)
        current = self.expand(current)
        current = F.gelu(current)
        current = self.grn(current)
        current = self.project(current)
        current = current.permute(0, 3, 1, 2)
        if spatial_mask is not None:
            current = current * spatial_mask
        if self.training and self.drop_path_rate > 0.0:
            keep_probability = 1.0 - self.drop_path_rate
            keep = torch.rand(
                (current.shape[0], 1, 1, 1),
                device=current.device,
                dtype=current.dtype,
            ) < keep_probability
            current = current * keep / keep_probability
        return shortcut + current


class ConvNeXtV2Stage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> None:
        super().__init__()
        if stride == 1 and in_channels == out_channels:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=stride,
                    stride=stride,
                ),
            )
        self.blocks = nn.ModuleList(
            ConvNeXtV2Block(out_channels) for _ in range(blocks)
        )
        self.downsample.apply(_initialize_module)

    def forward(
        self,
        inputs: torch.Tensor,
        unmasked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if unmasked is not None:
            inputs = inputs * _resize_boolean_mask(unmasked, inputs.shape[-2:])
        current = self.downsample(inputs)
        if unmasked is not None:
            current = current * _resize_boolean_mask(unmasked, current.shape[-2:])
        for block in self.blocks:
            current = block(current, unmasked)
        return current


class FCMAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        blocks: Sequence[int] = UNEXT2_ENCODER_BLOCKS,
        dims: Sequence[int] = UNEXT2_ENCODER_DIMS,
    ) -> None:
        super().__init__()
        if len(blocks) != 4 or len(dims) != 4:
            raise ValueError("UNeXt2 encoder requires exactly four stages.")
        self.block_counts = tuple(int(value) for value in blocks)
        self.dimensions = tuple(int(value) for value in dims)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=UNEXT2_STEM_STRIDE,
                stride=UNEXT2_STEM_STRIDE,
            ),
            LayerNorm2d(dims[0]),
        )
        self.stages = nn.ModuleList(
            ConvNeXtV2Stage(
                self.dimensions[index - 1] if index else self.dimensions[0],
                self.dimensions[index],
                self.block_counts[index],
                stride=1 if index == 0 else 2,
            )
            for index in range(4)
        )
        self.stem.apply(_initialize_module)

    def configure_drop_path(self, maximum_rate: float) -> list[float]:
        if not 0.0 <= maximum_rate < 1.0:
            raise ValueError("Encoder drop path maximum must be in [0, 1).")
        encoder_blocks = [block for stage in self.stages for block in stage.blocks]
        rates = torch.linspace(0.0, float(maximum_rate), len(encoder_blocks)).tolist()
        for block, rate in zip(encoder_blocks, rates):
            block.drop_path_rate = float(rate)
        return [float(rate) for rate in rates]

    def forward(
        self,
        inputs: torch.Tensor,
        mask_ratio: float = 0.0,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("UNeXt2 expects inputs with shape (batch, 1, H, W).")
        if min(inputs.shape[-2:]) < UNEXT2_TOTAL_STRIDE:
            raise ValueError("UNeXt2 input height and width must be at least 16.")
        if inputs.shape[-2] % UNEXT2_TOTAL_STRIDE or inputs.shape[-1] % UNEXT2_TOTAL_STRIDE:
            raise ValueError("UNeXt2 input height and width must be divisible by 16.")
        mask = (
            generate_fcmae_mask(inputs, mask_ratio)
            if mask_ratio > 0.0
            else None
        )
        unmasked = None if mask is None else ~mask
        current = self.stem(inputs)
        if unmasked is not None:
            current = current * _resize_boolean_mask(unmasked, current.shape[-2:])
        features: list[torch.Tensor] = []
        for stage in self.stages:
            current = stage(current, unmasked)
            features.append(current)
        full_mask = None if mask is None else _resize_boolean_mask(mask, inputs.shape[-2:])
        return features, full_mask


def _icnr_initialize_linear(linear: nn.Linear, scale: int = 2) -> None:
    repeats = scale**2
    if linear.out_features % repeats:
        raise ValueError("ICNR output channels must be divisible by scale squared.")
    base = torch.empty(
        linear.out_features // repeats,
        linear.in_features,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    nn.init.trunc_normal_(base, std=0.02)
    with torch.no_grad():
        linear.weight.copy_(base.repeat_interleave(repeats, dim=0))
        if linear.bias is not None:
            linear.bias.zero_()


class PixelShuffleUpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        blocks: int,
    ) -> None:
        super().__init__()
        if in_channels % 4:
            raise ValueError("Pixel-shuffle input channels must be divisible by four.")
        up_channels = in_channels // 4
        self.upsample = nn.PixelShuffle(2)
        self.projection = nn.Sequential(
            LayerNorm2d(up_channels + skip_channels),
            nn.Conv2d(up_channels + skip_channels, out_channels, kernel_size=1),
        )
        self.blocks = nn.Sequential(
            *(ConvNeXtV2Block(out_channels) for _ in range(blocks))
        )
        self.projection.apply(_initialize_module)
        _icnr_initialize_linear(self.blocks[-1].project)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        current = self.upsample(inputs)
        if current.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                "UNeXt2 decoder and skip shapes disagree: "
                f"{tuple(current.shape[-2:])} vs {tuple(skip.shape[-2:])}."
            )
        current = self.projection(torch.cat((current, skip), dim=1))
        return self.blocks(current)


class UNeXt2Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        blocks_per_stage: int,
        encoder_dims: Sequence[int] = UNEXT2_ENCODER_DIMS,
    ) -> None:
        super().__init__()
        if len(encoder_dims) != 4:
            raise ValueError("UNeXt2 decoder requires four encoder dimensions.")
        channels = (
            encoder_dims[-1],
            encoder_dims[-2],
            encoder_dims[-3],
            out_channels * 4,
        )
        skip_channels = (
            encoder_dims[-2],
            encoder_dims[-3],
            encoder_dims[-4],
        )
        self.stages = nn.ModuleList(
            PixelShuffleUpStage(
                channels[index],
                skip_channels[index],
                channels[index + 1],
                blocks_per_stage,
            )
            for index in range(3)
        )
        self.output_shuffle = nn.PixelShuffle(2)

    def forward_half(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("UNeXt2 decoder requires four encoder feature maps.")
        current = features[-1]
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            current = stage(current, skip)
        return current

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.output_shuffle(self.forward_half(features))


class UNeXt2FeatureDecoder(nn.Module):
    """Decode to a fixed half-resolution feature map for a separate head."""

    def __init__(
        self,
        feature_channels: int = UNEXT2_WIDE_HEAD_FEATURES,
        blocks_per_stage: int = UNEXT2_DECODER_BLOCKS,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or feature_channels % 4:
            raise ValueError(
                "UNeXt2 feature decoder channels must be positive and divisible by four."
            )
        channels = (
            UNEXT2_ENCODER_DIMS[-1],
            UNEXT2_ENCODER_DIMS[-2],
            UNEXT2_ENCODER_DIMS[-3],
            feature_channels,
        )
        skip_channels = (
            UNEXT2_ENCODER_DIMS[-2],
            UNEXT2_ENCODER_DIMS[-3],
            UNEXT2_ENCODER_DIMS[-4],
        )
        self.stages = nn.ModuleList(
            PixelShuffleUpStage(
                channels[index],
                skip_channels[index],
                channels[index + 1],
                blocks_per_stage,
            )
            for index in range(3)
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("UNeXt2 decoder requires four encoder feature maps.")
        current = features[-1]
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            current = stage(current, skip)
        return current


class UNeXt2WideOutputHead(nn.Module):
    """Upsample a shared feature map before marker-specific output projection."""

    def __init__(
        self,
        out_channels: int,
        feature_channels: int = UNEXT2_WIDE_HEAD_FEATURES,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("UNeXt2 output channels must be positive.")
        if feature_channels % 4:
            raise ValueError("UNeXt2 wide-head features must be divisible by four.")
        shuffled_channels = feature_channels // 4
        hidden_channels = out_channels * UNEXT2_WIDE_HEAD_EXPANSION
        self.upsample = nn.PixelShuffle(2)
        self.projection = nn.Sequential(
            nn.Conv2d(shuffled_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.projection.apply(_initialize_module)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(self.upsample(features))


class D4InvariantThumbnailFiLM(nn.Module):
    """Existing spatial-thumbnail condition adapted to a UNeXt2 bottleneck."""

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
        self.stats_input = nn.Linear(
            SPATIAL_CONDITION_DIMENSION,
            SPATIAL_FILM_HIDDEN_FEATURES,
        )
        self.stats_output = nn.Linear(SPATIAL_FILM_HIDDEN_FEATURES, channels * 2)
        thumbnail_channels = (16, 32, 64, SPATIAL_THUMBNAIL_EMBEDDING_FEATURES)
        layers: list[nn.Module] = []
        in_channels = self.thumbnail_channels
        for out_channels in thumbnail_channels:
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.thumbnail_encoder = nn.Sequential(*layers)
        self.thumbnail_output = nn.Linear(
            SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
            channels * 2,
        )
        nn.init.zeros_(self.stats_output.weight)
        nn.init.zeros_(self.stats_output.bias)
        nn.init.zeros_(self.thumbnail_output.weight)
        nn.init.zeros_(self.thumbnail_output.bias)

    def _thumbnail_embedding(self, thumbnail: torch.Tensor) -> torch.Tensor:
        transformed = []
        for transform_index in range(8):
            current = torch.rot90(thumbnail, transform_index % 4, dims=(-2, -1))
            if transform_index >= 4:
                current = torch.flip(current, dims=(-1,))
            transformed.append(current)
        batch_size = thumbnail.shape[0]
        encoded = self.thumbnail_encoder(torch.cat(transformed, dim=0)).flatten(1)
        return encoded.reshape(8, batch_size, -1).mean(dim=0)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        stats = condition[:, :SPATIAL_CONDITION_DIMENSION]
        thumbnail = condition[:, SPATIAL_CONDITION_DIMENSION:].reshape(
            -1,
            self.thumbnail_channels,
            self.thumbnail_side,
            self.thumbnail_side,
        )
        affine = self.stats_output(F.silu(self.stats_input(stats)))
        affine = affine + self.thumbnail_output(self._thumbnail_embedding(thumbnail))
        scale, shift = affine.chunk(2, dim=1)
        return features * (1.0 + scale[..., None, None]) + shift[..., None, None]


class D4InvariantTextureThumbnailFiLM(D4InvariantThumbnailFiLM):
    """D4-invariant FiLM for mean, contrast, gradient, and presence maps."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            channels,
            thumbnail_channels=SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
            thumbnail_side=SPATIAL_THUMBNAIL_TEXTURE_SIDE,
        )


class D4InvariantWholeRoiFiLM(D4InvariantThumbnailFiLM):
    """Add target-aware low-frequency context from the complete DAPI ROI."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)
        encoder_channels = (16, 32, 64, SPATIAL_WHOLE_ROI_EMBEDDING_FEATURES)
        layers: list[nn.Module] = []
        in_channels = SPATIAL_WHOLE_ROI_CHANNELS
        for out_channels in encoder_channels:
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.whole_roi_encoder = nn.Sequential(*layers)
        self.whole_roi_output = nn.Linear(
            SPATIAL_WHOLE_ROI_EMBEDDING_FEATURES,
            channels * 2,
        )
        nn.init.zeros_(self.whole_roi_output.weight)
        nn.init.zeros_(self.whole_roi_output.bias)

    def _whole_roi_embedding(self, thumbnail: torch.Tensor) -> torch.Tensor:
        transformed = []
        for transform_index in range(8):
            current = torch.rot90(thumbnail, transform_index % 4, dims=(-2, -1))
            if transform_index >= 4:
                current = torch.flip(current, dims=(-1,))
            transformed.append(current)
        batch_size = thumbnail.shape[0]
        encoded = self.whole_roi_encoder(torch.cat(transformed, dim=0)).flatten(1)
        return encoded.reshape(8, batch_size, -1).mean(dim=0)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        stats = condition[:, :SPATIAL_CONDITION_DIMENSION]
        local_end = SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        local_thumbnail = condition[
            :, SPATIAL_CONDITION_DIMENSION:local_end
        ].reshape(
            -1,
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        )
        whole_roi_thumbnail = condition[:, local_end:].reshape(
            -1,
            SPATIAL_WHOLE_ROI_CHANNELS,
            SPATIAL_WHOLE_ROI_SIDE,
            SPATIAL_WHOLE_ROI_SIDE,
        )
        affine = self.stats_output(F.silu(self.stats_input(stats)))
        affine = affine + self.thumbnail_output(
            self._thumbnail_embedding(local_thumbnail)
        )
        affine = affine + self.whole_roi_output(
            self._whole_roi_embedding(whole_roi_thumbnail)
        )
        scale, shift = affine.chunk(2, dim=1)
        return features * (1.0 + scale[..., None, None]) + shift[..., None, None]


class UNeXt2VirtualStainer(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
        wide_output_head: bool = False,
        model_size: str = "tiny",
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if condition_mode not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }:
            raise ValueError(
                "UNeXt2 received an unsupported conditioning mode."
            )
        self.condition_mode = condition_mode
        self.model_size = model_size
        encoder_blocks, encoder_dims = unext2_encoder_shape(model_size)
        condition_dimensions = {
            "none": 0,
            SPATIAL_THUMBNAIL_CONDITION_MODE: SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE: SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
            SPATIAL_WHOLE_ROI_CONDITION_MODE: SPATIAL_WHOLE_ROI_CONDITION_DIMENSION,
        }
        self.condition_dimension = condition_dimensions[condition_mode]
        self.encoder = FCMAEEncoder(blocks=encoder_blocks, dims=encoder_dims)
        self.wide_output_head = bool(wide_output_head)
        legacy_decoder = UNeXt2Decoder(
            out_channels,
            UNEXT2_DECODER_BLOCKS,
            encoder_dims=encoder_dims,
        )
        if not self.wide_output_head:
            self.decoder: nn.Module = legacy_decoder
        if condition_mode == SPATIAL_THUMBNAIL_CONDITION_MODE:
            self.condition_film: nn.Module | None = D4InvariantThumbnailFiLM(
                encoder_dims[-1]
            )
        elif condition_mode == SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE:
            self.condition_film = D4InvariantTextureThumbnailFiLM(
                encoder_dims[-1]
            )
        elif condition_mode == SPATIAL_WHOLE_ROI_CONDITION_MODE:
            self.condition_film = D4InvariantWholeRoiFiLM(
                encoder_dims[-1]
            )
        else:
            self.condition_film = None
        if self.wide_output_head:
            if model_size != "tiny":
                raise ValueError("The wide UNeXt2 output head supports Tiny only.")
            # Match the legacy RNG path through shared FiLM initialization.
            del legacy_decoder
            self.decoder = UNeXt2FeatureDecoder()
            self.output_head: nn.Module = UNeXt2WideOutputHead(out_channels)
        else:
            self.output_head = nn.Identity()
        self.output_bias = nn.Parameter(torch.full((out_channels,), -2.0))

    def configure_encoder_drop_path(self, maximum_rate: float) -> list[float]:
        return self.encoder.configure_drop_path(maximum_rate)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.decoder.parameters()
        yield from self.output_head.parameters()
        if self.condition_film is not None:
            yield from self.condition_film.parameters()
        yield self.output_bias

    def _validate_condition(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.condition_mode == "none":
            if condition is not None:
                raise ValueError("Unconditioned UNeXt2 received a condition tensor.")
            return None
        if condition is None:
            raise ValueError("Conditioned UNeXt2 requires a condition tensor.")
        expected = (inputs.shape[0], self.condition_dimension)
        if tuple(condition.shape) != expected:
            raise ValueError(
                f"UNeXt2 condition shape must be {expected}, got {tuple(condition.shape)}."
            )
        condition = condition.to(device=inputs.device, dtype=torch.float32)
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("UNeXt2 condition must contain only finite values.")
        return condition

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_encoder_features(inputs, condition)[0]

    def forward_with_encoder_features(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the pre-FiLM bottleneck for training-only representation losses."""
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        encoder_features = features[-1]
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)
        logits = self.output_head(self.decoder(features))
        prediction = torch.sigmoid(logits + self.output_bias[None, :, None, None])
        return prediction, encoder_features

    def forward_with_decoder_only_auxiliary_gradients(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Block auxiliary-channel gradients before the shared decoder."""
        if self.output_bias.numel() < 2:
            raise ValueError(
                "Decoder-only auxiliary gradients require at least two outputs."
            )
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)

        primary_logits = self.output_head(self.decoder(features))
        detached_features = [feature.detach() for feature in features]
        auxiliary_logits = self.output_head(self.decoder(detached_features))
        logits = torch.cat(
            (primary_logits[:, :1], auxiliary_logits[:, 1:]),
            dim=1,
        )
        return torch.sigmoid(logits + self.output_bias[None, :, None, None])


class UNeXt2LocalConditionVirtualStainer(UNeXt2VirtualStainer):
    """Ablate explicit whole-ROI geometry/statistics, retaining local context."""
    def __init__(self, out_channels=1, condition_mode="spatial-thumbnail-film"):
        if condition_mode != "spatial-thumbnail-film":
            raise ValueError('Local-condition ablation requires ordinary spatial thumbnail')
        super().__init__(out_channels=out_channels,condition_mode=condition_mode)

    def _validate_condition(self,inputs,condition):
        condition=super()._validate_condition(inputs,condition).clone()
        # First9 entries are patch statistics; next11 are global ROI descriptors.
        condition[:,9:20]=0
        return condition


def unext2_local_condition_model_specification():
    specification=dict(unext2_model_specification())
    specification['architecture']='unext2-fcmae-tiny-local-condition'
    specification['masked_condition_features']=['row_position','col_position','distance_top',
        'distance_bottom','distance_left','distance_right','roi_occupancy','roi_mean_mean',
        'roi_mean_std','roi_std_mean','roi_std_std']
    return specification


class UNeXt2SignalAuxVirtualStainer(UNeXt2VirtualStainer):
    """Train-only local signal occupancy supervision at the DAPI bottleneck."""
    def __init__(self, out_channels=1, condition_mode="none"):
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        with torch.random.fork_rng(devices=[]):
            self.signal_head = nn.Sequential(nn.Conv2d(768, 64, 1), nn.GELU(),
                                             nn.Conv2d(64, out_channels, 1))
            nn.init.zeros_(self.signal_head[-1].weight)
            nn.init.constant_(self.signal_head[-1].bias, -3.0)

    def decoder_parameters(self):
        yield from super().decoder_parameters()
        yield from self.signal_head.parameters()

    def forward_with_signal_logits(self, inputs, condition=None):
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        signal_logits = self.signal_head(features[-1])
        if self.condition_film is not None:
            features[-1] = self.condition_film(features[-1], condition)
        logits = self.output_head(self.decoder(features))
        prediction = torch.sigmoid(logits + self.output_bias[None, :, None, None])
        return prediction, signal_logits


def unext2_signal_aux_model_specification():
    specification = dict(unext2_model_specification())
    specification['architecture'] = 'unext2-fcmae-tiny-signal-aux'
    specification['training_auxiliary_head'] = {'stage': 4, 'hidden_channels': 64,
                                               'output': 'local_signal_occupancy'}
    return specification


class UNeXt2WideHeadVirtualStainer(UNeXt2VirtualStainer):
    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            condition_mode=condition_mode,
            wide_output_head=True,
        )


class UNeXt2BaseVirtualStainer(UNeXt2VirtualStainer):
    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            condition_mode=condition_mode,
            model_size="base",
        )


class UNeXt2CrossMarkerFeedbackVirtualStainer(UNeXt2VirtualStainer):
    """Tiny UNeXt2 with a zero-start auxiliary-to-primary residual head.

    The auxiliary marker logits are available during inference because all four
    markers are predicted from DAPI in one forward pass.  The final feedback
    projection is zero-initialized, so a fresh candidate has exactly the same
    output as the ordinary shared decoder until the new path learns a useful
    correction.
    """

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
        feedback_hidden_channels: int = 32,
    ) -> None:
        if out_channels < 2:
            raise ValueError(
                "Cross-marker feedback requires at least two output markers."
            )
        if feedback_hidden_channels <= 0:
            raise ValueError("feedback_hidden_channels must be positive.")
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        self.feedback_hidden_channels = int(feedback_hidden_channels)
        self.feedback_head = nn.Sequential(
            nn.Conv2d(out_channels - 1, self.feedback_hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.feedback_hidden_channels,
                self.feedback_hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(self.feedback_hidden_channels, 1, 3, padding=1),
        )
        self.feedback_head.apply(_initialize_module)
        # Keep the architecture an exact control at initialization.
        final_projection = self.feedback_head[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise RuntimeError("Cross-marker feedback head has an invalid output layer.")
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().decoder_parameters()
        yield from self.feedback_head.parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)
        logits = self.output_head(self.decoder(features))
        auxiliary_logits = logits[:, 1:]
        primary_residual = self.feedback_head(auxiliary_logits)
        logits = torch.cat((logits[:, :1] + primary_residual, auxiliary_logits), dim=1)
        return torch.sigmoid(logits + self.output_bias[None, :, None, None])


class UNeXt2DapiDetailResidualVirtualStainer(UNeXt2VirtualStainer):
    """UNeXt2 with a bounded, zero-start DAPI high-frequency logit residual."""

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
        detail_hidden_channels: int = 32,
        residual_scale: float = 0.25,
    ) -> None:
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if detail_hidden_channels <= 0:
            raise ValueError("detail_hidden_channels must be positive.")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1].")
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        self.detail_hidden_channels = int(detail_hidden_channels)
        self.detail_residual_scale = float(residual_scale)
        self.detail_head = nn.Sequential(
            nn.Conv2d(1, self.detail_hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.detail_hidden_channels,
                self.detail_hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(self.detail_hidden_channels, 1, 3, padding=1),
        )
        self.detail_head.apply(_initialize_module)
        final_projection = self.detail_head[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise RuntimeError("DAPI detail head has an invalid output layer.")
        # Preserve exact parent behavior until the branch learns a correction.
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    @staticmethod
    def _high_pass(inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (2, 2, 2, 2), mode="reflect")
        smooth = F.avg_pool2d(padded, kernel_size=5, stride=1)
        return inputs - smooth

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().decoder_parameters()
        yield from self.detail_head.parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)
        logits = self.output_head(self.decoder(features))
        detail_residual = self.detail_head(self._high_pass(inputs))
        primary_logits = logits[:, :1] + self.detail_residual_scale * torch.tanh(
            detail_residual
        )
        logits = torch.cat((primary_logits, logits[:, 1:]), dim=1)
        return torch.sigmoid(logits + self.output_bias[None, :, None, None])


class UNeXt2ShallowDapiSkipVirtualStainer(UNeXt2VirtualStainer):
    """UNeXt2 with a zero-gated DAPI shallow feature skip before PixelShuffle."""

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
        shallow_hidden_channels: int = 32,
    ) -> None:
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if shallow_hidden_channels <= 0:
            raise ValueError("shallow_hidden_channels must be positive.")
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        self.shallow_hidden_channels = int(shallow_hidden_channels)
        self.shallow_encoder = nn.Sequential(
            nn.Conv2d(1, self.shallow_hidden_channels, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, self.shallow_hidden_channels), self.shallow_hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                self.shallow_hidden_channels,
                self.shallow_hidden_channels,
                3,
                padding=1,
            ),
            nn.GELU(),
        )
        # The primary output occupies the first four pre-shuffle channels.
        self.shallow_projection = nn.Conv2d(
            self.shallow_hidden_channels,
            4,
            1,
        )
        self.shallow_encoder.apply(_initialize_module)
        self.shallow_projection.apply(_initialize_module)
        nn.init.zeros_(self.shallow_projection.weight)
        nn.init.zeros_(self.shallow_projection.bias)

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().decoder_parameters()
        yield from self.shallow_encoder.parameters()
        yield from self.shallow_projection.parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)
        half_logits = self.decoder.forward_half(features)
        shallow = self.shallow_projection(self.shallow_encoder(inputs))
        if half_logits.shape[-2:] != shallow.shape[-2:]:
            raise RuntimeError(
                "DAPI shallow skip and decoder feature shapes disagree: "
                f"{tuple(shallow.shape[-2:])} vs {tuple(half_logits.shape[-2:])}."
            )
        if half_logits.shape[1] < 4:
            raise RuntimeError("UNeXt2 decoder has no primary pre-shuffle channels.")
        half_logits = torch.cat(
            (half_logits[:, :4] + shallow, half_logits[:, 4:]),
            dim=1,
        )
        logits = self.decoder.output_shuffle(half_logits)
        return torch.sigmoid(logits + self.output_bias[None, :, None, None])


class UNeXt2MarkerDecoderVirtualStainer(UNeXt2VirtualStainer):
    """Share the encoder while giving every marker an independent decoder."""

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
    ) -> None:
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        # Building the primary path first preserves the single-marker RNG path.
        super().__init__(out_channels=1, condition_mode=condition_mode)
        self.auxiliary_decoders = nn.ModuleList(
            UNeXt2Decoder(1, UNEXT2_DECODER_BLOCKS)
            for _ in range(out_channels - 1)
        )
        self.output_bias = nn.Parameter(torch.full((out_channels,), -2.0))

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.decoder.parameters()
        yield from self.auxiliary_decoders.parameters()
        if self.condition_film is not None:
            yield from self.condition_film.parameters()
        yield self.output_bias

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._validate_condition(inputs, condition)
        features, _ = self.encoder(inputs)
        if self.condition_film is not None:
            if condition is None:
                raise RuntimeError("Conditioned UNeXt2 lost its condition tensor.")
            features[-1] = self.condition_film(features[-1], condition)
        logits = [self.decoder(features)]
        logits.extend(decoder(features) for decoder in self.auxiliary_decoders)
        combined = torch.cat(logits, dim=1)
        return torch.sigmoid(combined + self.output_bias[None, :, None, None])


class DapiFCMAE(nn.Module):
    def __init__(self, model_size: str = "tiny") -> None:
        super().__init__()
        self.model_size = model_size
        encoder_blocks, encoder_dims = unext2_encoder_shape(model_size)
        self.encoder = FCMAEEncoder(blocks=encoder_blocks, dims=encoder_dims)
        self.decoder = UNeXt2Decoder(
            1,
            FCMAE_DECODER_BLOCKS,
            encoder_dims=encoder_dims,
        )
        self.output_bias = nn.Parameter(torch.tensor([-2.0]))

    def forward(
        self,
        inputs: torch.Tensor,
        mask_ratio: float = FCMAE_MASK_RATIO,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = self.encoder(inputs, mask_ratio=mask_ratio)
        if mask is None:
            raise RuntimeError("DAPI FCMAE requires a positive mask ratio.")
        logits = self.decoder(features)
        reconstruction = torch.sigmoid(
            logits + self.output_bias[None, :, None, None]
        )
        return reconstruction, mask


def unext2_encoder_specification(model_size: str = "tiny") -> dict[str, object]:
    encoder_blocks, encoder_dims = unext2_encoder_shape(model_size)
    return {
        "version": 1,
        "architecture": f"convnext-v2-{model_size}-dense-fcmae",
        "input_channels": 1,
        "stem_kernel": 2,
        "stem_stride": 2,
        "stage_blocks": list(encoder_blocks),
        "stage_dims": list(encoder_dims),
        "total_stride": UNEXT2_TOTAL_STRIDE,
        "block": "convnext-v2-grn",
    }


def unext2_encoder_specification_from_encoder(
    encoder: FCMAEEncoder,
) -> dict[str, object]:
    for model_size in ("tiny", "base"):
        blocks, dims = unext2_encoder_shape(model_size)
        if encoder.block_counts == blocks and encoder.dimensions == dims:
            return unext2_encoder_specification(model_size)
    raise ValueError("FCMAE encoder shape is not a supported UNeXt2 size.")


def unext2_model_specification(model_size: str = "tiny") -> dict[str, object]:
    _, encoder_dims = unext2_encoder_shape(model_size)
    return {
        "version": 1,
        "architecture": f"unext2-fcmae-{model_size}",
        "encoder": unext2_encoder_specification(model_size),
        "decoder": {
            "mode": "pixelshuffle",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [encoder_dims[-2], encoder_dims[-3], "4*out_channels"],
            "output_activation": "sigmoid-with-channel-bias",
        },
    }


def unext2_widehead_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-widehead",
        "encoder": unext2_encoder_specification(),
        "decoder": {
            "mode": "pixelshuffle-feature-decoder",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [384, 192, UNEXT2_WIDE_HEAD_FEATURES],
        },
        "output_head": {
            "mode": "pixelshuffle-conv",
            "scale": 2,
            "feature_channels": UNEXT2_WIDE_HEAD_FEATURES,
            "hidden_channels": f"{UNEXT2_WIDE_HEAD_EXPANSION}*out_channels",
            "output_activation": "sigmoid-with-channel-bias",
        },
    }


def unext2_marker_decoders_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-marker-decoders",
        "encoder": unext2_encoder_specification(),
        "decoder": {
            "mode": "independent-pixelshuffle-per-marker",
            "shared_parameters": "encoder-and-conditioning-only",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [384, 192, 4],
            "output_activation": "sigmoid-with-channel-bias",
        },
    }


def unext2_cross_marker_feedback_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-cross-marker-feedback",
        "encoder": unext2_encoder_specification(),
        "decoder": {
            "mode": "pixelshuffle-cross-marker-feedback",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [384, 192, "4*out_channels"],
            "output_activation": "sigmoid-with-channel-bias",
        },
        "feedback": {
            "mode": "auxiliary-logit-residual",
            "hidden_channels": 32,
            "layers": 2,
            "zero_initialized_output": True,
        },
    }


def unext2_dapi_detail_residual_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-dapi-detail-residual",
        "encoder": unext2_encoder_specification(),
        "decoder": {
            "mode": "pixelshuffle-dapi-detail-residual",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [384, 192, "4*out_channels"],
            "output_activation": "sigmoid-with-channel-bias",
        },
        "detail_residual": {
            "mode": "dapi-high-pass-logit-residual",
            "kernel_size": 5,
            "hidden_channels": 32,
            "residual_scale": 0.25,
            "zero_initialized_output": True,
        },
    }


def unext2_shallow_dapi_skip_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-shallow-dapi-skip",
        "encoder": unext2_encoder_specification(),
        "decoder": {
            "mode": "pixelshuffle-zero-gated-shallow-dapi-skip",
            "blocks_per_stage": UNEXT2_DECODER_BLOCKS,
            "stage_channels": [384, 192, "4*out_channels"],
            "output_activation": "sigmoid-with-channel-bias",
        },
        "shallow_skip": {
            "mode": "raw-dapi-stride2-feature-to-primary-pre-shuffle",
            "hidden_channels": 32,
            "kernel_size": 5,
            "zero_initialized_projection": True,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_convnextv2_fcmae_tiny_state_mapping() -> dict[str, str]:
    """Map official ConvNeXt-V2 Tiny FCMAE encoder keys to this encoder."""

    return official_convnextv2_fcmae_state_mapping("tiny")


def official_convnextv2_fcmae_state_mapping(model_size: str) -> dict[str, str]:
    """Map official ConvNeXt-V2 FCMAE encoder keys to a local encoder size."""

    encoder_blocks, _ = unext2_encoder_shape(model_size)
    mapping = {
        "downsample_layers.0.1.weight": "stem.1.weight",
        "downsample_layers.0.1.bias": "stem.1.bias",
    }
    for stage in range(1, 4):
        for index in range(2):
            for parameter in ("weight", "bias"):
                mapping[
                    f"downsample_layers.{stage}.{index}.{parameter}"
                ] = f"stages.{stage}.downsample.{index}.{parameter}"
    block_fields = {
        "dwconv.weight": "depthwise.weight",
        "dwconv.bias": "depthwise.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
        "pwconv1.weight": "expand.weight",
        "pwconv1.bias": "expand.bias",
        "grn.gamma": "grn.gamma",
        "grn.beta": "grn.beta",
        "pwconv2.weight": "project.weight",
        "pwconv2.bias": "project.bias",
    }
    for stage, block_count in enumerate(encoder_blocks):
        for block in range(block_count):
            for source_field, target_field in block_fields.items():
                mapping[
                    f"stages.{stage}.{block}.{source_field}"
                ] = f"stages.{stage}.blocks.{block}.{target_field}"
    return mapping


def cytoland_vscyto2d_encoder_state_mapping() -> dict[str, str]:
    """Map Biohub's public VSCyto2D UNeXt2 encoder to this Tiny encoder."""

    mapping = {
        "model.encoder.stem.conv2d.weight": "stem.0.weight",
        "model.encoder.stem.conv2d.bias": "stem.0.bias",
        "model.encoder.stem.norm.weight": "stem.1.weight",
        "model.encoder.stem.norm.bias": "stem.1.bias",
    }
    for stage in range(1, 4):
        for index in range(2):
            for parameter in ("weight", "bias"):
                mapping[
                    f"model.encoder.stages.{stage}.downsample.{index}.{parameter}"
                ] = f"stages.{stage}.downsample.{index}.{parameter}"
    block_fields = {
        "dwconv.weight": "depthwise.weight",
        "layernorm.weight": "norm.weight",
        "layernorm.bias": "norm.bias",
        "mlp.fc1.weight": "expand.weight",
        "mlp.fc1.bias": "expand.bias",
        "mlp.grn.weight": "grn.gamma",
        "mlp.grn.bias": "grn.beta",
        "mlp.fc2.weight": "project.weight",
        "mlp.fc2.bias": "project.bias",
    }
    for stage, block_count in enumerate(UNEXT2_ENCODER_BLOCKS):
        for block in range(block_count):
            for source_field, target_field in block_fields.items():
                mapping[
                    f"model.encoder.stages.{stage}.blocks.{block}.{source_field}"
                ] = f"stages.{stage}.blocks.{block}.{target_field}"
    return mapping


def cytoland_vscyto2d_decoder_state_mapping(
    stages: Sequence[int] = (0, 1),
) -> dict[str, str]:
    """Map compatible Cytoland decoder stages to the local PixelShuffle decoder."""

    if any(stage not in {0, 1, 2} for stage in stages) or len(set(stages)) != len(
        stages
    ):
        raise ValueError("Cytoland decoder stages must be unique values from 0 to 2.")
    mapping: dict[str, str] = {}
    for stage in stages:
        prefix = f"model.decoder.decoder_stages.{stage}.conv"
        target_prefix = f"stages.{stage}"
        for parameter in ("weight", "bias"):
            mapping[f"{prefix}.downsample.0.{parameter}"] = (
                f"{target_prefix}.projection.0.{parameter}"
            )
            mapping[f"{prefix}.downsample.1.{parameter}"] = (
                f"{target_prefix}.projection.1.{parameter}"
            )
        block_fields = {
            "conv_dw.weight": "depthwise.weight",
            "conv_dw.bias": "depthwise.bias",
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
            "mlp.fc1.weight": "expand.weight",
            "mlp.fc1.bias": "expand.bias",
            "mlp.grn.weight": "grn.gamma",
            "mlp.grn.bias": "grn.beta",
            "mlp.fc2.weight": "project.weight",
            "mlp.fc2.bias": "project.bias",
        }
        for block in range(UNEXT2_DECODER_BLOCKS):
            for source_field, target_field in block_fields.items():
                mapping[f"{prefix}.blocks.{block}.{source_field}"] = (
                    f"{target_prefix}.blocks.{block}.{target_field}"
                )
    return mapping


def initialize_fcmae_encoder_from_official_convnextv2(
    encoder: FCMAEEncoder,
    checkpoint_path: Path,
    expected_sha256: str = OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256,
    mode: str = OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE,
) -> dict[str, object]:
    """Initialize compatible encoder tensors from an audited official model."""

    if not isinstance(encoder, FCMAEEncoder):
        raise TypeError("Official ConvNeXt-V2 initialization requires FCMAEEncoder.")
    source_spec = OFFICIAL_CONVNEXTV2_INITIALIZATION_SPECS.get(mode)
    if source_spec is None:
        raise ValueError(f"Unsupported official ConvNeXt-V2 mode: {mode!r}.")
    checkpoint_path = checkpoint_path.resolve()
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Expected official checkpoint SHA-256 is invalid.")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(
            "Official ConvNeXt-V2 checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {checkpoint_sha256}."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"model"}:
        raise ValueError("Unsupported official ConvNeXt-V2 FCMAE checkpoint format.")
    source_state = checkpoint["model"]
    if not isinstance(source_state, Mapping):
        raise ValueError("Official ConvNeXt-V2 checkpoint has no model state.")

    model_size = str(source_spec["model_size"])
    expected_encoder_spec = unext2_encoder_specification(model_size)
    if unext2_encoder_specification_from_encoder(encoder) != expected_encoder_spec:
        raise ValueError(
            f"Official ConvNeXt-V2 {model_size} initialization requires a "
            f"matching {model_size} encoder."
        )
    state_mapping = official_convnextv2_fcmae_state_mapping(model_size)
    skipped_source = {
        "downsample_layers.0.0.weight",
        "downsample_layers.0.0.bias",
    } | set(source_spec["additional_skipped_source_tensors"])
    if set(source_state) != set(state_mapping) | skipped_source:
        missing = sorted((set(state_mapping) | skipped_source) - set(source_state))
        unexpected = sorted(set(source_state) - set(state_mapping) - skipped_source)
        raise ValueError(
            f"Official ConvNeXt-V2 state keys do not match {model_size} FCMAE: "
            f"missing={missing}, unexpected={unexpected}."
        )

    target_state = encoder.state_dict()
    preserved_target = {"stem.0.weight", "stem.0.bias"}
    if set(target_state) != set(state_mapping.values()) | preserved_target:
        raise RuntimeError("Project FCMAE encoder state no longer matches the mapping.")

    updated_state = dict(target_state)
    reshaped: list[str] = []
    loaded_parameters = 0
    for source_key, target_key in state_mapping.items():
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f"Official state value is not a tensor: {source_key}.")
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"Official tensor dtype mismatch for {source_key}: "
                f"{source_tensor.dtype} vs {target_tensor.dtype}."
            )
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            loaded = source_tensor
        elif (
            source_key.endswith(("grn.gamma", "grn.beta"))
            and source_tensor.numel() == target_tensor.numel()
        ):
            loaded = source_tensor.reshape_as(target_tensor)
            reshaped.append(source_key)
        else:
            raise ValueError(
                f"Official tensor shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(source_tensor.shape)} vs {tuple(target_tensor.shape)}."
            )
        updated_state[target_key] = loaded
        loaded_parameters += target_tensor.numel()

    encoder.load_state_dict(updated_state, strict=True)
    encoder_parameters = sum(tensor.numel() for tensor in target_state.values())
    return {
        "version": 1,
        "mode": mode,
        "scope": "encoder-except-input-convolution",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_url": source_spec["source_url"],
        "license": source_spec["license"],
        "source_training": source_spec["source_training"],
        "source_tensors": len(source_state),
        "state_tensors_loaded": len(state_mapping),
        "parameters_loaded": loaded_parameters,
        "encoder_parameters": encoder_parameters,
        "parameter_coverage": loaded_parameters / encoder_parameters,
        "reshaped_source_tensors": reshaped,
        "preserved_target_tensors": sorted(preserved_target),
        "skipped_source_tensors": sorted(skipped_source),
        "encoder_spec": expected_encoder_spec,
    }


def initialize_fcmae_encoder_from_cytoland_vscyto2d(
    encoder: FCMAEEncoder,
    checkpoint_path: Path,
    expected_sha256: str = CYTOLAND_VSCYTO2D_TINY_SHA256,
    mode: str = CYTOLAND_VSCYTO2D_TINY_MODE,
) -> dict[str, object]:
    """Initialize a Tiny encoder from Biohub's audited Cytoland checkpoint."""

    if not isinstance(encoder, FCMAEEncoder):
        raise TypeError("Cytoland initialization requires FCMAEEncoder.")
    if mode != CYTOLAND_VSCYTO2D_TINY_MODE:
        raise ValueError(f"Unsupported Cytoland initialization mode: {mode!r}.")
    expected_encoder_spec = unext2_encoder_specification("tiny")
    if unext2_encoder_specification_from_encoder(encoder) != expected_encoder_spec:
        raise ValueError("Cytoland VSCyto2D initialization requires a Tiny encoder.")

    checkpoint_path = checkpoint_path.resolve()
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("Expected Cytoland checkpoint SHA-256 is invalid.")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(
            "Cytoland checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {checkpoint_sha256}."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Unsupported Cytoland checkpoint format.")
    source_state = checkpoint.get("state_dict")
    if not isinstance(source_state, Mapping):
        raise ValueError("Cytoland checkpoint has no state_dict.")

    state_mapping = cytoland_vscyto2d_encoder_state_mapping()
    skipped_source = {
        "model.encoder.stem.conv3d.weight",
        "model.encoder.stem.conv3d.bias",
    }
    source_encoder_keys = {
        key for key in source_state if key.startswith("model.encoder.")
    }
    if source_encoder_keys != set(state_mapping) | skipped_source:
        missing = sorted((set(state_mapping) | skipped_source) - source_encoder_keys)
        unexpected = sorted(source_encoder_keys - set(state_mapping) - skipped_source)
        raise ValueError(
            "Cytoland encoder state keys do not match VSCyto2D: "
            f"missing={missing}, unexpected={unexpected}."
        )

    target_state = encoder.state_dict()
    preserved_target = {
        f"stages.{stage}.blocks.{block}.depthwise.bias"
        for stage, block_count in enumerate(UNEXT2_ENCODER_BLOCKS)
        for block in range(block_count)
    }
    if set(target_state) != set(state_mapping.values()) | preserved_target:
        raise RuntimeError("Project FCMAE encoder state no longer matches Cytoland.")

    updated_state = dict(target_state)
    loaded_parameters = 0
    for source_key, target_key in state_mapping.items():
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f"Cytoland state value is not a tensor: {source_key}.")
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"Cytoland tensor dtype mismatch for {source_key}: "
                f"{source_tensor.dtype} vs {target_tensor.dtype}."
            )
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Cytoland tensor shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(source_tensor.shape)} vs {tuple(target_tensor.shape)}."
            )
        updated_state[target_key] = source_tensor
        loaded_parameters += target_tensor.numel()

    encoder.load_state_dict(updated_state, strict=True)
    encoder_parameters = sum(tensor.numel() for tensor in target_state.values())
    source_spec = CYTOLAND_VSCYTO2D_INITIALIZATION_SPEC
    return {
        "version": 1,
        "mode": mode,
        "scope": "encoder-except-source-absent-depthwise-biases",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_url": source_spec["source_url"],
        "license": source_spec["license"],
        "source_training": source_spec["source_training"],
        "source_tensors": len(source_state),
        "source_encoder_tensors": len(source_encoder_keys),
        "state_tensors_loaded": len(state_mapping),
        "parameters_loaded": loaded_parameters,
        "encoder_parameters": encoder_parameters,
        "parameter_coverage": loaded_parameters / encoder_parameters,
        "preserved_target_tensors": sorted(preserved_target),
        "skipped_source_tensors": sorted(skipped_source),
        "encoder_spec": expected_encoder_spec,
    }


def initialize_unext2_from_cytoland_vscyto2d(
    model: nn.Module,
    checkpoint_path: Path,
    expected_sha256: str = CYTOLAND_VSCYTO2D_TINY_SHA256,
    mode: str = CYTOLAND_VSCYTO2D_TINY_MODE,
    scope: str = "encoder-decoder",
) -> dict[str, object]:
    """Load a fixed Cytoland encoder or encoder plus compatible decoder stages."""

    if not isinstance(model, UNeXt2VirtualStainer):
        raise TypeError("Cytoland supervised initialization requires UNeXt2.")
    if model.model_size != "tiny" or model.wide_output_head:
        raise ValueError("Cytoland supervised initialization requires narrow Tiny UNeXt2.")
    if model.output_bias.numel() != 1:
        raise ValueError("Cytoland supervised initialization requires one output marker.")
    if scope not in {"encoder", "encoder-decoder"}:
        raise ValueError(f"Unsupported Cytoland initialization scope: {scope!r}.")

    report = initialize_fcmae_encoder_from_cytoland_vscyto2d(
        model.encoder,
        checkpoint_path,
        expected_sha256,
        mode,
    )
    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location="cpu", weights_only=True
    )
    source_state = checkpoint["state_dict"]
    compatible_mapping = cytoland_vscyto2d_decoder_state_mapping((0, 1))
    skipped_mapping = cytoland_vscyto2d_decoder_state_mapping((2,))
    source_decoder_keys = {
        key for key in source_state if key.startswith("model.decoder.")
    }
    expected_source_decoder_keys = set(compatible_mapping) | set(skipped_mapping)
    if source_decoder_keys != expected_source_decoder_keys:
        missing = sorted(expected_source_decoder_keys - source_decoder_keys)
        unexpected = sorted(source_decoder_keys - expected_source_decoder_keys)
        raise ValueError(
            "Cytoland decoder state keys do not match VSCyto2D: "
            f"missing={missing}, unexpected={unexpected}."
        )

    decoder_parameters_loaded = 0
    reshaped_source_tensors: list[str] = []
    if scope == "encoder-decoder":
        if not isinstance(model.decoder, UNeXt2Decoder):
            raise TypeError("Cytoland decoder initialization requires UNeXt2Decoder.")
        target_state = model.decoder.state_dict()
        preserved_target = {
            key for key in target_state if key.startswith("stages.2.")
        }
        if set(target_state) != set(compatible_mapping.values()) | preserved_target:
            raise RuntimeError("Project UNeXt2 decoder no longer matches Cytoland.")
        updated_state = dict(target_state)
        for source_key, target_key in compatible_mapping.items():
            source_tensor = source_state[source_key]
            target_tensor = target_state[target_key]
            if not isinstance(source_tensor, torch.Tensor):
                raise ValueError(f"Cytoland state value is not a tensor: {source_key}.")
            if source_tensor.dtype != target_tensor.dtype:
                raise ValueError(
                    f"Cytoland tensor dtype mismatch for {source_key}: "
                    f"{source_tensor.dtype} vs {target_tensor.dtype}."
                )
            if tuple(source_tensor.shape) == tuple(target_tensor.shape):
                loaded = source_tensor
            elif (
                source_key.endswith(("mlp.fc1.weight", "mlp.fc2.weight"))
                and source_tensor.numel() == target_tensor.numel()
            ):
                loaded = source_tensor.reshape_as(target_tensor)
                reshaped_source_tensors.append(source_key)
            else:
                raise ValueError(
                    f"Cytoland tensor shape mismatch for {source_key} -> "
                    f"{target_key}: {tuple(source_tensor.shape)} vs "
                    f"{tuple(target_tensor.shape)}."
                )
            updated_state[target_key] = loaded
            decoder_parameters_loaded += target_tensor.numel()
        model.decoder.load_state_dict(updated_state, strict=True)

    return {
        **report,
        "scope": (
            "encoder-and-compatible-decoder-stages-0-1"
            if scope == "encoder-decoder"
            else "encoder"
        ),
        "supervised_transfer_scope": scope,
        "source_decoder_tensors": len(source_decoder_keys),
        "decoder_state_tensors_loaded": (
            len(compatible_mapping) if scope == "encoder-decoder" else 0
        ),
        "decoder_parameters_loaded": decoder_parameters_loaded,
        "decoder_reshaped_source_tensors": reshaped_source_tensors,
        "skipped_source_decoder_tensors": sorted(skipped_mapping),
        "preserved_target_components": [
            "decoder-stage-2",
            "conditioning",
            "output-bias",
        ],
    }


def initialize_fcmae_encoder_from_public_checkpoint(
    encoder: FCMAEEncoder,
    checkpoint_path: Path,
    expected_sha256: str,
    mode: str,
) -> dict[str, object]:
    """Dispatch one audited public checkpoint format to its strict loader."""

    if mode == CYTOLAND_VSCYTO2D_TINY_MODE:
        return initialize_fcmae_encoder_from_cytoland_vscyto2d(
            encoder,
            checkpoint_path,
            expected_sha256,
            mode,
        )
    return initialize_fcmae_encoder_from_official_convnextv2(
        encoder,
        checkpoint_path,
        expected_sha256,
        mode,
    )


def initialize_unext2_encoder_from_fcmae(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not isinstance(model, UNeXt2VirtualStainer):
        raise TypeError("DAPI FCMAE initialization requires UNeXt2VirtualStainer.")
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("version") != 1:
        raise ValueError("Unsupported DAPI FCMAE checkpoint format.")
    if checkpoint.get("task") != "dapi-fcmae-pretraining":
        raise ValueError("Checkpoint is not a DAPI FCMAE pretraining artifact.")
    expected_specification = unext2_encoder_specification(model.model_size)
    if checkpoint.get("encoder_spec") != expected_specification:
        raise ValueError("DAPI FCMAE encoder specification does not match UNeXt2.")
    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping):
        raise ValueError("DAPI FCMAE checkpoint is missing its encoder state.")
    model.encoder.load_state_dict(state, strict=True)
    checkpoint_config = checkpoint.get("config")
    pretraining_markers = (
        list(checkpoint_config.get("pretraining_markers", ["DAPI"]))
        if isinstance(checkpoint_config, Mapping)
        else ["DAPI"]
    )
    return {
        "version": 1,
        "mode": (
            "fold-local-dapi-fcmae"
            if pretraining_markers == ["DAPI"]
            else "fold-local-multimarker-fcmae"
        ),
        "scope": "encoder",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "pretraining_epoch": int(checkpoint["epoch"]),
        "pretraining_val_rois": list(checkpoint["val_rois"]),
        "pretraining_markers": pretraining_markers,
        "encoder_spec": expected_specification,
        "state_tensors_loaded": len(state),
    }
