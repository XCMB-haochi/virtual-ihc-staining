from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DecoderConvBlock(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SkipDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.block = DecoderConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.block(torch.cat((inputs, skip), dim=1))


class ConvNeXtTinyUNet(nn.Module):
    """ConvNeXt-Tiny encoder with a lightweight full-resolution decoder."""

    def __init__(self, out_channels: int = 1) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        # weights=None is deliberate: only the explicit training initializer may
        # access torchvision weights. Checkpoint inference always uses this path.
        self.encoder = convnext_tiny(weights=None).features
        self.decode_16 = SkipDecoderBlock(768, 384, 384)
        self.decode_8 = SkipDecoderBlock(384, 192, 192)
        self.decode_4 = SkipDecoderBlock(192, 96, 96)
        self.decode_2 = DecoderConvBlock(96, 64)
        self.shallow_skip = DecoderConvBlock(1, 32)
        self.decode_1 = DecoderConvBlock(64 + 32, 32)
        self.head = nn.Conv2d(32, out_channels, 1)
        nn.init.constant_(self.head.bias, -2.0)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=True,
        )

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        decoder_modules = (
            self.decode_16,
            self.decode_8,
            self.decode_4,
            self.decode_2,
            self.shallow_skip,
            self.decode_1,
            self.head,
        )
        for module in decoder_modules:
            yield from module.parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            raise ValueError("ConvNeXt-Tiny U-Net does not support conditioning.")
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                "ConvNeXt-Tiny U-Net expects input shape (batch, 1, height, width)."
            )
        if min(inputs.shape[-2:]) < 32:
            raise ValueError("ConvNeXt-Tiny U-Net requires height and width >= 32.")

        shallow = self.shallow_skip(inputs)
        current = inputs.repeat(1, 3, 1, 1)
        current = (current - self.imagenet_mean) / self.imagenet_std

        encoder_features: list[torch.Tensor] = []
        for index, layer in enumerate(self.encoder):
            current = layer(current)
            if index in (1, 3, 5, 7):
                encoder_features.append(current)
        if len(encoder_features) != 4:
            raise RuntimeError("Unexpected torchvision ConvNeXt-Tiny feature layout.")
        feature_4, feature_8, feature_16, feature_32 = encoder_features

        current = self.decode_16(feature_32, feature_16)
        current = self.decode_8(current, feature_8)
        current = self.decode_4(current, feature_4)
        half_size = tuple((dimension + 1) // 2 for dimension in inputs.shape[-2:])
        current = F.interpolate(
            current, size=half_size, mode="bilinear", align_corners=False
        )
        current = self.decode_2(current)
        current = F.interpolate(
            current, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        current = self.decode_1(torch.cat((current, shallow), dim=1))
        return torch.sigmoid(self.head(current))


def convnext_tiny_unet(out_channels: int = 1) -> ConvNeXtTinyUNet:
    return ConvNeXtTinyUNet(out_channels=out_channels)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_convnext_tiny_imagenet1k_v1() -> tuple[
    Mapping[str, torch.Tensor], dict[str, object]
]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    state_dict = weights.get_state_dict(progress=True, check_hash=True)
    cache_name = Path(urlparse(weights.url).path).name
    cache_file = Path(torch.hub.get_dir()) / "checkpoints" / cache_name
    if not cache_file.is_file():
        raise FileNotFoundError(
            "Torchvision returned ConvNeXt-Tiny weights without the expected cache "
            f"artifact: {cache_file}"
        )
    checksum = _sha256(cache_file)
    if len(checksum) != 64:
        raise RuntimeError("Failed to compute the full SHA-256 weight checksum.")
    return state_dict, {
        "version": 1,
        "mode": "torchvision-imagenet",
        "scope": "encoder",
        "weights_enum": (
            "torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
        ),
        "weights_url": weights.url,
        "cache_file": str(cache_file.resolve()),
        "cache_file_sha256": checksum,
        "torchvision_version": torchvision.__version__,
    }


def initialize_convnext_tiny_encoder_imagenet1k_v1(
    model: nn.Module,
) -> dict[str, object]:
    if not isinstance(model, ConvNeXtTinyUNet):
        raise TypeError(
            "ImageNet ConvNeXt-Tiny encoder initialization requires ConvNeXtTinyUNet."
        )
    state_dict, provenance = _load_convnext_tiny_imagenet1k_v1()
    prefix = "features."
    encoder_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    expected_keys = set(model.encoder.state_dict())
    actual_keys = set(encoder_state)
    if actual_keys != expected_keys:
        raise ValueError(
            "Torchvision ConvNeXt-Tiny encoder state does not match the model: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}."
        )
    model.encoder.load_state_dict(encoder_state, strict=True)
    return {**provenance, "state_tensors_loaded": len(encoder_state)}
