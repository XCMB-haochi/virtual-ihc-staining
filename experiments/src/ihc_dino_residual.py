from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from ihc_dinov2 import (
    DINOV2_LAYERS,
    DINOV2_WEIGHT_PATH,
    initialize_dinov2_encoder,
    load_encoder_code,
)
from ihc_unext2 import UNeXt2VirtualStainer


DINOV2_RESIDUAL_SCALE = 0.15


def dinov2_residual_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-dinov2-residual",
        "base_model": "unext2-fcmae-tiny",
        "residual_encoder": "dinov2_vits14_reg4",
        "residual_encoder_revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "residual_encoder_layers": list(DINOV2_LAYERS),
        "residual_scale": DINOV2_RESIDUAL_SCALE,
        "residual_target": "CD68",
        "residual_initialization": "zero-final-projection",
    }


class DinoV2ResidualHead(nn.Module):
    def __init__(self, residual_scale: float = DINOV2_RESIDUAL_SCALE) -> None:
        super().__init__()
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("DINOv2 residual scale must be in (0, 1].")
        self.encoder = load_encoder_code()
        self.projection = nn.Conv2d(384, 64, kernel_size=1)
        self.refinement = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )
        final = self.refinement[-1]
        if not isinstance(final, nn.Conv2d):
            raise RuntimeError("DINOv2 residual head has an invalid final projection.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_scale = float(residual_scale)
        self.initialization_report: dict[str, object] | None = None

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection.parameters()
        yield from self.refinement.parameters()

    def initialize_public_encoder(self) -> dict[str, object]:
        holder = nn.Module()
        holder.encoder = self.encoder
        self.initialization_report = initialize_dinov2_encoder(holder, DINOV2_WEIGHT_PATH)
        return self.initialization_report

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_h, pad_w = (-height) % 14, (-width) % 14
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(inputs, (left, pad_w - left, top, pad_h - top), mode="reflect")
        rgb = padded.repeat(1, 3, 1, 1)
        mean = torch.tensor((0.485, 0.456, 0.406), device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        features = self.encoder.get_intermediate_layers(rgb, n=DINOV2_LAYERS[-1:], reshape=True)[0]
        features = self.projection(features)
        features = F.interpolate(features, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.refinement(features)
        residual = self.residual_scale * torch.tanh(residual)
        return residual[..., top : top + height, left : left + width]


class UNeXt2DinoResidualVirtualStainer(UNeXt2VirtualStainer):
    """Verified UNeXt2 baseline plus a zero-start DINOv2 CD68 residual."""

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
    ) -> None:
        if out_channels < 1:
            raise ValueError("DINOv2 residual model requires positive output channels.")
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        self.dinov2_residual = DinoV2ResidualHead()
        self.dinov2_residual.initialize_public_encoder()

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().encoder_parameters()
        yield from self.dinov2_residual.encoder_parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().decoder_parameters()
        yield from self.dinov2_residual.decoder_parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = super().forward(inputs, condition)
        residual = self.dinov2_residual(inputs)
        if prediction.shape[1] == 1:
            return (prediction + residual).clamp(0.0, 1.0)
        primary = (prediction[:, :1] + residual).clamp(0.0, 1.0)
        return torch.cat((primary, prediction[:, 1:]), dim=1)
