from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from ihc_dinov2 import (
    DINOV2_LAYERS,
    DINOV2_REVISION,
    DINOV2_WEIGHT_PATH,
    initialize_dinov2_encoder,
    load_encoder_code,
)
from ihc_unext2 import UNeXt2VirtualStainer


def dinov2_fusion_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "unext2-fcmae-tiny-dinov2-fusion",
        "base_model": "unext2-fcmae-tiny",
        "fusion_encoder": "dinov2_vits14_reg4",
        "fusion_encoder_revision": DINOV2_REVISION,
        "fusion_encoder_layers": [DINOV2_LAYERS[-1]],
        "fusion_channels": 768,
        "alignment": "symmetric-reflect-multiple14-grid-sample-to-unext2-bottleneck",
        "fusion": "zero-initialized-1x1-addition",
        "fusion_target": "unext2-bottleneck",
    }


def _align_dino_to_bottleneck(
    features: torch.Tensor,
    original_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> torch.Tensor:
    """Map DINO patch centers in padded image coordinates to UNeXt2 cells."""
    height, width = original_shape
    target_height, target_width = target_shape
    pad_height, pad_width = (-height) % 14, (-width) % 14
    top, left = pad_height // 2, pad_width // 2
    padded_height, padded_width = height + pad_height, width + pad_width
    theta = features.new_zeros((features.shape[0], 2, 3))
    theta[:, 0, 0] = width / padded_width
    theta[:, 1, 1] = height / padded_height
    theta[:, 0, 2] = (2 * left + width) / padded_width - 1
    theta[:, 1, 2] = (2 * top + height) / padded_height - 1
    grid = F.affine_grid(
        theta,
        (features.shape[0], features.shape[1], target_height, target_width),
        align_corners=False,
    )
    return F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


class DinoV2BottleneckFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = load_encoder_code()
        self.projection = nn.Conv2d(384, 768, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.initialization_report: dict[str, object] | None = None

    def dino_encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def fusion_parameters(self) -> Iterator[nn.Parameter]:
        return self.projection.parameters()

    def initialize_public_encoder(self) -> dict[str, object]:
        holder = nn.Module()
        holder.encoder = self.encoder
        self.initialization_report = initialize_dinov2_encoder(holder, DINOV2_WEIGHT_PATH)
        return self.initialization_report

    def forward(self, inputs: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_h, pad_w = (-height) % 14, (-width) % 14
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(inputs, (left, pad_w - left, top, pad_h - top), mode="reflect")
        rgb = padded.repeat(1, 3, 1, 1)
        mean = inputs.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = inputs.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        dino = self.encoder.get_intermediate_layers(
            rgb, n=DINOV2_LAYERS[-1:], reshape=True
        )[0]
        dino = _align_dino_to_bottleneck(dino, (height, width), target_shape)
        return self.projection(dino)


class UNeXt2DinoBottleneckFusionVirtualStainer(UNeXt2VirtualStainer):
    """UNeXt2-FCMAE with a zero-start DINOv2 bottleneck feature branch."""

    def __init__(
        self,
        out_channels: int = 1,
        condition_mode: str = "none",
    ) -> None:
        if out_channels < 1:
            raise ValueError("DINOv2 fusion model requires positive output channels.")
        super().__init__(out_channels=out_channels, condition_mode=condition_mode)
        self.dinov2_fusion = DinoV2BottleneckFusion()

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().encoder_parameters()
        yield from self.dinov2_fusion.dino_encoder_parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from super().decoder_parameters()
        yield from self.dinov2_fusion.fusion_parameters()

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
        features[-1] = features[-1] + self.dinov2_fusion(inputs, features[-1].shape[-2:])
        logits = self.output_head(self.decoder(features))
        return torch.sigmoid(logits + self.output_bias[None, :, None, None])
