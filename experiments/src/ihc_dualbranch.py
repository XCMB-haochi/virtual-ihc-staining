from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock


LOCAL_BASE_CHANNELS = 32
SWIN_DIMS = (48, 96, 192)
SWIN_DEPTHS = (2, 2, 4)
SWIN_HEADS = (3, 6, 12)
SWIN_WINDOW_SIZE = 8


class DualBranchConvBlock(nn.Module):
    """The established residual U-Net block, kept local to avoid import cycles."""

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


class SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=heads,
                    window_size=[SWIN_WINDOW_SIZE, SWIN_WINDOW_SIZE],
                    shift_size=(
                        [0, 0]
                        if index % 2 == 0
                        else [SWIN_WINDOW_SIZE // 2, SWIN_WINDOW_SIZE // 2]
                    ),
                    mlp_ratio=4.0,
                    dropout=0.0,
                    attention_dropout=0.0,
                    stochastic_depth_prob=0.0,
                )
                for index in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs
        for block in self.blocks:
            current = block(current)
        return current


class GatedFusion(nn.Module):
    def __init__(self, global_channels: int, local_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(
            global_channels, local_channels, kernel_size=1, bias=False
        )
        self.normalize = nn.GroupNorm(min(8, local_channels), local_channels)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.normalize(
            self.project(global_features.permute(0, 3, 1, 2).contiguous())
        )
        if projected.shape[-2:] != local_features.shape[-2:]:
            projected = F.interpolate(
                projected,
                size=local_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return local_features + self.gate * projected


class DualBranchSwinSmall(nn.Module):
    """Base32 local U-Net augmented with a compact multi-scale Swin branch."""

    def __init__(self, out_channels: int = 1) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        # Keep the complete local path first and in the same construction order as
        # the established base32 U-Net. With a reset RNG seed, its initial tensors
        # therefore match the control before the global branch consumes RNG state.
        channels = [LOCAL_BASE_CHANNELS * (2**index) for index in range(5)]
        self.encoders = nn.ModuleList(
            [
                DualBranchConvBlock(1, channels[0]),
                DualBranchConvBlock(channels[0], channels[1]),
                DualBranchConvBlock(channels[1], channels[2]),
                DualBranchConvBlock(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DualBranchConvBlock(channels[3], channels[4])
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
                DualBranchConvBlock(channels[3] * 2, channels[3]),
                DualBranchConvBlock(channels[2] * 2, channels[2]),
                DualBranchConvBlock(channels[1] * 2, channels[1]),
                DualBranchConvBlock(channels[0] * 2, channels[0]),
            ]
        )
        self.head = nn.Conv2d(channels[0], out_channels, 1)
        nn.init.constant_(self.head.bias, -2.0)

        self.patch_embed = nn.Conv2d(
            1,
            SWIN_DIMS[0],
            kernel_size=4,
            stride=4,
        )
        self.swin_stages = nn.ModuleList(
            [
                SwinStage(dim, depth, heads)
                for dim, depth, heads in zip(SWIN_DIMS, SWIN_DEPTHS, SWIN_HEADS)
            ]
        )
        self.patch_merges = nn.ModuleList(
            [PatchMerging(SWIN_DIMS[0]), PatchMerging(SWIN_DIMS[1])]
        )
        self.fusions = nn.ModuleList(
            [
                GatedFusion(SWIN_DIMS[0], channels[2]),
                GatedFusion(SWIN_DIMS[1], channels[3]),
                GatedFusion(SWIN_DIMS[2], channels[4]),
            ]
        )

    def _global_features(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        current = self.patch_embed(inputs).permute(0, 2, 3, 1).contiguous()
        features: list[torch.Tensor] = []
        for index, stage in enumerate(self.swin_stages):
            current = stage(current)
            features.append(current)
            if index < len(self.patch_merges):
                current = self.patch_merges[index](current)
        return features

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            raise ValueError("DualBranchSwinSmall does not support conditioning.")
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                "DualBranchSwinSmall expects input shape (batch, 1, height, width)."
            )
        if min(inputs.shape[-2:]) < 16:
            raise ValueError("DualBranchSwinSmall requires height and width >= 16.")

        global_features = self._global_features(inputs)
        skips: list[torch.Tensor] = []
        current = inputs
        for index, encoder in enumerate(self.encoders):
            current = encoder(current)
            if index >= 2:
                current = self.fusions[index - 2](
                    current, global_features[index - 2]
                )
            skips.append(current)
            current = self.pool(current)
        current = self.bottleneck(current)
        current = self.fusions[2](current, global_features[2])

        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            current = upconv(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = F.interpolate(
                    current,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            current = decoder(torch.cat((current, skip), dim=1))
        return torch.sigmoid(self.head(current))


def dualbranch_swin_small(out_channels: int = 1) -> DualBranchSwinSmall:
    return DualBranchSwinSmall(out_channels=out_channels)
