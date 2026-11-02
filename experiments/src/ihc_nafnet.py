"""NAFNet adapted for single-channel virtual IHC staining.

The architecture is adapted from Megvii's official NAFNet implementation:
https://github.com/megvii-research/NAFNet

The upstream project is Copyright (c) 2022 Megvii and is distributed under
the MIT License. This implementation is an independent, dependency-free
PyTorch adaptation of that architecture. Project-specific changes are:

* one input channel and a configurable number of output channels;
* removal of NAFNet's global image-restoration input residual; and
* sigmoid-bounded output with the same `-2` head bias as the U-Net baseline,
  followed by cropping to the original image size.

The default block counts reproduce the official NAFNet-width32 configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization used by the official NAFNet blocks."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight[None, :, None, None]
            + self.bias[None, :, None, None]
        )


class SimpleGate(nn.Module):
    """Replace an activation with element-wise multiplication of channel halves."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Nonlinear-activation-free restoration block from NAFNet."""

    def __init__(
        self,
        channels: int,
        depthwise_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if depthwise_expand <= 0 or ffn_expand <= 0:
            raise ValueError("NAFBlock expansion factors must be positive.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        depthwise_channels = channels * depthwise_expand
        ffn_channels = channels * ffn_expand
        if depthwise_channels % 2 or ffn_channels % 2:
            raise ValueError("Expanded NAFBlock channel counts must be even.")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.conv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.gate = SimpleGate()
        gated_channels = depthwise_channels // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1),
        )
        self.conv3 = nn.Conv2d(gated_channels, channels, 1)
        self.dropout1 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = (
            nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        )

        # Zero initialization makes each block an identity at initialization,
        # matching the official architecture's stable residual scaling.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = self.conv1(self.norm1(inputs))
        current = self.conv2(current)
        current = self.gate(current)
        current = current * self.channel_attention(current)
        current = self.dropout1(self.conv3(current))
        first_residual = inputs + current * self.beta

        current = self.conv4(self.norm2(first_residual))
        current = self.gate(current)
        current = self.dropout2(self.conv5(current))
        return first_residual + current * self.gamma


class NAFNet(nn.Module):
    """NAFNet-width32 adapted to predict virtual-staining marker channels."""

    def __init__(
        self,
        out_channels: int = 1,
        width: int = 32,
        enc_blk_nums: Sequence[int] = (1, 1, 1, 28),
        middle_blk_num: int = 1,
        dec_blk_nums: Sequence[int] = (1, 1, 1, 1),
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if width <= 0:
            raise ValueError("width must be positive.")
        if len(enc_blk_nums) == 0 or len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError(
                "Encoder and decoder block counts must have the same nonzero length."
            )
        if middle_blk_num < 0 or any(number < 0 for number in enc_blk_nums):
            raise ValueError("NAFNet block counts must be non-negative.")
        if any(number < 0 for number in dec_blk_nums):
            raise ValueError("NAFNet block counts must be non-negative.")

        self.out_channels = out_channels
        self.width = width
        self.enc_blk_nums = tuple(enc_blk_nums)
        self.middle_blk_num = middle_blk_num
        self.dec_blk_nums = tuple(dec_blk_nums)
        self.padder_size = 2 ** len(self.enc_blk_nums)

        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)
        nn.init.constant_(self.ending.bias, -2.0)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for block_count in self.enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(channels, dropout_rate=dropout_rate)
                        for _ in range(block_count)
                    ]
                )
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(channels, dropout_rate=dropout_rate)
                for _ in range(middle_blk_num)
            ]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in self.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(channels, dropout_rate=dropout_rate)
                        for _ in range(block_count)
                    ]
                )
            )

    def _pad_to_supported_size(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(inputs, (0, pad_width, 0, pad_height))

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            raise ValueError("NAFNet does not support conditioning tensors.")
        if inputs.ndim != 4:
            raise ValueError(
                f"NAFNet inputs must have shape (batch, 1, height, width), got "
                f"{tuple(inputs.shape)}."
            )
        if inputs.shape[1] != 1:
            raise ValueError(
                f"NAFNet requires one input channel, got {inputs.shape[1]}."
            )
        if inputs.shape[-2] <= 0 or inputs.shape[-1] <= 0:
            raise ValueError("NAFNet input height and width must be positive.")

        original_height, original_width = inputs.shape[-2:]
        current = self.intro(self._pad_to_supported_size(inputs))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            current = encoder(current)
            skips.append(current)
            current = down(current)

        current = self.middle_blks(current)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            current = decoder(up(current) + skip)

        # The official restoration residual ``current + inputs`` is deliberately
        # omitted because virtual IHC output is not the input imaging modality.
        output = torch.sigmoid(self.ending(current))
        return output[..., :original_height, :original_width]


def nafnet_width32(out_channels: int = 1) -> NAFNet:
    """Construct the fixed official NAFNet-width32 block configuration."""

    return NAFNet(
        out_channels=out_channels,
        width=32,
        enc_blk_nums=(1, 1, 1, 28),
        middle_blk_num=1,
        dec_blk_nums=(1, 1, 1, 1),
    )
