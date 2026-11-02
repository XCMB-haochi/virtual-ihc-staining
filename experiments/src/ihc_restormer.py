"""Restormer adapted for single-channel virtual IHC staining.

The architecture is adapted from the official Restormer implementation:
https://github.com/swz30/Restormer

The upstream project is Copyright (c) 2022 Syed Waqas Zamir and contributors
and is distributed under the MIT License. This implementation is an
independent, dependency-free PyTorch adaptation. Project-specific changes are:

* one input channel and a configurable number of output channels;
* a smaller fixed configuration for paired-data screening;
* removal of the input-to-output restoration residual because DAPI and IHC are
  different imaging modalities; and
* sigmoid-bounded output with the same `-2` head bias as the U-Net baseline.

The attention, gated depth-wise feed-forward network, hierarchical encoder,
decoder, and refinement layout follow the official Restormer design.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


RESTORMER_SMALL_DIM = 32
RESTORMER_SMALL_BLOCKS = (2, 3, 3, 4)
RESTORMER_SMALL_REFINEMENT_BLOCKS = 2
RESTORMER_SMALL_HEADS = (1, 2, 4, 8)
RESTORMER_SMALL_FFN_EXPANSION = 2.66


class BiasFreeLayerNorm(nn.Module):
    """Per-pixel channel normalization used by Restormer's bias-free blocks."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("LayerNorm channels must be positive.")
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.var(dim=1, keepdim=True, unbiased=False)
        return (
            inputs * torch.rsqrt(variance + self.eps) * self.weight[None, :, None, None]
        )


class GatedDepthwiseFeedForward(nn.Module):
    """Restormer gated depth-wise convolutional feed-forward network."""

    def __init__(self, channels: int, expansion: float, bias: bool = False) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        if hidden_channels <= 0:
            raise ValueError("Feed-forward expansion must produce positive channels.")
        self.project_in = nn.Conv2d(
            channels,
            hidden_channels * 2,
            1,
            bias=bias,
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, 1, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = self.depthwise(self.project_in(inputs)).chunk(2, dim=1)
        return self.project_out(F.gelu(first) * second)


class TransposedAttention(nn.Module):
    """Multi-DConv-head transposed self-attention (MDTA)."""

    def __init__(self, channels: int, heads: int, bias: bool = False) -> None:
        super().__init__()
        if heads <= 0 or channels % heads:
            raise ValueError("Attention channels must be divisible by positive heads.")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        query, key, value = self.qkv_depthwise(self.qkv(inputs)).chunk(3, dim=1)
        head_channels = channels // self.heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch,
                self.heads,
                head_channels,
                height * width,
            )

        query = F.normalize(split_heads(query), dim=-1)
        key = F.normalize(split_heads(key), dim=-1)
        value = split_heads(value)
        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = (attention @ value).reshape(batch, channels, height, width)
        return self.project_out(output)


class RestormerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        ffn_expansion: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = BiasFreeLayerNorm(channels)
        self.attention = TransposedAttention(channels, heads, bias=bias)
        self.norm2 = BiasFreeLayerNorm(channels)
        self.feed_forward = GatedDepthwiseFeedForward(
            channels,
            ffn_expansion,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs + self.attention(self.norm1(inputs))
        return current + self.feed_forward(self.norm2(current))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("Downsample channels must be positive and even.")
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("Upsample channels must be positive and even.")
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.body(inputs)


class Restormer(nn.Module):
    """Hierarchical restoration Transformer for virtual-staining regression."""

    def __init__(
        self,
        out_channels: int = 1,
        dim: int = RESTORMER_SMALL_DIM,
        num_blocks: Sequence[int] = RESTORMER_SMALL_BLOCKS,
        refinement_blocks: int = RESTORMER_SMALL_REFINEMENT_BLOCKS,
        heads: Sequence[int] = RESTORMER_SMALL_HEADS,
        ffn_expansion: float = RESTORMER_SMALL_FFN_EXPANSION,
    ) -> None:
        super().__init__()
        if out_channels <= 0 or dim <= 0 or dim % 2:
            raise ValueError(
                "Restormer output channels and even base dimension must be positive."
            )
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("Restormer requires four block and attention-head levels.")
        if any(count <= 0 for count in num_blocks) or refinement_blocks <= 0:
            raise ValueError("Restormer block counts must be positive.")
        if ffn_expansion <= 0:
            raise ValueError("Restormer feed-forward expansion must be positive.")
        for level, head_count in enumerate(heads):
            if head_count <= 0 or (dim * (2**level)) % head_count:
                raise ValueError("Restormer level channels must be divisible by heads.")

        self.out_channels = out_channels
        self.dim = dim
        self.num_blocks = tuple(int(count) for count in num_blocks)
        self.refinement_blocks = refinement_blocks
        self.heads = tuple(int(count) for count in heads)
        self.ffn_expansion = float(ffn_expansion)
        self.condition_mode = "none"
        self.padder_size = 8

        def blocks(level: int, count: int) -> nn.Sequential:
            channels = dim * (2**level)
            return nn.Sequential(
                *[
                    RestormerBlock(
                        channels,
                        self.heads[level],
                        self.ffn_expansion,
                    )
                    for _ in range(count)
                ]
            )

        self.patch_embed = nn.Conv2d(1, dim, 3, padding=1, bias=False)
        self.encoder_level1 = blocks(0, self.num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = blocks(1, self.num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = blocks(2, self.num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = blocks(3, self.num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = blocks(2, self.num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = blocks(1, self.num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(
            *[
                RestormerBlock(dim * 2, self.heads[0], self.ffn_expansion)
                for _ in range(self.num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                RestormerBlock(dim * 2, self.heads[0], self.ffn_expansion)
                for _ in range(refinement_blocks)
            ]
        )
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1)
        nn.init.constant_(self.output.bias, -2.0)

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
            raise ValueError("Restormer does not support conditioning tensors.")
        if inputs.ndim != 4:
            raise ValueError(
                "Restormer inputs must have shape (batch, 1, height, width), got "
                f"{tuple(inputs.shape)}."
            )
        if inputs.shape[1] != 1:
            raise ValueError(
                f"Restormer requires one input channel, got {inputs.shape[1]}."
            )
        if inputs.shape[-2] <= 0 or inputs.shape[-1] <= 0:
            raise ValueError("Restormer input height and width must be positive.")

        original_height, original_width = inputs.shape[-2:]
        current = self.patch_embed(self._pad_to_supported_size(inputs))
        level1 = self.encoder_level1(current)
        level2 = self.encoder_level2(self.down1_2(level1))
        level3 = self.encoder_level3(self.down2_3(level2))
        latent = self.latent(self.down3_4(level3))

        current = self.up4_3(latent)
        current = self.reduce_level3(torch.cat((current, level3), dim=1))
        current = self.decoder_level3(current)
        current = self.up3_2(current)
        current = self.reduce_level2(torch.cat((current, level2), dim=1))
        current = self.decoder_level2(current)
        current = self.up2_1(current)
        current = self.decoder_level1(torch.cat((current, level1), dim=1))
        current = self.refinement(current)

        # The official restoration residual is omitted because the target is a
        # different staining modality rather than a corrected DAPI image.
        output = torch.sigmoid(self.output(current))
        return output[..., :original_height, :original_width]


def restormer_small(out_channels: int = 1) -> Restormer:
    return Restormer(out_channels=out_channels)
