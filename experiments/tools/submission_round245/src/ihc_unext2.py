from __future__ import annotations
import hashlib
from collections.abc import Iterator,Mapping,Sequence
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from ihc_spatial import *


UNEXT2_ENCODER_BLOCKS = (3, 3, 9, 3)


UNEXT2_ENCODER_DIMS = (96, 192, 384, 768)


UNEXT2_STEM_STRIDE = 2


UNEXT2_TOTAL_STRIDE = 16


UNEXT2_DECODER_BLOCKS = 2


FCMAE_DECODER_BLOCKS = 1


FCMAE_MASK_RATIO = 0.5


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
        if condition_mode not in ('none',SPATIAL_THUMBNAIL_CONDITION_MODE) or wide_output_head or model_size!='tiny':
            raise ValueError('Only submitted Tiny/spatial-thumbnail architecture is included')
        self.condition_mode=condition_mode
        self.model_size=model_size
        self.condition_dimension=0 if condition_mode=='none' else SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        encoder_blocks,encoder_dims=unext2_encoder_shape(model_size)
        self.encoder=FCMAEEncoder(blocks=encoder_blocks,dims=encoder_dims)
        self.wide_output_head=False
        self.decoder=UNeXt2Decoder(out_channels,UNEXT2_DECODER_BLOCKS,encoder_dims=encoder_dims)
        self.condition_film=D4InvariantThumbnailFiLM(encoder_dims[-1]) if condition_mode==SPATIAL_THUMBNAIL_CONDITION_MODE else None
        self.output_head=nn.Identity()
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def unext2_encoder_shape(model_size='tiny'):
    if model_size!='tiny':raise ValueError('Only the submitted Tiny architecture is included')
    return UNEXT2_ENCODER_BLOCKS,UNEXT2_ENCODER_DIMS
