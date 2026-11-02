from __future__ import annotations
from torch import nn
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from pathlib import Path
import hashlib
from collections.abc import Mapping
UNEXT2_ENCODER_BLOCKS = (3, 3, 9, 3)
UNEXT2_ENCODER_DIMS = (96, 192, 384, 768)
UNEXT2_BASE_ENCODER_BLOCKS = (3, 3, 27, 3)
UNEXT2_BASE_ENCODER_DIMS = (128, 256, 512, 1024)
UNEXT2_STEM_STRIDE = 2
UNEXT2_TOTAL_STRIDE = 16
UNEXT2_DECODER_BLOCKS = 2
UNEXT2_WIDE_HEAD_FEATURES = 48
FCMAE_DECODER_BLOCKS = 1
FCMAE_MASK_RATIO = 0.5
OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE = 'official-convnextv2-fcmae-imagenet1k-tiny'
OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256 = 'b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef'
OFFICIAL_CONVNEXTV2_FCMAE_TINY_SOURCE = 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt'
OFFICIAL_CONVNEXTV2_FCMAE_LICENSE = 'CC-BY-NC-4.0'
OFFICIAL_CONVNEXTV2_INITIALIZATION_SPECS = {OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE: {'sha256': OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256, 'source_url': OFFICIAL_CONVNEXTV2_FCMAE_TINY_SOURCE, 'license': OFFICIAL_CONVNEXTV2_FCMAE_LICENSE, 'source_training': 'self-supervised-fcmae-imagenet1k', 'model_size': 'tiny', 'additional_skipped_source_tensors': ()}}

def unext2_encoder_shape(model_size: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shapes = {'tiny': (UNEXT2_ENCODER_BLOCKS, UNEXT2_ENCODER_DIMS), 'base': (UNEXT2_BASE_ENCODER_BLOCKS, UNEXT2_BASE_ENCODER_DIMS)}
    try:
        return shapes[model_size]
    except KeyError as error:
        raise ValueError(f'Unsupported UNeXt2 model size: {model_size!r}.') from error

def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class LayerNorm2d(nn.Module):

    def __init__(self, channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs.permute(0, 2, 3, 1)
        current = F.layer_norm(current, (self.weight.numel(),), self.weight, self.bias, self.eps)
        return current.permute(0, 3, 1, 2)

class GlobalResponseNorm(nn.Module):
    """Channel-last ConvNeXt V2 global response normalization."""

    def __init__(self, channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(inputs, ord=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return inputs + self.gamma * (inputs * normalized) + self.beta

def _resize_boolean_mask(mask: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    target_height, target_width = (int(value) for value in spatial_shape)
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.dtype != torch.bool:
        raise ValueError('Mask must be a boolean tensor with shape (batch, 1, H, W).')
    height, width = mask.shape[-2:]
    if target_height % height or target_width % width:
        raise ValueError(f'Target feature shape must be divisible by the FCMAE mask shape: {(target_height, target_width)} vs {(height, width)}.')
    if (height, width) == (target_height, target_width):
        return mask
    return mask.repeat_interleave(target_height // height, dim=-2).repeat_interleave(target_width // width, dim=-1)

def generate_fcmae_mask(inputs: torch.Tensor, mask_ratio: float=FCMAE_MASK_RATIO, total_stride: int=UNEXT2_TOTAL_STRIDE) -> torch.Tensor:
    if inputs.ndim != 4:
        raise ValueError('FCMAE inputs must have shape (batch, channels, H, W).')
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError('FCMAE mask_ratio must be strictly between 0 and 1.')
    height, width = inputs.shape[-2:]
    if height % total_stride or width % total_stride:
        raise ValueError(f'FCMAE input shape {(height, width)} must be divisible by {total_stride}.')
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

    def __init__(self, channels: int, drop_path_rate: float=0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError('ConvNeXt-V2 drop path rate must be in [0, 1).')
        self.drop_path_rate = float(drop_path_rate)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-06)
        self.expand = nn.Linear(channels, channels * 4)
        self.grn = GlobalResponseNorm(channels * 4)
        self.project = nn.Linear(channels * 4, channels)
        self.apply(_initialize_module)

    def forward(self, inputs: torch.Tensor, unmasked: torch.Tensor | None=None) -> torch.Tensor:
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
            keep = torch.rand((current.shape[0], 1, 1, 1), device=current.device, dtype=current.dtype) < keep_probability
            current = current * keep / keep_probability
        return shortcut + current

class ConvNeXtV2Stage(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> None:
        super().__init__()
        if stride == 1 and in_channels == out_channels:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(LayerNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride))
        self.blocks = nn.ModuleList((ConvNeXtV2Block(out_channels) for _ in range(blocks)))
        self.downsample.apply(_initialize_module)

    def forward(self, inputs: torch.Tensor, unmasked: torch.Tensor | None=None) -> torch.Tensor:
        if unmasked is not None:
            inputs = inputs * _resize_boolean_mask(unmasked, inputs.shape[-2:])
        current = self.downsample(inputs)
        if unmasked is not None:
            current = current * _resize_boolean_mask(unmasked, current.shape[-2:])
        for block in self.blocks:
            current = block(current, unmasked)
        return current

class FCMAEEncoder(nn.Module):

    def __init__(self, in_channels: int=1, blocks: Sequence[int]=UNEXT2_ENCODER_BLOCKS, dims: Sequence[int]=UNEXT2_ENCODER_DIMS) -> None:
        super().__init__()
        if len(blocks) != 4 or len(dims) != 4:
            raise ValueError('UNeXt2 encoder requires exactly four stages.')
        self.block_counts = tuple((int(value) for value in blocks))
        self.dimensions = tuple((int(value) for value in dims))
        self.stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=UNEXT2_STEM_STRIDE, stride=UNEXT2_STEM_STRIDE), LayerNorm2d(dims[0]))
        self.stages = nn.ModuleList((ConvNeXtV2Stage(self.dimensions[index - 1] if index else self.dimensions[0], self.dimensions[index], self.block_counts[index], stride=1 if index == 0 else 2) for index in range(4)))
        self.stem.apply(_initialize_module)

    def configure_drop_path(self, maximum_rate: float) -> list[float]:
        if not 0.0 <= maximum_rate < 1.0:
            raise ValueError('Encoder drop path maximum must be in [0, 1).')
        encoder_blocks = [block for stage in self.stages for block in stage.blocks]
        rates = torch.linspace(0.0, float(maximum_rate), len(encoder_blocks)).tolist()
        for block, rate in zip(encoder_blocks, rates):
            block.drop_path_rate = float(rate)
        return [float(rate) for rate in rates]

    def forward(self, inputs: torch.Tensor, mask_ratio: float=0.0) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError('UNeXt2 expects inputs with shape (batch, 1, H, W).')
        if min(inputs.shape[-2:]) < UNEXT2_TOTAL_STRIDE:
            raise ValueError('UNeXt2 input height and width must be at least 16.')
        if inputs.shape[-2] % UNEXT2_TOTAL_STRIDE or inputs.shape[-1] % UNEXT2_TOTAL_STRIDE:
            raise ValueError('UNeXt2 input height and width must be divisible by 16.')
        mask = generate_fcmae_mask(inputs, mask_ratio) if mask_ratio > 0.0 else None
        unmasked = None if mask is None else ~mask
        current = self.stem(inputs)
        if unmasked is not None:
            current = current * _resize_boolean_mask(unmasked, current.shape[-2:])
        features: list[torch.Tensor] = []
        for stage in self.stages:
            current = stage(current, unmasked)
            features.append(current)
        full_mask = None if mask is None else _resize_boolean_mask(mask, inputs.shape[-2:])
        return (features, full_mask)

def _icnr_initialize_linear(linear: nn.Linear, scale: int=2) -> None:
    repeats = scale ** 2
    if linear.out_features % repeats:
        raise ValueError('ICNR output channels must be divisible by scale squared.')
    base = torch.empty(linear.out_features // repeats, linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype)
    nn.init.trunc_normal_(base, std=0.02)
    with torch.no_grad():
        linear.weight.copy_(base.repeat_interleave(repeats, dim=0))
        if linear.bias is not None:
            linear.bias.zero_()

class PixelShuffleUpStage(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int) -> None:
        super().__init__()
        if in_channels % 4:
            raise ValueError('Pixel-shuffle input channels must be divisible by four.')
        up_channels = in_channels // 4
        self.upsample = nn.PixelShuffle(2)
        self.projection = nn.Sequential(LayerNorm2d(up_channels + skip_channels), nn.Conv2d(up_channels + skip_channels, out_channels, kernel_size=1))
        self.blocks = nn.Sequential(*(ConvNeXtV2Block(out_channels) for _ in range(blocks)))
        self.projection.apply(_initialize_module)
        _icnr_initialize_linear(self.blocks[-1].project)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        current = self.upsample(inputs)
        if current.shape[-2:] != skip.shape[-2:]:
            raise ValueError(f'UNeXt2 decoder and skip shapes disagree: {tuple(current.shape[-2:])} vs {tuple(skip.shape[-2:])}.')
        current = self.projection(torch.cat((current, skip), dim=1))
        return self.blocks(current)

class UNeXt2Decoder(nn.Module):

    def __init__(self, out_channels: int, blocks_per_stage: int, encoder_dims: Sequence[int]=UNEXT2_ENCODER_DIMS) -> None:
        super().__init__()
        if len(encoder_dims) != 4:
            raise ValueError('UNeXt2 decoder requires four encoder dimensions.')
        channels = (encoder_dims[-1], encoder_dims[-2], encoder_dims[-3], out_channels * 4)
        skip_channels = (encoder_dims[-2], encoder_dims[-3], encoder_dims[-4])
        self.stages = nn.ModuleList((PixelShuffleUpStage(channels[index], skip_channels[index], channels[index + 1], blocks_per_stage) for index in range(3)))
        self.output_shuffle = nn.PixelShuffle(2)

    def forward_half(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError('UNeXt2 decoder requires four encoder feature maps.')
        current = features[-1]
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            current = stage(current, skip)
        return current

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.output_shuffle(self.forward_half(features))

class UNeXt2FeatureDecoder(nn.Module):
    """Decode to a fixed half-resolution feature map for a separate head."""

    def __init__(self, feature_channels: int=UNEXT2_WIDE_HEAD_FEATURES, blocks_per_stage: int=UNEXT2_DECODER_BLOCKS) -> None:
        super().__init__()
        if feature_channels <= 0 or feature_channels % 4:
            raise ValueError('UNeXt2 feature decoder channels must be positive and divisible by four.')
        channels = (UNEXT2_ENCODER_DIMS[-1], UNEXT2_ENCODER_DIMS[-2], UNEXT2_ENCODER_DIMS[-3], feature_channels)
        skip_channels = (UNEXT2_ENCODER_DIMS[-2], UNEXT2_ENCODER_DIMS[-3], UNEXT2_ENCODER_DIMS[-4])
        self.stages = nn.ModuleList((PixelShuffleUpStage(channels[index], skip_channels[index], channels[index + 1], blocks_per_stage) for index in range(3)))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError('UNeXt2 decoder requires four encoder feature maps.')
        current = features[-1]
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            current = stage(current, skip)
        return current

class DapiFCMAE(nn.Module):

    def __init__(self, model_size: str='tiny') -> None:
        super().__init__()
        self.model_size = model_size
        encoder_blocks, encoder_dims = unext2_encoder_shape(model_size)
        self.encoder = FCMAEEncoder(blocks=encoder_blocks, dims=encoder_dims)
        self.decoder = UNeXt2Decoder(1, FCMAE_DECODER_BLOCKS, encoder_dims=encoder_dims)
        self.output_bias = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, inputs: torch.Tensor, mask_ratio: float=FCMAE_MASK_RATIO) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = self.encoder(inputs, mask_ratio=mask_ratio)
        if mask is None:
            raise RuntimeError('DAPI FCMAE requires a positive mask ratio.')
        logits = self.decoder(features)
        reconstruction = torch.sigmoid(logits + self.output_bias[None, :, None, None])
        return (reconstruction, mask)

def unext2_encoder_specification(model_size: str='tiny') -> dict[str, object]:
    encoder_blocks, encoder_dims = unext2_encoder_shape(model_size)
    return {'version': 1, 'architecture': f'convnext-v2-{model_size}-dense-fcmae', 'input_channels': 1, 'stem_kernel': 2, 'stem_stride': 2, 'stage_blocks': list(encoder_blocks), 'stage_dims': list(encoder_dims), 'total_stride': UNEXT2_TOTAL_STRIDE, 'block': 'convnext-v2-grn'}

def unext2_encoder_specification_from_encoder(encoder: FCMAEEncoder) -> dict[str, object]:
    for model_size in ('tiny', 'base'):
        blocks, dims = unext2_encoder_shape(model_size)
        if encoder.block_counts == blocks and encoder.dimensions == dims:
            return unext2_encoder_specification(model_size)
    raise ValueError('FCMAE encoder shape is not a supported UNeXt2 size.')

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def official_convnextv2_fcmae_state_mapping(model_size: str) -> dict[str, str]:
    """Map official ConvNeXt-V2 FCMAE encoder keys to a local encoder size."""
    encoder_blocks, _ = unext2_encoder_shape(model_size)
    mapping = {'downsample_layers.0.1.weight': 'stem.1.weight', 'downsample_layers.0.1.bias': 'stem.1.bias'}
    for stage in range(1, 4):
        for index in range(2):
            for parameter in ('weight', 'bias'):
                mapping[f'downsample_layers.{stage}.{index}.{parameter}'] = f'stages.{stage}.downsample.{index}.{parameter}'
    block_fields = {'dwconv.weight': 'depthwise.weight', 'dwconv.bias': 'depthwise.bias', 'norm.weight': 'norm.weight', 'norm.bias': 'norm.bias', 'pwconv1.weight': 'expand.weight', 'pwconv1.bias': 'expand.bias', 'grn.gamma': 'grn.gamma', 'grn.beta': 'grn.beta', 'pwconv2.weight': 'project.weight', 'pwconv2.bias': 'project.bias'}
    for stage, block_count in enumerate(encoder_blocks):
        for block in range(block_count):
            for source_field, target_field in block_fields.items():
                mapping[f'stages.{stage}.{block}.{source_field}'] = f'stages.{stage}.blocks.{block}.{target_field}'
    return mapping

def initialize_fcmae_encoder_from_official_convnextv2(encoder: FCMAEEncoder, checkpoint_path: Path, expected_sha256: str=OFFICIAL_CONVNEXTV2_FCMAE_TINY_SHA256, mode: str=OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE) -> dict[str, object]:
    """Initialize compatible encoder tensors from an audited official model."""
    if not isinstance(encoder, FCMAEEncoder):
        raise TypeError('Official ConvNeXt-V2 initialization requires FCMAEEncoder.')
    source_spec = OFFICIAL_CONVNEXTV2_INITIALIZATION_SPECS.get(mode)
    if source_spec is None:
        raise ValueError(f'Unsupported official ConvNeXt-V2 mode: {mode!r}.')
    checkpoint_path = checkpoint_path.resolve()
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any((character not in '0123456789abcdef' for character in expected_sha256)):
        raise ValueError('Expected official checkpoint SHA-256 is invalid.')
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != expected_sha256:
        raise ValueError(f'Official ConvNeXt-V2 checkpoint SHA-256 mismatch: expected {expected_sha256}, got {checkpoint_sha256}.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {'model'}:
        raise ValueError('Unsupported official ConvNeXt-V2 FCMAE checkpoint format.')
    source_state = checkpoint['model']
    if not isinstance(source_state, Mapping):
        raise ValueError('Official ConvNeXt-V2 checkpoint has no model state.')
    model_size = str(source_spec['model_size'])
    expected_encoder_spec = unext2_encoder_specification(model_size)
    if unext2_encoder_specification_from_encoder(encoder) != expected_encoder_spec:
        raise ValueError(f'Official ConvNeXt-V2 {model_size} initialization requires a matching {model_size} encoder.')
    state_mapping = official_convnextv2_fcmae_state_mapping(model_size)
    skipped_source = {'downsample_layers.0.0.weight', 'downsample_layers.0.0.bias'} | set(source_spec['additional_skipped_source_tensors'])
    if set(source_state) != set(state_mapping) | skipped_source:
        missing = sorted((set(state_mapping) | skipped_source) - set(source_state))
        unexpected = sorted(set(source_state) - set(state_mapping) - skipped_source)
        raise ValueError(f'Official ConvNeXt-V2 state keys do not match {model_size} FCMAE: missing={missing}, unexpected={unexpected}.')
    target_state = encoder.state_dict()
    preserved_target = {'stem.0.weight', 'stem.0.bias'}
    if set(target_state) != set(state_mapping.values()) | preserved_target:
        raise RuntimeError('Project FCMAE encoder state no longer matches the mapping.')
    updated_state = dict(target_state)
    reshaped: list[str] = []
    loaded_parameters = 0
    for source_key, target_key in state_mapping.items():
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f'Official state value is not a tensor: {source_key}.')
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(f'Official tensor dtype mismatch for {source_key}: {source_tensor.dtype} vs {target_tensor.dtype}.')
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            loaded = source_tensor
        elif source_key.endswith(('grn.gamma', 'grn.beta')) and source_tensor.numel() == target_tensor.numel():
            loaded = source_tensor.reshape_as(target_tensor)
            reshaped.append(source_key)
        else:
            raise ValueError(f'Official tensor shape mismatch for {source_key} -> {target_key}: {tuple(source_tensor.shape)} vs {tuple(target_tensor.shape)}.')
        updated_state[target_key] = loaded
        loaded_parameters += target_tensor.numel()
    encoder.load_state_dict(updated_state, strict=True)
    encoder_parameters = sum((tensor.numel() for tensor in target_state.values()))
    return {'version': 1, 'mode': mode, 'scope': 'encoder-except-input-convolution', 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha256, 'source_url': source_spec['source_url'], 'license': source_spec['license'], 'source_training': source_spec['source_training'], 'source_tensors': len(source_state), 'state_tensors_loaded': len(state_mapping), 'parameters_loaded': loaded_parameters, 'encoder_parameters': encoder_parameters, 'parameter_coverage': loaded_parameters / encoder_parameters, 'reshaped_source_tensors': reshaped, 'preserved_target_tensors': sorted(preserved_target), 'skipped_source_tensors': sorted(skipped_source), 'encoder_spec': expected_encoder_spec}
