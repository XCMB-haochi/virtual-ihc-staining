from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ihc_convnext_unet import DecoderConvBlock, IMAGENET_MEAN, IMAGENET_STD, SkipDecoderBlock
from ihc_spatial import SPATIAL_THUMBNAIL_CONDITION_DIMENSION, SPATIAL_THUMBNAIL_CONDITION_MODE
from ihc_unext2 import D4InvariantThumbnailFiLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_REPOSITORY = PROJECT_ROOT / "transfers" / "dinov2"
DINOV2_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_reg4_pretrain.pth"
)
DINOV2_WEIGHT_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
DINOV2_WEIGHT_PATH = PROJECT_ROOT / "runs/public_checkpoints/dinov2_vits14_reg4_pretrain.pth"
DINOV2_LAYERS = (2, 5, 8, 11)


def load_encoder_code() -> nn.Module:
    if not (DINOV2_REPOSITORY / "dinov2/models/vision_transformer.py").is_file():
        raise FileNotFoundError("Run tools/prepare_dinov2.py to prepare the pinned DINOv2 source.")
    revision = subprocess.check_output(
        ["git", "-C", str(DINOV2_REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != DINOV2_REVISION:
        raise ValueError("DINOv2 source revision differs from the registered model.")
    subprocess.run(["git", "-C", str(DINOV2_REPOSITORY), "diff", "--exit-code", "HEAD", "--", "dinov2"],
                   check=True, capture_output=True)
    sys.path.insert(0, str(DINOV2_REPOSITORY))
    from dinov2.hub.backbones import dinov2_vits14_reg
    module_path = Path(sys.modules[dinov2_vits14_reg.__module__].__file__).resolve()
    if not module_path.is_relative_to(DINOV2_REPOSITORY.resolve()):
        raise ValueError("Another DINOv2 package was imported instead of the pinned source.")

    # Checkpoint restoration is offline; public weights are loaded only at stage 1.
    return dinov2_vits14_reg(pretrained=False)


def dinov2_model_specification() -> dict[str, object]:
    return {
        "version": 1,
        "architecture": "dinov2-small-unet",
        "encoder": "dinov2_vits14_reg",
        "encoder_revision": DINOV2_REVISION,
        "encoder_layers": list(DINOV2_LAYERS),
        "feature_channels": [96, 192, 384, 384],
        "feature_strides": [4, 8, 16, 32],
        "encoder_normalization": "imagenet",
        "padding": "symmetric-reflect-multiple14-then-crop",
        "decoder": "bilinear-conv-groupnorm-silu",
        "full_resolution_skip": "shallow-dapi-32",
        "output_activation": "sigmoid",
    }


class DinoV2SmallUNet(nn.Module):
    def __init__(self, out_channels: int = 1, condition_mode: str = "none") -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if condition_mode not in {"none", SPATIAL_THUMBNAIL_CONDITION_MODE}:
            raise ValueError("DINOv2 supports none or spatial-thumbnail-film conditioning.")
        self.encoder = load_encoder_code()
        self.condition_mode = condition_mode
        self.condition_dimension = (
            SPATIAL_THUMBNAIL_CONDITION_DIMENSION if condition_mode != "none" else 0
        )
        self.projections = nn.ModuleList(nn.Conv2d(384, width, 1) for width in (96, 192, 384, 384))
        self.condition_film = D4InvariantThumbnailFiLM(384) if condition_mode != "none" else None
        self.decode_16 = SkipDecoderBlock(384, 384, 384)
        self.decode_8 = SkipDecoderBlock(384, 192, 192)
        self.decode_4 = SkipDecoderBlock(192, 96, 96)
        self.decode_2 = DecoderConvBlock(96, 64)
        self.shallow_skip = DecoderConvBlock(1, 32)
        self.decode_1 = DecoderConvBlock(96, 32)
        self.head = nn.Conv2d(32, out_channels, 1)
        nn.init.constant_(self.head.bias, -2.0)
        self.register_buffer("input_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder."):
                yield parameter

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1 or min(inputs.shape[-2:]) < 32:
            raise ValueError("DINOv2 requires Bx1xHxW input with height and width >=32.")
        if self.condition_film is None:
            if condition is not None:
                raise ValueError("Unconditioned DINOv2 received a condition tensor.")
        elif condition is None or tuple(condition.shape) != (inputs.shape[0], self.condition_dimension):
            raise ValueError("DINOv2 condition shape mismatch.")
        height, width = inputs.shape[-2:]
        pad_h, pad_w = (-height) % 14, (-width) % 14
        top, left = pad_h // 2, pad_w // 2
        padded = F.pad(inputs, (left, pad_w-left, top, pad_h-top), mode="reflect")
        shape = padded.shape[-2:]
        rgb = (padded.repeat(1, 3, 1, 1) - self.input_mean) / self.input_std
        layers = self.encoder.get_intermediate_layers(rgb, n=DINOV2_LAYERS, reshape=True)
        # Reassemble the ViT's common token grid into a convolutional feature pyramid.
        features = [
            F.interpolate(project(layer), size=(shape[0]//stride, shape[1]//stride),
                          mode="bilinear", align_corners=False)
            for project, layer, stride in zip(self.projections, layers, (4, 8, 16, 32))
        ]
        if self.condition_film is not None:
            features[-1] = self.condition_film(features[-1], condition.to(device=inputs.device, dtype=torch.float32))
        current = self.decode_16(features[3], features[2])
        current = self.decode_8(current, features[1])
        current = self.decode_4(current, features[0])
        current = self.decode_2(F.interpolate(current, size=(shape[0]//2, shape[1]//2),
                                             mode="bilinear", align_corners=False))
        current = F.interpolate(current, size=shape, mode="bilinear", align_corners=False)
        current = self.decode_1(torch.cat((current, self.shallow_skip(padded)), dim=1))
        return torch.sigmoid(self.head(current))[..., top:top+height, left:left+width]


def initialize_dinov2_encoder(model: DinoV2SmallUNet, path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != DINOV2_WEIGHT_SHA256:
        raise ValueError("DINOv2 pretrained checkpoint SHA-256 mismatch.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state, strict=True)
    return {"mode": "official-dinov2-vits14-reg4", "checkpoint": str(path.resolve()),
            "checkpoint_sha256": actual, "encoder_revision": DINOV2_REVISION,
            "loaded_tensors": len(state), "scope": "encoder"}
