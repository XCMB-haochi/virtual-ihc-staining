from __future__ import annotations
from torch import nn
from network import FCMAEEncoder
from network import UNeXt2FeatureDecoder
import torch
MARKERS = ('CD68', 'CD45RO', 'HLA-DR', 'Vimentin')

class SingleMarkerStainer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = FCMAEEncoder()
        self.decoder = UNeXt2FeatureDecoder(feature_channels=48)
        self.marker_film = nn.ModuleList((nn.Embedding(4, 2 * c) for c in (96, 192, 384, 768)))
        for layer in self.marker_film:
            nn.init.normal_(layer.weight, std=0.01)
        self.head = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(12, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.bias = nn.Embedding(4, 1)
        nn.init.constant_(self.bias.weight, -2.0)

    def forward(self, image, marker):
        if image.ndim != 4 or image.shape[0:2] != (1, 1):
            raise ValueError('Exactly one single-channel official image is required')
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError('Input dimensions must be divisible by16; no external padding')
        if marker.shape != (1,) or marker.dtype != torch.long:
            raise ValueError('Exactly one integer target marker is required')
        features, _ = self.encoder(image)
        conditioned = []
        for feature, embedding in zip(features, self.marker_film):
            scale, offset = embedding(marker).chunk(2, dim=1)
            conditioned.append(feature * (1 + scale[:, :, None, None]) + offset[:, :, None, None])
        return torch.sigmoid(self.output_head(marker)(self.decoder(conditioned)) + self.bias(marker)[:, :, None, None])

    def output_head(self, marker):
        return self.head
