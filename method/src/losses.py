from __future__ import annotations
from torch import nn
import torch.nn.functional as F
import torch

def _ssim_uniform_impl(prediction: torch.Tensor, target: torch.Tensor, window_size: int=7) -> torch.Tensor:
    """Match scikit-image's default uniform-window SSIM on normalized images."""
    if window_size <= 1 or window_size % 2 != 1:
        raise ValueError('window_size must be an odd integer greater than one.')
    channels = prediction.shape[1]
    kernel = torch.full((channels, 1, window_size, window_size), 1.0 / window_size ** 2, device=prediction.device, dtype=prediction.dtype)
    mu_prediction = F.conv2d(prediction, kernel, groups=channels)
    mu_target = F.conv2d(target, kernel, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    covariance_normalization = window_size ** 2 / (window_size ** 2 - 1)
    variance_prediction = covariance_normalization * (F.conv2d(prediction.square(), kernel, groups=channels) - mu_prediction_sq)
    variance_target = covariance_normalization * (F.conv2d(target.square(), kernel, groups=channels) - mu_target_sq)
    covariance = covariance_normalization * (F.conv2d(prediction * target, kernel, groups=channels) - mu_product)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_product + c1) * (2 * covariance + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (variance_prediction + variance_target + c2)
    return (numerator / denominator.clamp_min(1e-08)).mean(dim=(-2, -1))

def ssim_uniform_per_channel(prediction: torch.Tensor, target: torch.Tensor, window_size: int=7) -> torch.Tensor:
    with torch.amp.autocast(prediction.device.type, enabled=False):
        return _ssim_uniform_impl(prediction.float(), target.float(), window_size)

def apply_d4_transform(inputs: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Apply one indexed D4 transform while preserving the last two axes."""
    if type(transform_index) is not int or not 0 <= transform_index < 8:
        raise ValueError('D4 transform index must be an integer from 0 to 7.')
    rotations = transform_index % 4
    transformed = torch.rot90(inputs, rotations, dims=(-2, -1))
    if transform_index >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed

def invert_d4_transform(inputs: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Invert the corresponding indexed D4 transform."""
    if type(transform_index) is not int or not 0 <= transform_index < 8:
        raise ValueError('D4 transform index must be an integer from 0 to 7.')
    restored = inputs
    if transform_index >= 4:
        restored = torch.flip(restored, dims=(-1,))
    return torch.rot90(restored, -(transform_index % 4), dims=(-2, -1))

class CompetitionLoss(nn.Module):
    """Fixed training objective used by the final teachers and student."""

    def __init__(self, ssim_mode='uniform7'):
        super().__init__()
        assert ssim_mode == 'uniform7'

    def forward(self, prediction, target):
        ssim_loss = 1.0 - ssim_uniform_per_channel(prediction, target).mean()
        l1_loss = F.l1_loss(prediction, target)
        mse_loss = F.mse_loss(prediction, target)
        return 0.7 * ssim_loss + 0.2 * l1_loss + 0.1 * mse_loss
