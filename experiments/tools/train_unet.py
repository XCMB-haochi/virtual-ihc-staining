from __future__ import annotations

import argparse
import copy
import io
import math
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    MARKERS,
    fit_roi_mean_conditioning,
    image_names,
    parse_patch_name,
    rbf_roi_weights,
    resolve_data_root,
    roi_intensity_means,
    split_names_by_roi,
)
from ihc_convnext_unet import (
    ConvNeXtTinyUNet,
    initialize_convnext_tiny_encoder_imagenet1k_v1,
)
from ihc_dinov2 import DinoV2SmallUNet, DINOV2_WEIGHT_PATH, initialize_dinov2_encoder
from ihc_dino_residual import UNeXt2DinoResidualVirtualStainer
from ihc_dino_fusion import UNeXt2DinoBottleneckFusionVirtualStainer
from ihc_unext2 import (
    CYTOLAND_VSCYTO2D_TINY_MODE,
    UNeXt2VirtualStainer,
    initialize_unext2_encoder_from_fcmae,
    initialize_unext2_from_cytoland_vscyto2d,
)
from ihc_torch import (
    CompetitionLoss,
    ExactCompetitionLoss,
    PairedPatchDataset,
    UNet,
    aligned_condition_from_augmentation,
    apply_d4_transform,
    batch_condition_tensor,
    build_model,
    center_crop_like,
    checkpoint_conditioning,
    checkpoint_model_specification,
    competition_score,
    invert_d4_transform,
    model_condition_d4_transform,
    model_specification,
    psnr_from_global_error,
    shift_tensor_edge,
)
from ihc_spatial import (
    SPATIAL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
    SPATIAL_THUMBNAIL_CONDITION_MODE,
    SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
    SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
    SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
    SPATIAL_WHOLE_ROI_CONDITION_MODE,
    build_spatial_stats_conditioning,
    build_spatial_thumbnail_aligned_conditioning,
    build_spatial_thumbnail_conditioning,
    build_spatial_thumbnail_multiscale_conditioning,
    build_spatial_thumbnail_ring_conditioning,
    build_spatial_thumbnail_texture_conditioning,
    build_spatial_whole_roi_conditioning,
)
from ihc_sampling import ActivityHardMixSampler, training_target_activities
from ihc_gradient_projection import PrimaryGradientProjection
from ihc_feature_guidance import (
    build_feature_guidance,
    FrozenFCMAERelationalLoss,
    FrozenFCMAEPerceptualLoss,
    load_frozen_multimarker_fcmae_teacher,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_distillation_target_coverage(directory: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError(f'Distillation targets missing for {len(missing)} training images: {missing[:3]}')


def d4_consistency_transform_index(global_batch_index: int) -> int:
    if type(global_batch_index) is not int or global_batch_index < 0:
        raise ValueError("Global batch index must be a non-negative integer.")
    return 1 + global_batch_index % 7


def validate_d4_consistency_options(
    d4_consistency_weight: float,
    loss_profile: str,
) -> None:
    if not np.isfinite(d4_consistency_weight) or d4_consistency_weight < 0:
        raise ValueError("--d4-consistency-weight must be finite and non-negative.")
    if d4_consistency_weight > 0 and loss_profile != "hybrid":
        raise ValueError("D4 consistency currently requires --loss-profile=hybrid.")


def validate_feature_guidance_options(
    args: argparse.Namespace,
    markers: tuple[str, ...],
) -> None:
    guidance_mode = getattr(args, "feature_guidance_mode", "perceptual")
    if guidance_mode not in {"perceptual", "encoder-relational", "encoder-relational-robust"}:
        raise ValueError("Unknown feature-guidance mode.")
    if guidance_mode in {"encoder-relational", "encoder-relational-robust"} and (
        args.feature_guidance_weight <= 0 or args.feature_guidance_stage != 4
        or args.feature_guidance_batch_samples < 3
        or getattr(args, "gradient_accumulation_steps", 1) != 1
        or getattr(args, "auxiliary_gradient_scope", "all") != "all"
    ):
        raise ValueError("Encoder relational guidance requires stage4, positive weight, >=3 samples, and one full batch.")
    if not np.isfinite(args.feature_guidance_weight) or args.feature_guidance_weight < 0:
        raise ValueError("--feature-guidance-weight must be finite and non-negative.")
    checkpoint_arguments = (
        args.feature_guidance_checkpoint,
        args.feature_guidance_checkpoint_sha256,
        args.feature_guidance_marker,
    )
    if args.feature_guidance_weight == 0:
        if any(value is not None for value in checkpoint_arguments):
            raise ValueError(
                "Feature-guidance checkpoint arguments require a positive weight."
            )
        return
    if not all(value is not None for value in checkpoint_arguments):
        raise ValueError(
            "Feature guidance requires checkpoint, SHA-256, marker, and positive weight."
        )
    if args.loss_profile != "hybrid":
        raise ValueError("Feature guidance requires --loss-profile=hybrid.")
    if args.model_arch != "unext2-fcmae-tiny":
        raise ValueError("Feature guidance currently requires unext2-fcmae-tiny.")
    if args.feature_guidance_marker not in markers:
        raise ValueError("Feature-guidance marker must be among --markers.")
    if not 1 <= args.feature_guidance_stage <= 4:
        raise ValueError("--feature-guidance-stage must be from 1 to 4.")
    if args.feature_guidance_crop_size <= 0 or args.feature_guidance_crop_size % 16:
        raise ValueError("--feature-guidance-crop-size must be divisible by 16.")
    if args.feature_guidance_crop_size > 256:
        raise ValueError("--feature-guidance-crop-size cannot exceed 256.")
    if args.feature_guidance_batch_samples <= 0:
        raise ValueError("--feature-guidance-batch-samples must be positive.")
    digest = str(args.feature_guidance_checkpoint_sha256)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("--feature-guidance-checkpoint-sha256 must be lowercase SHA-256.")


def validate_mixstyle_options(
    probability: float,
    alpha: float,
    model_arch: str,
) -> None:
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("--mixstyle-probability must be finite and in [0, 1].")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("--mixstyle-alpha must be finite and positive.")
    if probability > 0.0 and model_arch != "unet":
        raise ValueError("MixStyle is currently supported only by U-Net.")


def validate_activity_sampler_options(args: argparse.Namespace) -> None:
    mode = str(args.activity_sampler).strip().casefold()
    if mode not in {"none", "hard-mix"}:
        raise ValueError("--activity-sampler must be none or hard-mix.")
    if not np.isfinite(args.activity_sampler_start_ratio) or not (
        0.0 <= args.activity_sampler_start_ratio < 1.0
    ):
        raise ValueError("--activity-sampler-start-ratio must be in [0, 1).")
    if not np.isfinite(args.activity_sampler_hard_fraction) or not (
        0.0 < args.activity_sampler_hard_fraction < 1.0
    ):
        raise ValueError("--activity-sampler-hard-fraction must be in (0, 1).")
    if (
        isinstance(args.activity_sampler_bins, bool)
        or args.activity_sampler_bins < 2
    ):
        raise ValueError("--activity-sampler-bins must be at least two.")
    if mode == "hard-mix" and args.robust_objective != "none":
        raise ValueError("Activity hard-mix cannot be combined with GroupDRO.")


def gradient_accumulation_sample_scale(
    batch_index: int,
    micro_batch_size: int,
    current_batch_size: int,
    accumulation_steps: int,
    dataset_size: int,
) -> float:
    """Weight a micro-batch by its sample share in the current update group."""
    values = (
        batch_index,
        micro_batch_size,
        current_batch_size,
        accumulation_steps,
        dataset_size,
    )
    if any(type(value) is not int for value in values):
        raise ValueError("Gradient accumulation sizes must be integers.")
    if batch_index < 0 or min(values[1:]) <= 0:
        raise ValueError("Gradient accumulation sizes must be positive.")
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    group_first_sample = group_start * micro_batch_size
    group_sample_count = min(
        accumulation_steps * micro_batch_size,
        dataset_size - group_first_sample,
    )
    if group_sample_count <= 0 or current_batch_size > group_sample_count:
        raise ValueError("Gradient accumulation batch index exceeds the dataset.")
    return current_batch_size / group_sample_count


def validate_context_halos(context_halo: int, target_context_halo: int) -> None:
    if not 0 <= context_halo <= 256:
        raise ValueError("--context-halo must be between 0 and 256.")
    if not 0 <= target_context_halo <= context_halo:
        raise ValueError(
            "--train-target-context-halo must be between 0 and --context-halo."
        )


def parse_source_offset(raw: str) -> tuple[int, int]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "Source offset must contain two comma-separated integers."
        ) from error
    if len(values) != 2 or any(not -8 <= value <= 8 for value in values):
        raise ValueError("Source offset must contain two integers from -8 to 8.")
    return values


def supervised_prediction(
    model: torch.nn.Module,
    source: torch.Tensor,
    condition: torch.Tensor | None,
    auxiliary_gradient_scope: str,
) -> torch.Tensor:
    if auxiliary_gradient_scope == "all":
        return model(source, condition)
    if auxiliary_gradient_scope != "decoder-only":
        raise ValueError(
            f"Unsupported auxiliary gradient scope: {auxiliary_gradient_scope}"
        )
    forward = getattr(
        model,
        "forward_with_decoder_only_auxiliary_gradients",
        None,
    )
    if not callable(forward):
        raise TypeError(
            "Decoder-only auxiliary gradients require a compatible UNeXt2 model."
        )
    return forward(source, condition)


def signal_occupancy_loss(logits, target, marker_index):
    """Supervise the fraction of >=64/255 pixels in each 16x16 target block.

    This is a fixed training statistic, not a biological positivity annotation.
    The original image targets and inference predictions are not changed.
    """
    if not 0 <= marker_index < target.shape[1] or logits.shape[1] != target.shape[1]:
        raise ValueError('Signal auxiliary marker index/channel mismatch')
    if target.shape[-2] % 16 or target.shape[-1] % 16:
        raise ValueError('Signal auxiliary target dimensions must be divisible by16')
    binary=(target[:,marker_index:marker_index+1].detach() >= 64/255).float()
    occupancy=torch.nn.functional.avg_pool2d(binary,16,16)
    selected=center_crop_like(logits[:,marker_index:marker_index+1],occupancy).float()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(selected,occupancy,
         pos_weight=selected.new_tensor(4.),reduction='none')
    return (loss.mean(dim=(1,2,3))/(1+3*occupancy.mean(dim=(1,2,3)))).mean()


def training_batch_losses(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor | None,
    criterion: CompetitionLoss | ExactCompetitionLoss,
    d4_consistency_weight: float,
    transform_index: int | None = None,
    sample_weights: torch.Tensor | None = None,
    auxiliary_gradient_scope: str = "all",
    feature_guidance: FrozenFCMAEPerceptualLoss | None = None,
    feature_guidance_weight: float = 0.0,
    feature_guidance_marker_index: int = 0,
    primary_criterion: CompetitionLoss | None = None,
    primary_loss_output: list[torch.Tensor] | None = None,
    distillation_target: torch.Tensor | None = None,
    distillation_criterion: CompetitionLoss | None = None,
    distillation_weight: float = 0.0,
    distillation_marker_index: int = 0,
    signal_auxiliary_weight: float = 0.0,
    signal_auxiliary_marker_index: int = 0,
    signal_auxiliary_output: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_d4_consistency_options(
        d4_consistency_weight,
        "hybrid" if isinstance(criterion, CompetitionLoss) else "exact-score",
    )
    encoder_features = None
    signal_logits = None
    if signal_auxiliary_weight > 0:
        if feature_guidance is not None or auxiliary_gradient_scope != 'all':
            raise ValueError('Signal auxiliary screen cannot combine guidance or gradient routing')
        raw_prediction, signal_logits = model.forward_with_signal_logits(source, condition)
    elif isinstance(feature_guidance, FrozenFCMAERelationalLoss):
        if auxiliary_gradient_scope != "all":
            raise ValueError("Encoder relational guidance requires all-gradient supervision.")
        raw_prediction, encoder_features = model.forward_with_encoder_features(source, condition)
    else:
        raw_prediction = supervised_prediction(
            model,
            source,
            condition,
            auxiliary_gradient_scope,
        )
    prediction = center_crop_like(raw_prediction, target)
    if sample_weights is None:
        supervised_loss = criterion(prediction, target)
    else:
        supervised_loss = criterion(
            prediction,
            target,
            sample_weights=sample_weights,
        )

    if distillation_target is None:
        if distillation_weight != 0.0 or distillation_criterion is not None:
            raise ValueError("Distillation configuration requires teacher targets.")
    else:
        if distillation_criterion is None or not 0.0 < distillation_weight < 1.0:
            raise ValueError("Teacher targets require a distillation loss and weight in (0, 1).")
        if (
            distillation_target.ndim != 4
            or distillation_target.shape[:2] != prediction[:, :1].shape[:2]
        ):
            raise ValueError("Distillation targets must be batch-aligned single-channel images.")
        if not 0 <= distillation_marker_index < prediction.shape[1]:
            raise ValueError('Distillation marker index outside output channels')
        distillation_prediction = center_crop_like(
            prediction[:, distillation_marker_index:distillation_marker_index+1], distillation_target
        )
        distillation_loss = distillation_criterion(
            distillation_prediction, distillation_target
        )
        supervised_loss = (
            (1.0 - distillation_weight) * supervised_loss
            + distillation_weight * distillation_loss
        )

    if primary_criterion is not None:
        if primary_loss_output is None or not isinstance(criterion, CompetitionLoss) or criterion.channel_weights is None:
            raise ValueError("Primary projection requires weighted auxiliary loss and a primary-loss output.")
        primary_loss_output.append(float(criterion.channel_weights[0]) * primary_criterion(prediction[:, :1], target[:, :1]))

    consistency_loss = supervised_loss.detach().new_zeros(())
    if d4_consistency_weight > 0:
        if transform_index is None:
            raise ValueError(
                "A D4 transform index is required when consistency is enabled."
            )
        transformed_source = apply_d4_transform(source, transform_index)
        transformed_prediction = supervised_prediction(
            model,
            transformed_source,
            model_condition_d4_transform(model, condition, transform_index),
            auxiliary_gradient_scope,
        )
        restored_prediction = invert_d4_transform(
            transformed_prediction,
            transform_index,
        )
        restored_prediction = center_crop_like(restored_prediction, target)
        if sample_weights is None:
            consistency_loss = criterion(prediction, restored_prediction)
        else:
            consistency_loss = criterion(
                prediction,
                restored_prediction,
                sample_weights=sample_weights,
            )

    if feature_guidance is None:
        if feature_guidance_weight != 0.0:
            raise ValueError("Positive feature-guidance weight requires its loss module.")
        feature_loss = supervised_loss.detach().new_zeros(())
    else:
        if feature_guidance_weight <= 0.0:
            raise ValueError("Feature-guidance loss module requires a positive weight.")
        if not 0 <= feature_guidance_marker_index < prediction.shape[1]:
            raise ValueError("Feature-guidance marker index is outside prediction channels.")
        feature_loss = feature_guidance(
            encoder_features if encoder_features is not None else prediction[
                :, feature_guidance_marker_index : feature_guidance_marker_index + 1
            ],
            target[:, feature_guidance_marker_index : feature_guidance_marker_index + 1],
        )
    total_loss = (
        supervised_loss
        + d4_consistency_weight * consistency_loss
        + feature_guidance_weight * feature_loss
    )
    if signal_logits is not None:
        signal_loss=signal_occupancy_loss(signal_logits,target,signal_auxiliary_marker_index)
        total_loss=total_loss+signal_auxiliary_weight*signal_loss
        if signal_auxiliary_output is not None:
            signal_auxiliary_output.append(signal_loss.detach())
    return total_loss, supervised_loss, consistency_loss, feature_loss


def prepare_output_dir(path: Path) -> None:
    """Create an output directory, refusing to reuse a non-empty run."""
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path is not a directory: {path}")
        if next(path.iterdir(), None) is not None:
            raise FileExistsError(
                f"Output directory is not empty: {path}. "
                "Choose a new directory; resume training is not supported."
            )
        return
    path.mkdir(parents=True, exist_ok=False)


def seeded_data_generator(seed: int) -> torch.Generator:
    """Return a loader RNG isolated from model initialization and global RNG use."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_group_risk_loader(
    data_root: Path,
    training_names: Sequence[str],
    markers: tuple[str, ...],
    context_halo: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    seed: int,
    target_context_halo: int = 0,
    target_source_offset: tuple[int, int] = (0, 0),
) -> DataLoader:
    """Build the isolated deterministic loader used for epoch risk estimates."""
    return DataLoader(
        PairedPatchDataset(
            data_root,
            training_names,
            markers=markers,
            augment=False,
            context_halo=context_halo,
            target_context_halo=target_context_halo,
            target_source_offset=target_source_offset,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=seeded_data_generator(seed),
    )


def parse_marker_loss_weights(
    raw: str | None,
    markers: tuple[str, ...],
) -> list[float] | None:
    if raw is None:
        return None
    try:
        weights = [float(item.strip()) for item in raw.split(",")]
    except ValueError as error:
        raise ValueError("Marker loss weights must be comma-separated numbers.") from error
    if len(weights) != len(markers):
        raise ValueError(
            f"Expected {len(markers)} marker loss weights for {markers}, got {len(weights)}."
        )
    values = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Marker loss weights must be finite and non-negative.")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Marker loss weights must have a finite positive sum.")
    return (values / total).tolist()


def add_robust_objective_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robust-objective",
        choices=("none", "group-dro"),
        default="none",
        help="Optionally optimize the worst training ROI with epoch-level GroupDRO.",
    )
    parser.add_argument(
        "--group-dro-eta",
        type=float,
        default=5.0,
        help="Exponentiated-gradient step size for epoch-level GroupDRO.",
    )


def validate_robust_objective_options(args: argparse.Namespace) -> None:
    if not np.isfinite(args.group_dro_eta) or args.group_dro_eta <= 0:
        raise ValueError("--group-dro-eta must be finite and positive.")
    if args.robust_objective == "none":
        return
    if args.robust_objective != "group-dro":
        raise ValueError(f"Unsupported robust objective: {args.robust_objective}")
    if args.loss_profile != "hybrid":
        raise ValueError("GroupDRO requires --loss-profile=hybrid.")
    if args.roi_weight_mode != "none":
        raise ValueError(
            "GroupDRO cannot be combined with --roi-weight-mode=test-mean-rbf."
        )
    if args.ema_decay != 0:
        raise ValueError("GroupDRO requires --ema-decay=0.")
    if getattr(args, "marker_loss_weights_resolved", None) is not None:
        raise ValueError("GroupDRO requires --marker-loss-weights to be unset.")


def empirical_group_counts(names: Sequence[str]) -> dict[str, int]:
    if not names:
        raise ValueError("Group counts require at least one training image.")
    counts = Counter(parse_patch_name(name).roi for name in names)
    return {group: counts[group] for group in sorted(counts)}


def empirical_group_probabilities(names: Sequence[str]) -> dict[str, float]:
    counts = empirical_group_counts(names)
    total = float(sum(counts.values()))
    return {group: count / total for group, count in counts.items()}


def validate_training_group_coverage(
    expected_names: Sequence[str],
    actual_names: Sequence[str],
) -> tuple[str, ...]:
    expected_groups = {parse_patch_name(name).roi for name in expected_names}
    actual_groups = {parse_patch_name(name).roi for name in actual_names}
    if actual_groups != expected_groups:
        missing = sorted(expected_groups - actual_groups)
        unexpected = sorted(actual_groups - expected_groups)
        raise ValueError(
            "GroupDRO post-limit training groups do not match the full split: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return tuple(sorted(expected_groups))


def _validate_group_distribution(
    distribution: Mapping[str, float],
    label: str,
    expected_groups: set[str] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    groups = tuple(sorted(distribution))
    if not groups:
        raise ValueError(f"{label} must contain at least one group.")
    if expected_groups is not None and set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        unexpected = sorted(set(groups) - expected_groups)
        raise ValueError(
            f"{label} groups do not match: missing={missing}, unexpected={unexpected}."
        )
    values = np.asarray([distribution[group] for group in groups], dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} values must be finite and positive.")
    total = float(values.sum())
    if not np.isfinite(total) or not np.isclose(total, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} values must sum to one, got {total}.")
    return groups, values


def group_dro_q_over_p(
    group_q: Mapping[str, float],
    group_p: Mapping[str, float],
) -> dict[str, float]:
    groups, q_values = _validate_group_distribution(group_q, "GroupDRO q")
    p_groups, p_values = _validate_group_distribution(
        group_p,
        "GroupDRO p",
        expected_groups=set(groups),
    )
    if p_groups != groups:
        raise RuntimeError("Sorted GroupDRO p and q groups disagree.")
    result = {
        group: float(q_value / p_value)
        for group, q_value, p_value in zip(groups, q_values, p_values)
    }
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        float32_values = np.asarray(list(result.values()), dtype=np.float32)
    if not np.isfinite(float32_values).all() or (float32_values <= 0).any():
        raise ValueError("GroupDRO q/p must remain finite and positive in float32.")
    return result


def effective_group_count(group_q: Mapping[str, float]) -> float:
    _, q_values = _validate_group_distribution(group_q, "GroupDRO q")
    return float(1.0 / np.square(q_values).sum())


def update_group_dro_q(
    group_q: Mapping[str, float],
    group_risks: Mapping[str, float],
    eta: float,
) -> dict[str, float]:
    groups, q_values = _validate_group_distribution(group_q, "GroupDRO q")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("GroupDRO eta must be finite and positive.")
    risk_groups = set(group_risks)
    expected_groups = set(groups)
    if risk_groups != expected_groups:
        missing = sorted(expected_groups - risk_groups)
        unexpected = sorted(risk_groups - expected_groups)
        raise ValueError(
            "GroupDRO risks do not match q groups: "
            f"missing={missing}, unexpected={unexpected}."
        )
    risks = np.asarray([group_risks[group] for group in groups], dtype=np.float64)
    if not np.isfinite(risks).all():
        raise ValueError("GroupDRO risks must be finite.")
    centered_risks = risks - risks.mean()
    with np.errstate(over="ignore", invalid="ignore"):
        logits = np.log(q_values) + eta * centered_risks
    if not np.isfinite(logits).all():
        raise ValueError("GroupDRO update logits must be finite.")
    logits -= logits.max()
    updated = np.exp(logits)
    updated /= updated.sum()
    result = {group: float(value) for group, value in zip(groups, updated)}
    _validate_group_distribution(result, "Updated GroupDRO q")
    return result


def initialize_group_dro(
    mode: str,
    training_names: Sequence[str],
    eta: float,
) -> dict[str, object] | None:
    if mode == "none":
        return None
    if mode != "group-dro":
        raise ValueError(f"Unsupported robust objective: {mode}")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("GroupDRO eta must be finite and positive.")
    group_counts = empirical_group_counts(training_names)
    total_count = float(sum(group_counts.values()))
    group_p = {
        group: count / total_count for group, count in group_counts.items()
    }
    uniform_probability = 1.0 / len(group_p)
    group_q = {group: uniform_probability for group in group_p}
    return {
        "version": 1,
        "mode": "group-dro",
        "eta": float(eta),
        "groups": list(group_p),
        "counts": group_counts,
        "p": group_p,
        "initial_q": group_q,
        "initial_q_over_p": group_dro_q_over_p(group_q, group_p),
        "initial_effective_groups": effective_group_count(group_q),
        "q_reset_scope": "process-stage",
        "risk_pass": {
            "scope": "full-training-fold",
            "augment": False,
            "shuffle": False,
            "amp": False,
            "dtype": "float32",
            "model": "online-non-ema",
        },
    }


def group_dro_epoch_report(
    group_p: Mapping[str, float],
    q_before: Mapping[str, float],
    group_risks: Mapping[str, float],
    risk_counts: Mapping[str, int],
    eta: float,
) -> tuple[dict[str, object], dict[str, float]]:
    groups, _ = _validate_group_distribution(q_before, "GroupDRO q")
    if set(risk_counts) != set(groups):
        missing = sorted(set(groups) - set(risk_counts))
        unexpected = sorted(set(risk_counts) - set(groups))
        raise ValueError(
            "GroupDRO risk counts do not match q groups: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if any(
        not isinstance(risk_counts[group], int) or risk_counts[group] <= 0
        for group in groups
    ):
        raise ValueError("GroupDRO risk counts must be positive integers.")
    q_before_copy = {group: float(q_before[group]) for group in groups}
    q_after = update_group_dro_q(q_before_copy, group_risks, eta)
    q_over_p_before = group_dro_q_over_p(q_before_copy, group_p)
    q_over_p_after = group_dro_q_over_p(q_after, group_p)
    q_over_p_before_values = list(q_over_p_before.values())
    q_over_p_after_values = list(q_over_p_after.values())
    report = {
        "risks": {group: float(group_risks[group]) for group in groups},
        "risk_counts": {group: risk_counts[group] for group in groups},
        "risk_mean": float(np.mean([group_risks[group] for group in groups])),
        "p": {group: float(group_p[group]) for group in groups},
        "q_before": q_before_copy,
        "q_after": q_after,
        "q_over_p": q_over_p_before,
        "q_over_p_before": q_over_p_before,
        "q_over_p_after": q_over_p_after,
        "q_over_p_range": {
            "before": {
                "min": min(q_over_p_before_values),
                "max": max(q_over_p_before_values),
            },
            "after": {
                "min": min(q_over_p_after_values),
                "max": max(q_over_p_after_values),
            },
        },
        "effective_groups": {
            "before": effective_group_count(q_before_copy),
            "after": effective_group_count(q_after),
        },
    }
    return report, q_after


def git_commit(project_root: Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def runtime_metadata(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpus: list[dict[str, object]] = []
    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "git_commit": git_commit(project_root),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": gpu_count,
        "gpus": gpus,
    }


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Save beside the destination and atomically replace it when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def model_spec_from_args(args: argparse.Namespace) -> dict[str, object]:
    specification = getattr(args, "model_spec", None)
    if specification is not None:
        return specification
    return model_specification("unet", args.base_channels, args.block_type)


def _checkpoint_markers_for_subset_transfer(
    checkpoint: dict[str, object],
) -> tuple[str, ...]:
    raw_markers = checkpoint.get("markers")
    if not isinstance(raw_markers, (list, tuple)) or not raw_markers:
        raise ValueError(
            "Marker-subset initialization requires a non-empty list or tuple in "
            "checkpoint markers metadata."
        )
    if any(not isinstance(marker, str) or not marker for marker in raw_markers):
        raise ValueError("Checkpoint markers metadata contains an invalid marker name.")
    markers = tuple(raw_markers)
    if len(set(markers)) != len(markers):
        raise ValueError("Checkpoint markers metadata contains duplicate marker names.")
    return markers


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: dict[str, object],
    target_markers: tuple[str, ...],
    target_model_spec: dict[str, object],
    target_conditioning: dict[str, object],
    *,
    marker_subset: bool = False,
    add_conditioning: bool = False,
) -> dict[str, object]:
    """Initialize a run strictly, or explicitly transfer U-Net marker channels."""
    if marker_subset and add_conditioning:
        raise ValueError("Marker-subset and add-conditioning initialization are exclusive.")
    if add_conditioning:
        if tuple(checkpoint.get("markers", ())) != target_markers:
            raise ValueError(
                f"Initial checkpoint markers {checkpoint.get('markers')} do not match "
                f"{target_markers}."
            )
        if checkpoint_model_specification(checkpoint) != target_model_spec:
            raise ValueError("Initial checkpoint model specification does not match the run.")
        source_conditioning = checkpoint_conditioning(checkpoint)
        source_mode = source_conditioning["mode"]
        target_mode = target_conditioning.get("mode")
        valid_transition = (
            source_mode == "none" and target_mode == SPATIAL_CONDITION_MODE
        ) or (
            source_mode == SPATIAL_CONDITION_MODE
            and target_mode == SPATIAL_THUMBNAIL_CONDITION_MODE
        ) or (
            source_mode == SPATIAL_THUMBNAIL_CONDITION_MODE
            and target_mode == SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE
        ) or (
            source_mode == SPATIAL_THUMBNAIL_CONDITION_MODE
            and target_mode == SPATIAL_THUMBNAIL_RING_CONDITION_MODE
        ) or (
            source_mode == SPATIAL_THUMBNAIL_CONDITION_MODE
            and target_mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE
        )
        if not valid_transition and target_mode == SPATIAL_CONDITION_MODE:
            raise ValueError("Add-conditioning initialization requires an unconditioned source.")
        if not valid_transition:
            raise ValueError(
                "Unsupported add-conditioning transition: "
                f"{source_mode!r} -> {target_mode!r}."
            )
        source_state = checkpoint.get("model")
        if not isinstance(source_state, dict):
            raise ValueError("Initial checkpoint model state must be a dictionary.")
        target_state = model.state_dict()
        shared_keys = set(source_state)
        missing_shared = sorted(shared_keys - set(target_state))
        added_keys = sorted(set(target_state) - shared_keys)
        if missing_shared or not added_keys or any(
            not key.startswith("condition_film.") for key in added_keys
        ):
            raise ValueError(
                "Add-conditioning state keys are incompatible: "
                f"missing_shared={missing_shared}, added={added_keys}."
            )
        for key, value in source_state.items():
            if target_state[key].shape != value.shape:
                raise ValueError(
                    f"Initial checkpoint tensor shape differs for {key}: "
                    f"{tuple(value.shape)} != {tuple(target_state[key].shape)}."
                )
        initialized = dict(target_state)
        initialized.update(source_state)
        model.load_state_dict(initialized, strict=True)
        return {
            "version": 1,
            "mode": "add-conditioning",
            "transition": f"{source_mode}-to-{target_mode}",
            "source_conditioning": source_conditioning,
            "target_conditioning": target_conditioning,
            "added_tensors": added_keys,
            "source_markers": list(target_markers),
            "target_markers": list(target_markers),
        }

    if not marker_subset:
        if tuple(checkpoint.get("markers", ())) != target_markers:
            raise ValueError(
                f"Initial checkpoint markers {checkpoint.get('markers')} do not match "
                f"{target_markers}."
            )
        if checkpoint_model_specification(checkpoint) != target_model_spec:
            raise ValueError("Initial checkpoint model specification does not match the run.")
        if checkpoint_conditioning(checkpoint) != target_conditioning:
            raise ValueError("Initial checkpoint conditioning does not match the run.")
        model.load_state_dict(checkpoint["model"])
        return {
            "version": 1,
            "mode": "strict",
            "source_markers": list(target_markers),
            "target_markers": list(target_markers),
        }

    source_markers = _checkpoint_markers_for_subset_transfer(checkpoint)
    if not target_markers or any(
        not isinstance(marker, str) or not marker for marker in target_markers
    ):
        raise ValueError("Target markers contain an invalid marker name.")
    if len(set(target_markers)) != len(target_markers):
        raise ValueError("Target markers contain duplicate marker names.")
    missing_markers = [
        marker for marker in target_markers if marker not in source_markers
    ]
    if missing_markers:
        raise ValueError(
            f"Target markers are missing from initial checkpoint: {missing_markers}."
        )

    source_model_spec = checkpoint_model_specification(checkpoint)
    if (
        source_model_spec.get("architecture") != "unet"
        or target_model_spec.get("architecture") != "unet"
    ):
        raise ValueError("Marker-subset initialization is supported only for U-Net.")
    if source_model_spec != target_model_spec:
        raise ValueError("Initial checkpoint model specification does not match the run.")
    if checkpoint_conditioning(checkpoint) != target_conditioning:
        raise ValueError("Initial checkpoint conditioning does not match the run.")

    source_state = checkpoint.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("Initial checkpoint model state must be a dictionary.")
    target_state = model.state_dict()
    head_keys = ("head.weight", "head.bias")
    if any(key not in source_state or key not in target_state for key in head_keys):
        raise ValueError("Initial checkpoint does not contain the expected U-Net head.")

    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise ValueError(
            "Initial checkpoint U-Net state keys do not match the run: "
            f"missing={missing}, unexpected={unexpected}."
        )

    for key in head_keys:
        source_tensor = source_state[key]
        target_tensor = target_state[key]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f"Initial checkpoint tensor {key} is not a tensor.")
        if source_tensor.ndim == 0 or source_tensor.shape[0] != len(source_markers):
            source_channels = source_tensor.shape[0] if source_tensor.ndim else 0
            raise ValueError(
                f"Initial checkpoint tensor {key} has {source_channels} marker "
                f"channels but metadata lists {len(source_markers)}."
            )
        if target_tensor.ndim == 0 or target_tensor.shape[0] != len(target_markers):
            raise ValueError(
                f"Target U-Net tensor {key} does not match its marker metadata."
            )

    source_indices = [source_markers.index(marker) for marker in target_markers]
    transferred_state: dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f"Initial checkpoint tensor {key} is not a tensor.")
        candidate = (
            source_tensor.index_select(
                0,
                torch.tensor(
                    source_indices,
                    dtype=torch.long,
                    device=source_tensor.device,
                ),
            )
            if key in head_keys
            else source_tensor
        )
        if candidate.shape != target_tensor.shape:
            raise ValueError(
                f"Initial checkpoint tensor {key} has incompatible shape "
                f"{tuple(candidate.shape)}; expected {tuple(target_tensor.shape)}."
            )
        transferred_state[key] = candidate
    model.load_state_dict(transferred_state, strict=True)

    return {
        "version": 1,
        "mode": "marker-subset",
        "source_markers": list(source_markers),
        "target_markers": list(target_markers),
        "target_to_source_indices": source_indices,
        "head_channel_mapping": [
            {
                "marker": marker,
                "target_index": target_index,
                "source_index": source_index,
            }
            for target_index, (marker, source_index) in enumerate(
                zip(target_markers, source_indices)
            )
        ],
        "shared_state_tensors_copied": len(target_state) - len(head_keys),
        "sliced_head_tensors": list(head_keys),
    }


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def set_encoder_freeze_for_epoch(model: torch.nn.Module, epoch: int, freeze_epochs: int) -> bool:
    if freeze_epochs == 0:
        return False
    if not isinstance(model, UNeXt2VirtualStainer):
        raise ValueError("Encoder freeze requires UNeXt2.")
    frozen = epoch <= freeze_epochs
    for parameter in model.encoder_parameters():
        parameter.requires_grad_(not frozen)
        if frozen:
            parameter.grad = None
    return frozen


def build_optimizer(
    model: torch.nn.Module,
    architecture: str,
    learning_rate: float,
    encoder_learning_rate: float | None,
    weight_decay: float,
    encoder_stage_decay: float = 1.0,
) -> tuple[AdamW, list[dict[str, object]]]:
    """Build AdamW while keeping legacy models on their original single group."""
    if not math.isfinite(encoder_stage_decay) or not 0 < encoder_stage_decay <= 1:
        raise ValueError("encoder_stage_decay must be in (0,1].")
    if encoder_stage_decay != 1.0 and architecture != "unext2-fcmae-tiny":
        raise ValueError("Stage decay is supported only for unext2-fcmae-tiny.")
    if architecture in {"unext2-fcmae-tiny-dinov2-residual", "unext2-fcmae-tiny-dinov2-fusion"}:
        if architecture == "unext2-fcmae-tiny-dinov2-residual":
            if not isinstance(model, UNeXt2DinoResidualVirtualStainer):
                raise TypeError("DINOv2 residual requires UNeXt2DinoResidualVirtualStainer.")
            dino_encoder = model.dinov2_residual.encoder
        else:
            if not isinstance(model, UNeXt2DinoBottleneckFusionVirtualStainer):
                raise TypeError("DINOv2 fusion requires UNeXt2DinoBottleneckFusionVirtualStainer.")
            dino_encoder = model.dinov2_fusion.encoder
        groups = [
            {"params": list(model.decoder_parameters()), "lr": learning_rate, "group_name": "decoder_head"},
            {"params": list(model.encoder.parameters()), "lr": learning_rate if encoder_learning_rate is None else encoder_learning_rate, "group_name": "encoder"},
            {"params": list(dino_encoder.parameters()), "lr": learning_rate * 0.1, "group_name": "dinov2_encoder"},
        ]
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(parameter) for parameter in model.parameters()}:
            raise RuntimeError("DINOv2 hybrid optimizer groups must cover each parameter once.")
        optimizer = AdamW(groups, weight_decay=weight_decay)
        return optimizer, [
            {"name": group["group_name"], "initial_learning_rate": group["lr"],
             "parameters": sum(parameter.numel() for parameter in group["params"])}
            for group in groups
        ]
    grouped_architectures = {
        "convnext-tiny-unet",
        "dinov2-small-unet",
        "unext2-fcmae-tiny",
        "unext2-fcmae-base",
        "unext2-fcmae-tiny-widehead",
        "unext2-fcmae-tiny-marker-decoders",
        "unext2-fcmae-tiny-cross-marker-feedback",
        "unext2-fcmae-tiny-dapi-detail-residual",
        "unext2-fcmae-tiny-shallow-dapi-skip",
        "unext2-fcmae-tiny-local-condition",
        "unext2-fcmae-tiny-signal-aux",
    }
    if architecture not in grouped_architectures:
        if encoder_learning_rate is not None:
            raise ValueError(
                "--encoder-learning-rate requires ConvNeXt U-Net or UNeXt2."
            )
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        return optimizer, [
            {
                "name": "all",
                "initial_learning_rate": learning_rate,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        ]

    if architecture == "convnext-tiny-unet":
        if not isinstance(model, ConvNeXtTinyUNet):
            raise TypeError("convnext-tiny-unet requires ConvNeXtTinyUNet.")
    elif architecture == "dinov2-small-unet":
        if not isinstance(model, DinoV2SmallUNet):
            raise TypeError("dinov2-small-unet requires DinoV2SmallUNet.")
    elif not isinstance(model, UNeXt2VirtualStainer):
        raise TypeError(f"{architecture} requires UNeXt2VirtualStainer.")
    decoder_parameters = list(model.decoder_parameters())
    encoder_parameters = list(model.encoder_parameters())
    all_parameters = list(model.parameters())
    grouped_ids = [id(parameter) for parameter in decoder_parameters + encoder_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("ConvNeXt optimizer parameter groups overlap.")
    if set(grouped_ids) != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError("ConvNeXt optimizer parameter groups do not cover the model.")
    resolved_encoder_lr = (
        learning_rate if encoder_learning_rate is None else encoder_learning_rate
    )
    parameter_groups = [
        {
            "params": decoder_parameters,
            "lr": learning_rate,
            "group_name": "decoder_head",
        },
        {
            "params": encoder_parameters,
            "lr": resolved_encoder_lr,
            "group_name": "encoder",
        },
    ]
    if encoder_stage_decay != 1.0:
        modules = [('stem', model.encoder.stem), *[(f'stage{i}', module) for i,module in enumerate(model.encoder.stages)]]
        parameter_groups = [parameter_groups[0]] + [
            {'params': list(module.parameters()), 'lr': resolved_encoder_lr * encoder_stage_decay ** (len(modules)-1-i),
             'group_name': f'encoder_{name}'} for i,(name,module) in enumerate(modules)
        ]
        ids = [id(p) for group in parameter_groups for p in group['params']]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
            raise RuntimeError('Stage-decay groups must cover every parameter exactly once.')
    optimizer = AdamW(parameter_groups, weight_decay=weight_decay)
    report = [{'name': g['group_name'], 'initial_learning_rate': g['lr'],
               'parameters': sum(p.numel() for p in g['params'])} for g in parameter_groups]
    return optimizer, report


def limit_evenly(names: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(names) <= limit:
        return names
    indices = np.linspace(0, len(names) - 1, num=limit, dtype=int)
    return [names[index] for index in indices]


@torch.inference_mode()
def evaluate_group_risks(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompetitionLoss,
    device: torch.device,
    expected_counts: Mapping[str, int],
    condition_by_roi: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Measure unweighted per-ROI risks in a deterministic float32 pass."""
    expected_groups = tuple(sorted(expected_counts))
    if not expected_groups or len(set(expected_groups)) != len(expected_groups):
        raise ValueError("GroupDRO risk evaluation requires unique training groups.")
    if any(
        not isinstance(expected_counts[group], int) or expected_counts[group] <= 0
        for group in expected_groups
    ):
        raise ValueError("Expected GroupDRO risk counts must be positive integers.")
    risk_sums = {group: 0.0 for group in expected_groups}
    risk_counts = {group: 0 for group in expected_groups}
    was_training = model.training
    model.eval()
    try:
        for source, target, batch_names in loader:
            source = source.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            target = target.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            condition = batch_condition_tensor(batch_names, condition_by_roi, device)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                prediction = center_crop_like(model(source, condition), target).float()
                per_sample = criterion.per_sample(prediction, target).float()
            if per_sample.ndim != 1 or per_sample.shape[0] != len(batch_names):
                raise RuntimeError(
                    "CompetitionLoss.per_sample returned an invalid GroupDRO shape."
                )
            losses = per_sample.detach().to(device="cpu", dtype=torch.float64).tolist()
            for name, loss in zip(batch_names, losses):
                group = parse_patch_name(name).roi
                if group not in risk_sums:
                    raise ValueError(
                        f"GroupDRO risk pass encountered non-training group {group}."
                    )
                if not np.isfinite(loss):
                    raise ValueError(f"GroupDRO risk for {group} is non-finite.")
                risk_sums[group] += float(loss)
                risk_counts[group] += 1
    finally:
        model.train(was_training)

    count_mismatches = {
        group: {
            "expected": expected_counts[group],
            "observed": risk_counts[group],
        }
        for group in expected_groups
        if risk_counts[group] != expected_counts[group]
    }
    if count_mismatches:
        raise ValueError(
            f"GroupDRO risk counts do not match the training set: {count_mismatches}."
        )
    risks = {
        group: risk_sums[group] / risk_counts[group] for group in expected_groups
    }
    if not all(np.isfinite(value) for value in risks.values()):
        raise ValueError("GroupDRO risks must be finite.")
    return risks, risk_counts


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    markers: tuple[str, ...],
    jpeg_quality: int,
    condition_by_roi: dict[str, float] | None = None,
    output_source_offset: tuple[int, int] = (0, 0),
    selection_ssim_mode: str = "uniform7",
) -> dict[str, object]:
    if selection_ssim_mode not in ("uniform7", "gaussian11"):
        raise ValueError("Unknown checkpoint-selection SSIM mode")
    ssim_options = (
        dict(win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        if selection_ssim_mode == "gaussian11"
        else dict(win_size=7, gaussian_weights=False, use_sample_covariance=True)
    )
    model.eval()
    ssim_sum = np.zeros(len(markers), dtype=np.float64)
    psnr_sum = np.zeros(len(markers), dtype=np.float64)
    official_score_sum = np.zeros(len(markers), dtype=np.float64)
    squared_error_sum = np.zeros(len(markers), dtype=np.float64)
    pixel_count = np.zeros(len(markers), dtype=np.int64)
    count = 0
    for source, target, batch_names in loader:
        source = source.to(device, non_blocking=True)
        condition = batch_condition_tensor(batch_names, condition_by_roi, device)
        prediction_tensor = center_crop_like(model(source, condition), target)
        prediction = (
            shift_tensor_edge(prediction_tensor, output_source_offset).cpu().numpy()
            * 255.0
        )
        target_np = target.numpy() * 255.0
        for batch_index in range(prediction.shape[0]):
            for marker_index in range(len(markers)):
                candidate = prediction[batch_index, marker_index]
                if jpeg_quality > 0:
                    buffer = io.BytesIO()
                    gray = np.clip(np.rint(candidate), 0, 255).astype(np.uint8)
                    Image.fromarray(gray, mode="L").convert("RGB").save(
                        buffer, format="JPEG", quality=jpeg_quality
                    )
                    buffer.seek(0)
                    with Image.open(buffer) as decoded:
                        candidate = np.asarray(decoded.convert("L"), dtype=np.float32)
                truth = target_np[batch_index, marker_index]
                image_ssim = structural_similarity(
                    truth, candidate, data_range=255, **ssim_options
                )
                image_psnr = peak_signal_noise_ratio(
                    truth, candidate, data_range=255
                )
                ssim_sum[marker_index] += image_ssim
                psnr_sum[marker_index] += image_psnr
                official_score_sum[marker_index] += competition_score(image_ssim, image_psnr)
                error = truth.astype(np.float64) - candidate.astype(np.float64)
                squared_error_sum[marker_index] += float(np.square(error).sum())
                pixel_count[marker_index] += int(error.size)
        count += source.shape[0]

    ssim = (ssim_sum / count).tolist()
    psnr = (psnr_sum / count).tolist()
    global_psnr = [
        psnr_from_global_error(squared_error_sum[index], int(pixel_count[index]))
        for index in range(len(markers))
    ]
    mean_image_scores = (official_score_sum / count).tolist()
    global_mse_scores = [
        competition_score(ssim[index], global_psnr[index])
        for index in range(len(markers))
    ]
    return {
        "count": count,
        "mean_ssim": float(np.mean(ssim)),
        "mean_psnr": float(np.mean(psnr)),
        "mean_competition_score": float(np.mean(mean_image_scores)),
        "mean_competition_score_mean_image_psnr": float(
            np.mean(mean_image_scores)
        ),
        "mean_competition_score_global_mse_psnr": float(np.mean(global_mse_scores)),
        "implementation": "scikit-image",
        "ssim_mode": selection_ssim_mode,
        "ssim_options": ssim_options,
        "jpeg_quality": jpeg_quality if jpeg_quality > 0 else None,
        "markers": {
            marker: {
                "ssim": ssim[index],
                "psnr": psnr[index],
                "competition_score": mean_image_scores[index],
                "psnr_mean_image": psnr[index],
                "competition_score_mean_image_psnr": mean_image_scores[index],
                "psnr_global_mse": global_psnr[index],
                "competition_score_global_mse_psnr": global_mse_scores[index],
            }
            for index, marker in enumerate(markers)
        },
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_score: float | None,
    args: argparse.Namespace,
    markers: tuple[str, ...],
) -> None:
    model_spec = model_spec_from_args(args)
    atomic_torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "markers": markers,
            "base_channels": args.base_channels,
            "block_type": args.block_type,
            "model_spec": model_spec,
            "conditioning": args.conditioning_spec,
            "initialization": getattr(args, "init_checkpoint_report", None),
            "args": vars(args),
        },
        path,
    )


def save_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    score: float | None,
    args: argparse.Namespace,
    markers: tuple[str, ...],
    psnr_aggregation: str,
) -> None:
    """Save a compact inference checkpoint for an alternate validation metric."""
    model_spec = model_spec_from_args(args)
    atomic_torch_save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_score": score,
            "markers": markers,
            "base_channels": args.base_channels,
            "block_type": args.block_type,
            "model_spec": model_spec,
            "conditioning": args.conditioning_spec,
            "initialization": getattr(args, "init_checkpoint_report", None),
            "args": vars(args),
            "selection_psnr_aggregation": psnr_aggregation,
        },
        path,
    )


def parse_validation_epochs(raw: str | None, total_epochs: int) -> list[int]:
    if total_epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if raw is None:
        return list(range(1, total_epochs + 1))
    try:
        epochs = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            "--validation-epochs must contain comma-separated integers."
        ) from error
    if (
        not epochs
        or epochs != sorted(set(epochs))
        or epochs[0] < 1
        or epochs[-1] > total_epochs
    ):
        raise ValueError(
            "--validation-epochs must be unique increasing epochs within "
            "[1, --epochs]."
        )
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a grayscale virtual-staining model.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unet_multitask"))
    parser.add_argument("--val-rois", default="ROI020,ROI021,ROI022,ROI023,ROI024")
    parser.add_argument(
        "--train-rois",
        help=(
            "Optional comma-separated training ROI subset. It must be disjoint "
            "from --val-rois; omitting it uses every non-validation ROI."
        ),
    )
    parser.add_argument(
        "--markers",
        default=",".join(MARKERS),
        help="Comma-separated target markers.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--validation-epochs",
        help=(
            "Optional comma-separated epochs that run validation and checkpoint "
            "selection; omitting it validates every epoch."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model-arch",
        choices=(
            "unet",
            "nafnet-width32",
            "convnext-tiny-unet",
            "dinov2-small-unet",
            "unext2-fcmae-tiny-dinov2-residual",
            "unext2-fcmae-tiny-dinov2-fusion",
            "unext2-fcmae-tiny",
            "unext2-fcmae-base",
            "unext2-fcmae-tiny-widehead",
            "unext2-fcmae-tiny-marker-decoders",
            "unext2-fcmae-tiny-cross-marker-feedback",
            "unext2-fcmae-tiny-dapi-detail-residual",
            "unext2-fcmae-tiny-shallow-dapi-skip",
            "unext2-fcmae-tiny-local-condition",
            "unext2-fcmae-tiny-signal-aux",
            "dualbranch-swin-s",
            "restormer-small",
        ),
        default="unet",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--block-type", choices=("residual", "dense"), default="residual")
    parser.add_argument(
        "--condition-mode",
        choices=(
            "none",
            "roi-mean-film",
            SPATIAL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE,
            SPATIAL_THUMBNAIL_RING_CONDITION_MODE,
            SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        ),
        default="none",
        help=(
            "Inject input-only ROI statistics, spatial statistics, or a DAPI "
            "neighborhood thumbnail through bottleneck FiLM or aligned fusion."
        ),
    )
    parser.add_argument("--context-halo", type=int, default=0)
    parser.add_argument(
        "--train-target-context-halo",
        type=int,
        default=0,
        help=(
            "Supervise this halo around each training target; validation targets "
            "remain official 256x256 centers. Must not exceed --context-halo."
        ),
    )
    parser.add_argument(
        "--train-target-source-offset",
        default="0,0",
        help="Sample training targets from this fixed row,column source offset.",
    )
    parser.add_argument(
        "--validation-output-source-offset",
        default="0,0",
        help="Sample validation predictions from this fixed row,column source offset.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Initialize from an exactly matching architecture and conditioning spec; "
            "legacy checkpoints cannot implicitly add FiLM parameters."
        ),
    )
    parser.add_argument(
        "--init-marker-subset",
        action="store_true",
        help=(
            "Explicitly initialize a reordered marker subset from a compatible "
            "multi-marker U-Net checkpoint, including marker-indexed head slices."
        ),
    )
    parser.add_argument(
        "--init-add-conditioning",
        action="store_true",
        help=(
            "Copy all shared weights from an unconditioned U-Net checkpoint and "
            "retain the new zero-output spatial conditioning tensors."
        ),
    )
    parser.add_argument(
        "--encoder-init",
        choices=("none", "imagenet1k-v1", "dinov2"),
        default="none",
        help=(
            "Opt-in public encoder initialization for a fresh stage1 run; "
            "checkpoint initialization always requires none."
        ),
    )
    parser.add_argument(
        "--fcmae-encoder-init-checkpoint",
        type=Path,
        help=(
            "Initialize a fresh UNeXt2 stage1 encoder from a fold-local DAPI "
            "FCMAE checkpoint; the downstream checkpoint remains self-contained."
        ),
    )
    parser.add_argument(
        "--cytoland-init-checkpoint",
        type=Path,
        help="Initialize fresh Tiny UNeXt2 from the audited Cytoland checkpoint.",
    )
    parser.add_argument(
        "--cytoland-init-mode",
        choices=(CYTOLAND_VSCYTO2D_TINY_MODE,),
    )
    parser.add_argument("--cytoland-init-sha256")
    parser.add_argument(
        "--cytoland-init-scope",
        choices=("encoder", "encoder-decoder"),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--minimum-learning-rate",
        type=float,
        default=0.0,
        help="Minimum learning rate reached by the epoch-level cosine schedule.",
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        help=(
            "Optional ConvNeXt/UNeXt2 encoder LR; --learning-rate remains "
            "the decoder/head LR."
        ),
    )
    parser.add_argument(
        "--encoder-drop-path-rate",
        type=float,
        default=0.0,
        help="Maximum linearly increasing stochastic-depth rate in UNeXt2 encoder.",
    )
    parser.add_argument("--encoder-stage-decay", type=float, default=1.0, help="UNeXt2-Tiny stage-wise LR decay from final encoder stage toward the stem.")
    parser.add_argument("--encoder-freeze-epochs", type=int, default=0, help="Initial supervised epochs training only the decoder and conditioning branches.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="Clip the global gradient norm when positive; zero disables clipping.",
    )
    parser.add_argument("--ssim-weight", type=float, default=0.7)
    parser.add_argument("--l1-weight", type=float, default=0.2)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument(
        "--target-l1-weight-strength",
        type=float,
        default=0.0,
        help=(
            "Multiply the L1 pixel term by 1 + strength * target and normalize "
            "weights per image/channel; SSIM and MSE remain unchanged."
        ),
    )
    parser.add_argument(
        "--marker-loss-weights",
        help=(
            "Comma-separated non-negative channel weights in --markers order; "
            "values are normalized to sum to one."
        ),
    )
    parser.add_argument(
        "--auxiliary-gradient-scope",
        choices=("all", "decoder-only"),
        default="all",
        help=(
            "Route every marker through the full model, or stop auxiliary-marker "
            "gradients at the shared UNeXt2 decoder input."
        ),
    )
    parser.add_argument(
        "--loss-profile",
        choices=("hybrid", "exact-score"),
        default="hybrid",
        help=(
            "hybrid uses SSIM/L1/MSE weights; exact-score directly optimizes "
            "the organizer's mean-image SSIM/PSNR formula."
        ),
    )
    parser.add_argument(
        "--ssim-mode",
        choices=("gaussian11", "uniform7", "gaussian11_valid"),
        default="gaussian11",
        help="SSIM loss implementation; uniform7 matches local scikit-image scoring.",
    )
    parser.add_argument("--pyramid-weight", type=float, default=0.0)
    parser.add_argument(
        "--source-photometric-augmentation",
        action="store_true",
        help="Apply fixed-range gamma/gain/offset augmentation to input DAPI only.",
    )
    parser.add_argument("--paired-affine-probability", type=float, default=0.0)
    parser.add_argument("--paired-affine-outer-halo", type=int, default=0)
    parser.add_argument("--paired-affine-degrees", type=float, default=0.0)
    parser.add_argument("--paired-affine-scale-min", type=float, default=1.0)
    parser.add_argument("--paired-affine-scale-max", type=float, default=1.0)
    parser.add_argument(
        "--d4-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for symmetric hybrid loss between the original prediction "
            "and one deterministically cycled non-identity D4 prediction."
        ),
    )
    parser.add_argument("--feature-guidance-checkpoint", type=Path)
    parser.add_argument("--feature-guidance-checkpoint-sha256")
    parser.add_argument("--feature-guidance-marker", choices=MARKERS)
    parser.add_argument("--feature-guidance-weight", type=float, default=0.0)
    parser.add_argument("--feature-guidance-mode", choices=("perceptual", "encoder-relational", "encoder-relational-robust"), default="perceptual")
    parser.add_argument("--feature-guidance-stage", type=int, default=4)
    parser.add_argument("--feature-guidance-crop-size", type=int, default=256)
    parser.add_argument("--feature-guidance-batch-samples", type=int, default=4)
    parser.add_argument(
        "--distillation-target-dir",
        type=Path,
        help="Directory of leakage-safe primary-marker teacher targets keyed by patch name.",
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument('--signal-auxiliary-weight',type=float,default=0.0)
    parser.add_argument('--distillation-marker', choices=MARKERS,
                        help='Explicit output marker for distillation; defaults to the first output for historical compatibility.')
    parser.add_argument("--primary-gradient-projection", action="store_true")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay in [0, 1); 0 disables EMA.",
    )
    parser.add_argument(
        "--mixstyle-probability",
        type=float,
        default=0.0,
        help="Probability of mixing shallow U-Net feature statistics per layer.",
    )
    parser.add_argument(
        "--mixstyle-alpha",
        type=float,
        default=0.1,
        help="Symmetric Beta concentration for MixStyle interpolation.",
    )
    parser.add_argument(
        "--selection-marker",
        choices=MARKERS,
        help="Select best.pt by this marker instead of the mean over all outputs.",
    )
    parser.add_argument(
        "--selection-psnr-aggregation",
        choices=("mean-image", "global-mse"),
        default="mean-image",
        help="Choose how validation PSNR is aggregated for best.pt selection.",
    )
    parser.add_argument(
        "--selection-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used by the scikit-image checkpoint-selection metric; 0 uses float output.",
    )
    parser.add_argument(
        "--selection-ssim-mode", choices=("uniform7", "gaussian11"),
        default="uniform7", help="Validation metric, independent of the training SSIM surrogate.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument(
        "--roi-weight-mode",
        choices=("none", "test-mean-rbf"),
        default="none",
        help="Optionally weight training ROI losses by proximity to test DAPI mean.",
    )
    parser.add_argument("--roi-weight-alpha", type=float, default=0.5)
    parser.add_argument("--roi-weight-sigma", type=float, default=1.5)
    parser.add_argument(
        "--activity-sampler",
        choices=("none", "hard-mix"),
        default="none",
        help="Optionally oversample the highest target-activity quartile late in training.",
    )
    parser.add_argument("--activity-sampler-start-ratio", type=float, default=0.75)
    parser.add_argument("--activity-sampler-hard-fraction", type=float, default=0.30)
    parser.add_argument("--activity-sampler-bins", type=int, default=4)
    add_robust_objective_arguments(parser)
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Keep compact per-epoch weights for final-pipeline checkpoint selection.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on every official training ROI without validation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.signal_auxiliary_weight) or args.signal_auxiliary_weight < 0:
        raise ValueError('Signal auxiliary weight must be finite and nonnegative')
    if (args.model_arch == 'unext2-fcmae-tiny-signal-aux') != (args.signal_auxiliary_weight > 0):
        raise ValueError('Signal auxiliary architecture and positive weight must be used together')
    if args.signal_auxiliary_weight > 0 and (args.loss_profile != 'hybrid' or
            args.feature_guidance_weight or args.distillation_weight or args.d4_consistency_weight or
            args.primary_gradient_projection or args.auxiliary_gradient_scope != 'all' or
            args.robust_objective != 'none' or args.roi_weight_mode != 'none'):
        raise ValueError('Signal auxiliary screen requires plain hybrid supervision only')

    if args.train_all and args.train_rois is not None:
        raise ValueError("--train-all and --train-rois are mutually exclusive.")
    if not 0.0 <= args.distillation_weight < 1.0:
        raise ValueError("--distillation-weight must be in [0, 1).")
    if (args.distillation_target_dir is None) != (args.distillation_weight == 0.0):
        raise ValueError(
            "Distillation target directory and positive weight must be configured together."
        )
    if args.distillation_target_dir is not None:
        args.distillation_target_dir = args.distillation_target_dir.resolve()
        if args.loss_profile != "hybrid":
            raise ValueError("Distillation currently requires hybrid loss.")
        if args.paired_affine_probability > 0.0:
            raise ValueError("Distillation does not support paired affine augmentation.")
    if (
        not np.isfinite(args.target_l1_weight_strength)
        or args.target_l1_weight_strength < 0
    ):
        raise ValueError("--target-l1-weight-strength must be finite and non-negative.")
    args.validation_epochs_resolved = parse_validation_epochs(
        args.validation_epochs,
        args.epochs,
    )
    args.train_target_source_offset_resolved = parse_source_offset(
        args.train_target_source_offset
    )
    args.validation_output_source_offset_resolved = parse_source_offset(
        args.validation_output_source_offset
    )
    prepare_output_dir(args.output_dir)
    set_seed(args.seed)
    markers = tuple(item.strip() for item in args.markers.split(",") if item.strip())
    if args.distillation_marker is not None and args.distillation_marker not in markers:
        raise ValueError('Distillation marker must be a trained output')
    args.distillation_marker_index = markers.index(args.distillation_marker) if args.distillation_marker else 0
    unknown_markers = sorted(set(markers) - set(MARKERS))
    if not markers or unknown_markers:
        raise ValueError(f"Invalid marker selection: {markers}; unknown={unknown_markers}")
    args.marker_loss_weights_resolved = parse_marker_loss_weights(
        args.marker_loss_weights, markers
    )
    validate_feature_guidance_options(args, markers)
    if args.primary_gradient_projection:
        if (args.model_arch != "unext2-fcmae-tiny" or markers != ("CD68", "CD45RO", "HLA-DR", "Vimentin")
                or args.marker_loss_weights_resolved != [0.85, 0.05, 0.05, 0.05]
                or args.loss_profile != "hybrid" or args.auxiliary_gradient_scope != "all"
                or args.d4_consistency_weight != 0 or args.feature_guidance_weight != 0
                or args.robust_objective != "none" or args.roi_weight_mode != "none"
                or args.activity_sampler != "none" or args.pyramid_weight != 0
                or args.target_l1_weight_strength != 0 or args.ema_decay != 0):
            raise ValueError("Primary gradient projection requires the fixed Tiny auxiliary15 hybrid recipe.")
    args.model_spec = model_specification(
        args.model_arch,
        base_channels=args.base_channels,
        block_type=args.block_type,
    )
    if args.selection_marker is not None and args.selection_marker not in markers:
        raise ValueError(
            f"Selection marker {args.selection_marker} is not among trained markers {markers}."
        )
    if not 0 <= args.selection_jpeg_quality <= 100:
        raise ValueError("--selection-jpeg-quality must be 0 or an integer from 1 to 100.")
    validate_context_halos(args.context_halo, args.train_target_context_halo)
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must be in [0, 1).")
    validate_mixstyle_options(
        args.mixstyle_probability,
        args.mixstyle_alpha,
        args.model_arch,
    )
    validate_d4_consistency_options(
        args.d4_consistency_weight,
        args.loss_profile,
    )
    if not np.isfinite(args.roi_weight_alpha) or args.roi_weight_alpha < 0:
        raise ValueError("--roi-weight-alpha must be finite and non-negative.")
    if not np.isfinite(args.roi_weight_sigma) or args.roi_weight_sigma <= 0:
        raise ValueError("--roi-weight-sigma must be finite and positive.")
    if args.roi_weight_mode != "none" and args.loss_profile != "hybrid":
        raise ValueError("ROI weighting currently requires --loss-profile=hybrid.")
    if args.marker_loss_weights_resolved is not None and args.loss_profile != "hybrid":
        raise ValueError("Marker loss weights currently require --loss-profile=hybrid.")
    if args.auxiliary_gradient_scope == "decoder-only":
        if args.model_arch != "unext2-fcmae-tiny":
            raise ValueError(
                "Decoder-only auxiliary gradients require unext2-fcmae-tiny."
            )
        if len(markers) < 2 or args.marker_loss_weights_resolved is None:
            raise ValueError(
                "Decoder-only auxiliary gradients require weighted multi-marker training."
            )
        if args.selection_marker != markers[0]:
            raise ValueError(
                "Decoder-only auxiliary gradients require the selection marker first."
            )
        if args.d4_consistency_weight > 0:
            raise ValueError(
                "Decoder-only auxiliary gradients do not support D4 consistency."
            )
    validate_robust_objective_options(args)
    validate_activity_sampler_options(args)
    if (
        args.model_arch
        not in {
            "unet",
            "dinov2-small-unet",
            "unext2-fcmae-tiny-dinov2-residual",
            "unext2-fcmae-tiny-dinov2-fusion",
            "unext2-fcmae-tiny",
            "unext2-fcmae-base",
            "unext2-fcmae-tiny-widehead",
            "unext2-fcmae-tiny-marker-decoders",
            "unext2-fcmae-tiny-cross-marker-feedback",
            "unext2-fcmae-tiny-dapi-detail-residual",
            "unext2-fcmae-tiny-shallow-dapi-skip",
            "unext2-fcmae-tiny-local-condition",
            "unext2-fcmae-tiny-signal-aux",
        }
        and args.condition_mode != "none"
    ):
        raise ValueError(
            "ROI conditioning is supported only by U-Net, UNeXt2 and DINOv2."
        )
    if args.model_arch == "dinov2-small-unet" and args.condition_mode not in {
        "none", SPATIAL_THUMBNAIL_CONDITION_MODE
    }:
        raise ValueError("DINOv2 requires none or spatial-thumbnail-film conditioning.")
    if (
        args.model_arch
        in {
            "unext2-fcmae-tiny",
            "unext2-fcmae-base",
            "unext2-fcmae-tiny-widehead",
            "unext2-fcmae-tiny-marker-decoders",
            "unext2-fcmae-tiny-cross-marker-feedback",
            "unext2-fcmae-tiny-dapi-detail-residual",
            "unext2-fcmae-tiny-shallow-dapi-skip",
            "unext2-fcmae-tiny-local-condition",
            "unext2-fcmae-tiny-signal-aux",
        }
        and args.condition_mode
        not in {
            "none",
            SPATIAL_THUMBNAIL_CONDITION_MODE,
            SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
            SPATIAL_WHOLE_ROI_CONDITION_MODE,
        }
    ):
        raise ValueError(
            "UNeXt2 received an unsupported conditioning mode."
        )
    if args.init_marker_subset and args.init_checkpoint is None:
        raise ValueError("--init-marker-subset requires --init-checkpoint.")
    if args.init_add_conditioning and args.init_checkpoint is None:
        raise ValueError("--init-add-conditioning requires --init-checkpoint.")
    if args.init_add_conditioning and args.init_marker_subset:
        raise ValueError(
            "--init-add-conditioning and --init-marker-subset are mutually exclusive."
        )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive.")
    if (
        not np.isfinite(args.minimum_learning_rate)
        or args.minimum_learning_rate < 0
        or args.minimum_learning_rate >= args.learning_rate
    ):
        raise ValueError(
            "--minimum-learning-rate must be finite, non-negative, and lower than "
            "--learning-rate."
        )
    if args.encoder_freeze_epochs < 0 or args.encoder_freeze_epochs >= args.epochs:
        raise ValueError("encoder-freeze-epochs must be nonnegative and smaller than epochs.")
    if args.encoder_freeze_epochs and args.model_arch != "unext2-fcmae-tiny":
        raise ValueError("Encoder freeze currently supports unext2-fcmae-tiny only.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if not np.isfinite(args.gradient_clip_norm) or args.gradient_clip_norm < 0:
        raise ValueError("--gradient-clip-norm must be finite and non-negative.")
    if args.encoder_learning_rate is not None and (
        not np.isfinite(args.encoder_learning_rate)
        or args.encoder_learning_rate <= 0
    ):
        raise ValueError("--encoder-learning-rate must be finite and positive.")
    if (
        args.encoder_learning_rate is not None
        and args.model_arch
        not in {
            "convnext-tiny-unet",
            "dinov2-small-unet",
            "unext2-fcmae-tiny-dinov2-residual",
            "unext2-fcmae-tiny-dinov2-fusion",
            "unext2-fcmae-tiny",
            "unext2-fcmae-base",
            "unext2-fcmae-tiny-widehead",
            "unext2-fcmae-tiny-marker-decoders",
            "unext2-fcmae-tiny-cross-marker-feedback",
            "unext2-fcmae-tiny-dapi-detail-residual",
            "unext2-fcmae-tiny-shallow-dapi-skip",
            "unext2-fcmae-tiny-local-condition",
            "unext2-fcmae-tiny-signal-aux",
        }
    ):
        raise ValueError(
            "--encoder-learning-rate requires ConvNeXt U-Net or UNeXt2."
        )
    if not np.isfinite(args.encoder_drop_path_rate) or not (
        0.0 <= args.encoder_drop_path_rate < 1.0
    ):
        raise ValueError("--encoder-drop-path-rate must be finite and in [0, 1).")
    if args.encoder_drop_path_rate > 0 and args.model_arch not in {
        "unext2-fcmae-tiny-dinov2-fusion",
        "unext2-fcmae-tiny",
        "unext2-fcmae-base",
        "unext2-fcmae-tiny-widehead",
        "unext2-fcmae-tiny-marker-decoders",
        "unext2-fcmae-tiny-cross-marker-feedback",
        "unext2-fcmae-tiny-dapi-detail-residual",
        "unext2-fcmae-tiny-shallow-dapi-skip",
        "unext2-fcmae-tiny-local-condition",
        "unext2-fcmae-tiny-signal-aux",
    }:
        raise ValueError("--encoder-drop-path-rate requires UNeXt2.")
    if args.encoder_init == "imagenet1k-v1" and args.model_arch != "convnext-tiny-unet":
        raise ValueError("--encoder-init is supported only by convnext-tiny-unet.")
    if args.encoder_init == "dinov2" and args.model_arch != "dinov2-small-unet":
        raise ValueError("--encoder-init=dinov2 requires dinov2-small-unet.")
    if args.encoder_init != "none" and args.init_checkpoint is not None:
        raise ValueError(
            "--encoder-init is only valid for a fresh stage1 run and cannot be "
            "combined with --init-checkpoint."
        )
    if args.fcmae_encoder_init_checkpoint is not None:
        if args.model_arch not in {
            "unext2-fcmae-tiny",
            "unext2-fcmae-tiny-dinov2-residual",
            "unext2-fcmae-tiny-dinov2-fusion",
            "unext2-fcmae-base",
            "unext2-fcmae-tiny-widehead",
            "unext2-fcmae-tiny-marker-decoders",
            "unext2-fcmae-tiny-cross-marker-feedback",
            "unext2-fcmae-tiny-dapi-detail-residual",
            "unext2-fcmae-tiny-shallow-dapi-skip",
            "unext2-fcmae-tiny-local-condition",
            "unext2-fcmae-tiny-signal-aux",
        }:
            raise ValueError(
                "--fcmae-encoder-init-checkpoint requires a UNeXt2 architecture."
            )
        if args.init_checkpoint is not None or args.encoder_init != "none":
            raise ValueError(
                "FCMAE encoder initialization is only valid for a fresh stage1 run."
            )
    cytoland_arguments = (
        args.cytoland_init_checkpoint,
        args.cytoland_init_mode,
        args.cytoland_init_sha256,
        args.cytoland_init_scope,
    )
    if any(value is not None for value in cytoland_arguments) and not all(
        value is not None for value in cytoland_arguments
    ):
        raise ValueError(
            "Cytoland initialization requires checkpoint, mode, SHA-256, and scope."
        )
    if args.cytoland_init_checkpoint is not None:
        if args.model_arch != "unext2-fcmae-tiny":
            raise ValueError("Cytoland initialization requires narrow Tiny UNeXt2.")
        if (
            args.init_checkpoint is not None
            or args.encoder_init != "none"
            or args.fcmae_encoder_init_checkpoint is not None
        ):
            raise ValueError(
                "Cytoland initialization is only valid for a fresh stage1 run."
            )
    data_root = resolve_data_root(args.dataset)
    names = image_names(data_root / "train" / "DAPI")
    val_rois = {item.strip().upper() for item in args.val_rois.split(",") if item.strip()}
    train_rois = (
        {
            item.strip().upper()
            for item in args.train_rois.split(",")
            if item.strip()
        }
        if args.train_rois is not None
        else None
    )
    if args.train_all:
        train_names = names
        val_names: list[str] = []
    else:
        train_names, val_names = split_names_by_roi(names, val_rois, train_rois)
    args.train_rois_resolved = sorted(
        {parse_patch_name(name).roi for name in train_names}
    )
    full_train_names = list(train_names)
    condition_by_roi: Mapping[str, float | Sequence[float] | np.ndarray] | None = None
    if args.condition_mode == "roi-mean-film":
        condition_by_roi, args.conditioning_spec = fit_roi_mean_conditioning(
            data_root / "train" / "DAPI",
            names,
            full_train_names,
        )
    elif args.condition_mode == SPATIAL_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = build_spatial_stats_conditioning(
            data_root / "train" / "DAPI",
            names,
        )
    elif args.condition_mode == SPATIAL_THUMBNAIL_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = build_spatial_thumbnail_conditioning(
            data_root / "train" / "DAPI",
            names,
        )
    elif args.condition_mode == SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = (
            build_spatial_thumbnail_texture_conditioning(
                data_root / "train" / "DAPI",
                names,
            )
        )
    elif args.condition_mode == SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = (
            build_spatial_thumbnail_multiscale_conditioning(
                data_root / "train" / "DAPI",
                names,
            )
        )
    elif args.condition_mode == SPATIAL_THUMBNAIL_RING_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = (
            build_spatial_thumbnail_ring_conditioning(
                data_root / "train" / "DAPI",
                names,
            )
        )
    elif args.condition_mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = (
            build_spatial_thumbnail_aligned_conditioning(
                data_root / "train" / "DAPI",
                names,
            )
        )
    elif args.condition_mode == SPATIAL_WHOLE_ROI_CONDITION_MODE:
        condition_by_roi, args.conditioning_spec = (
            build_spatial_whole_roi_conditioning(
                data_root / "train" / "DAPI",
                names,
            )
        )
    else:
        args.conditioning_spec = {"version": 1, "mode": "none"}
    train_names = limit_evenly(train_names, args.max_train_images)
    if args.distillation_target_dir is not None:
        validate_distillation_target_coverage(args.distillation_target_dir, train_names)
    val_names = limit_evenly(val_names, args.max_val_images)
    if not train_names or (not args.train_all and not val_names):
        raise ValueError("Requested training/validation split is empty.")

    if args.robust_objective == "group-dro":
        validate_training_group_coverage(full_train_names, train_names)
    group_dro_state = initialize_group_dro(
        args.robust_objective,
        train_names,
        args.group_dro_eta,
    )

    roi_weights: dict[str, float] | None = None
    roi_weight_report: dict[str, object] = {"mode": "none"}
    if args.roi_weight_mode == "test-mean-rbf":
        test_names = image_names(data_root / "test" / "DAPI")
        train_roi_means, _ = roi_intensity_means(
            data_root / "train" / "DAPI",
            train_names,
        )
        test_roi_means, test_center = roi_intensity_means(
            data_root / "test" / "DAPI",
            test_names,
        )
        roi_weights = rbf_roi_weights(
            train_names,
            train_roi_means,
            test_center,
            args.roi_weight_alpha,
            args.roi_weight_sigma,
        )
        roi_weight_report = {
            "mode": args.roi_weight_mode,
            "target_center": test_center,
            "alpha": args.roi_weight_alpha,
            "sigma": args.roi_weight_sigma,
            "test_roi_means": test_roi_means,
            "train_roi_means": train_roi_means,
            "train_roi_weights": roi_weights,
            "tile_weight_mean": float(
                np.mean(
                    [roi_weights[parse_patch_name(name).roi] for name in train_names]
                )
            ),
        }
    args.roi_weighting = roi_weight_report

    activity_sampler = None
    activity_sampling_report: dict[str, object] = {"mode": "none"}
    if args.activity_sampler == "hard-mix":
        activity_marker = str(args.selection_marker or markers[0])
        activities = training_target_activities(
            data_root,
            train_names,
            activity_marker,
        )
        start_epoch = int(
            math.ceil(args.epochs * args.activity_sampler_start_ratio)
        )
        activity_sampler = ActivityHardMixSampler(
            activities,
            seed=args.seed,
            num_bins=args.activity_sampler_bins,
            start_epoch=start_epoch,
            hard_fraction=args.activity_sampler_hard_fraction,
        )
        activity_sampling_report = {
            "mode": args.activity_sampler,
            "marker": activity_marker,
            "num_bins": args.activity_sampler_bins,
            "start_ratio": args.activity_sampler_start_ratio,
            "start_epoch_zero_based": start_epoch,
            "hard_fraction": args.activity_sampler_hard_fraction,
            "training_images": len(train_names),
            "hard_images": len(activity_sampler.hard_indices),
            "quantile_boundaries": list(activity_sampler.quantile_boundaries),
            "activity_min": min(activities),
            "activity_max": max(activities),
            "activity_mean": float(np.mean(activities)),
        }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    train_loader = DataLoader(
        PairedPatchDataset(
            data_root,
            train_names,
            markers=markers,
            augment=True,
            context_halo=args.context_halo,
            target_context_halo=args.train_target_context_halo,
            target_source_offset=args.train_target_source_offset_resolved,
            source_photometric_augmentation=args.source_photometric_augmentation,
            paired_affine_probability=args.paired_affine_probability,
            paired_affine_outer_halo=args.paired_affine_outer_halo,
            paired_affine_degrees=args.paired_affine_degrees,
            paired_affine_scale_min=args.paired_affine_scale_min,
            paired_affine_scale_max=args.paired_affine_scale_max,
            distillation_target_dir=args.distillation_target_dir,
            return_augmentation=(
                args.condition_mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE
            ),
        ),
        batch_size=args.batch_size,
        shuffle=activity_sampler is None,
        sampler=activity_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=seeded_data_generator(args.seed),
    )
    group_risk_loader = None
    if group_dro_state is not None:
        group_risk_loader = build_group_risk_loader(
            data_root=data_root,
            training_names=train_names,
            markers=markers,
            context_halo=args.context_halo,
            target_context_halo=args.train_target_context_halo,
            target_source_offset=args.train_target_source_offset_resolved,
            batch_size=args.batch_size,
            workers=args.workers,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
    val_loader = None
    if val_names:
        val_loader = DataLoader(
            PairedPatchDataset(
                data_root,
                val_names,
                markers=markers,
                augment=False,
                context_halo=args.context_halo,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

    model = build_model(
        architecture=args.model_arch,
        out_channels=len(markers),
        base_channels=args.base_channels,
        block_type=args.block_type,
        condition_mode=args.condition_mode,
    ).to(device)
    args.init_checkpoint_report = None
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        args.init_checkpoint_report = initialize_model_from_checkpoint(
            model,
            initial,
            markers,
            args.model_spec,
            args.conditioning_spec,
            marker_subset=args.init_marker_subset,
            add_conditioning=args.init_add_conditioning,
        )
    elif args.encoder_init == "imagenet1k-v1":
        args.init_checkpoint_report = (
            initialize_convnext_tiny_encoder_imagenet1k_v1(model)
        )
    elif args.encoder_init == "dinov2":
        args.init_checkpoint_report = initialize_dinov2_encoder(model, DINOV2_WEIGHT_PATH)
    elif args.fcmae_encoder_init_checkpoint is not None:
        fcmae_checkpoint_path = args.fcmae_encoder_init_checkpoint.resolve()
        args.init_checkpoint_report = initialize_unext2_encoder_from_fcmae(
            model,
            fcmae_checkpoint_path,
            device,
        )
        args.fcmae_encoder_init_checkpoint = str(fcmae_checkpoint_path)
    elif args.cytoland_init_checkpoint is not None:
        cytoland_checkpoint_path = args.cytoland_init_checkpoint.resolve()
        args.init_checkpoint_report = initialize_unext2_from_cytoland_vscyto2d(
            model,
            cytoland_checkpoint_path,
            str(args.cytoland_init_sha256),
            str(args.cytoland_init_mode),
            str(args.cytoland_init_scope),
        )
        args.cytoland_init_checkpoint = str(cytoland_checkpoint_path)
    feature_guidance = None
    args.feature_guidance_report = None
    args.feature_guidance_marker_index = None
    if args.feature_guidance_checkpoint is not None:
        feature_checkpoint_path = args.feature_guidance_checkpoint.resolve()
        teacher, args.feature_guidance_report = (
            load_frozen_multimarker_fcmae_teacher(
                feature_checkpoint_path,
                str(args.feature_guidance_checkpoint_sha256),
                val_rois,
                device,
            )
        )
        args.feature_guidance_report["objective"] = args.feature_guidance_mode
        feature_guidance = build_feature_guidance(
            teacher,
            args.feature_guidance_mode,
            stage=args.feature_guidance_stage,
            crop_size=args.feature_guidance_crop_size,
            batch_samples=args.feature_guidance_batch_samples,
        )
        args.feature_guidance_checkpoint = str(feature_checkpoint_path)
        args.feature_guidance_marker_index = markers.index(
            str(args.feature_guidance_marker)
        )
    if args.encoder_drop_path_rate > 0:
        if not isinstance(model, UNeXt2VirtualStainer):
            raise TypeError("Encoder drop path requires UNeXt2VirtualStainer.")
        args.encoder_drop_path_rates = model.configure_encoder_drop_path(
            args.encoder_drop_path_rate
        )
    else:
        args.encoder_drop_path_rates = []
    args.dinov2_residual_initialization_report = getattr(
        getattr(model, "dinov2_residual", None), "initialization_report", None
    )
    if isinstance(model, UNeXt2DinoBottleneckFusionVirtualStainer):
        args.dinov2_fusion_initialization_report = (
            model.dinov2_fusion.initialize_public_encoder()
            if args.init_checkpoint is None else {"mode": "restored-with-full-checkpoint"}
        )
    if args.mixstyle_probability > 0.0:
        if not isinstance(model, UNet):
            raise TypeError("MixStyle configuration requires a U-Net model.")
        model.configure_mixstyle(
            args.mixstyle_probability,
            args.mixstyle_alpha,
        )
    if args.loss_profile == "exact-score":
        if args.pyramid_weight > 0:
            raise ValueError(
                "--pyramid-weight must be 0 when --loss-profile=exact-score."
            )
        criterion = ExactCompetitionLoss(ssim_mode=args.ssim_mode)
    else:
        criterion = CompetitionLoss(
            ssim_weight=args.ssim_weight,
            l1_weight=args.l1_weight,
            mse_weight=args.mse_weight,
            pyramid_weight=args.pyramid_weight,
            target_l1_weight_strength=args.target_l1_weight_strength,
            ssim_mode=args.ssim_mode,
            channel_weights=args.marker_loss_weights_resolved,
        )
    distillation_criterion = (
        CompetitionLoss(
            ssim_weight=args.ssim_weight,
            l1_weight=args.l1_weight,
            mse_weight=args.mse_weight,
            ssim_mode=args.ssim_mode,
        )
        if args.distillation_target_dir is not None
        else None
    )
    optimizer, args.optimizer_parameter_groups = build_optimizer(
        model,
        args.model_arch,
        args.learning_rate,
        args.encoder_learning_rate,
        args.weight_decay,
        args.encoder_stage_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    projector = PrimaryGradientProjection(model.parameters()) if args.primary_gradient_projection else None
    primary_criterion = (CompetitionLoss(ssim_weight=args.ssim_weight, l1_weight=args.l1_weight,
                         mse_weight=args.mse_weight, ssim_mode=args.ssim_mode)
                         if projector is not None else None)
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        ema_model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.distillation_target_dir is not None:
        args.distillation_target_dir = str(args.distillation_target_dir)

    config = {
        **vars(args),
        "dataset": str(data_root),
        "output_dir": str(args.output_dir),
        "device_resolved": str(device),
        "amp": use_amp,
        "markers": markers,
        "train_images": len(train_names),
        "val_images": len(val_names),
        "parameters": parameter_count,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "runtime": runtime_metadata(),
        "init_checkpoint": (
            str(args.init_checkpoint.resolve()) if args.init_checkpoint is not None else None
        ),
    }
    if group_dro_state is not None:
        config["group_dro"] = group_dro_state
    config["activity_sampling"] = activity_sampling_report
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    best_score: float | None = None
    alternate_best_scores = {
        "mean-image": -float("inf"),
        "global-mse": -float("inf"),
    }
    log_path = args.output_dir / "metrics.jsonl"
    validation_epochs = set(args.validation_epochs_resolved)
    global_batch_index = 0
    group_p: dict[str, float] | None = None
    group_q: dict[str, float] | None = None
    if group_dro_state is not None:
        group_p = {
            str(group): float(probability)
            for group, probability in group_dro_state["p"].items()
        }
        group_q = {
            str(group): float(probability)
            for group, probability in group_dro_state["initial_q"].items()
        }
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        encoder_frozen = set_encoder_freeze_for_epoch(model, epoch, args.encoder_freeze_epochs)
        loss_sum = 0.0
        supervised_loss_sum = 0.0
        consistency_loss_sum = 0.0
        feature_guidance_loss_sum = 0.0
        signal_auxiliary_loss_sum = 0.0
        sample_count = 0
        epoch_roi_weights = roi_weights
        if group_q is not None:
            if group_p is None:
                raise RuntimeError("GroupDRO q exists without empirical p.")
            epoch_roi_weights = group_dro_q_over_p(group_q, group_p)
        optimizer_steps = 0
        projection_stats = {"finite_updates": 0, "conflicts": 0, "nonfinite_skips": 0,
                            "coefficient_sum": 0.0, "maximum_coefficient": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            distillation_target = None
            if len(batch) == 3:
                source, target, batch_names = batch
                augmentation = None
            elif len(batch) == 4:
                source, target, auxiliary_target, batch_names = batch
                if auxiliary_target.ndim == 4:
                    distillation_target = auxiliary_target
                    augmentation = None
                else:
                    augmentation = auxiliary_target
            else:
                raise RuntimeError(f"Unexpected training batch length: {len(batch)}.")
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if distillation_target is not None:
                distillation_target = distillation_target.to(
                    device, non_blocking=True
                )
            condition = batch_condition_tensor(batch_names, condition_by_roi, device)
            if augmentation is not None:
                if args.condition_mode != SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE:
                    raise RuntimeError(
                        "Only aligned thumbnail conditioning may return augmentation "
                        "metadata."
                    )
                if condition is None:
                    raise RuntimeError(
                        "Aligned thumbnail augmentation requires a condition tensor."
                    )
                condition = aligned_condition_from_augmentation(
                    condition,
                    augmentation,
                )
            sample_weights = None
            if epoch_roi_weights is not None:
                sample_weights = torch.tensor(
                    [
                        epoch_roi_weights[parse_patch_name(name).roi]
                        for name in batch_names
                    ],
                    device=device,
                    dtype=torch.float32,
                )
            transform_index = (
                d4_consistency_transform_index(global_batch_index)
                if args.d4_consistency_weight > 0
                else None
            )
            primary_losses = [] if projector is not None else None
            signal_losses = []
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                (
                    loss,
                    supervised_loss,
                    consistency_loss,
                    feature_guidance_loss,
                ) = training_batch_losses(
                    model,
                    source,
                    target,
                    condition,
                    criterion,
                    args.d4_consistency_weight,
                    transform_index=transform_index,
                    sample_weights=sample_weights,
                    auxiliary_gradient_scope=args.auxiliary_gradient_scope,
                    feature_guidance=feature_guidance,
                    feature_guidance_weight=args.feature_guidance_weight,
                    feature_guidance_marker_index=(
                        args.feature_guidance_marker_index or 0
                    ),
                    primary_criterion=primary_criterion,
                    primary_loss_output=primary_losses,
                    distillation_target=distillation_target,
                    distillation_criterion=distillation_criterion,
                    distillation_weight=args.distillation_weight,
                    distillation_marker_index=args.distillation_marker_index,
                    signal_auxiliary_weight=args.signal_auxiliary_weight,
                    signal_auxiliary_marker_index=markers.index(args.selection_marker) if args.selection_marker else 0,
                    signal_auxiliary_output=signal_losses,
                )
            sample_scale = gradient_accumulation_sample_scale(
                batch_index,
                args.batch_size,
                source.shape[0],
                args.gradient_accumulation_steps,
                len(train_loader.dataset),
            )
            scaled_loss = loss * sample_scale
            if projector is not None:
                projector.accumulate(primary_losses[0] * sample_scale)
            scaler.scale(scaled_loss).backward()
            group_complete = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if group_complete:
                if projector is not None:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    projected = projector.project()
                    if not projected["finite"]:
                        if not use_amp:
                            raise FloatingPointError("Non-finite projection gradients.")
                        projection_stats["nonfinite_skips"] += 1
                    else:
                        projection_stats["finite_updates"] += 1
                        projection_stats["conflicts"] += int(projected["conflict"])
                        projection_stats["coefficient_sum"] += projected["coefficient"]
                        projection_stats["maximum_coefficient"] = max(projection_stats["maximum_coefficient"], projected["coefficient"])
                if args.gradient_clip_norm > 0:
                    if not use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if projector is not None:
                    projector.reset()
                optimizer_steps += 1
                if ema_model is not None:
                    update_ema(ema_model, model, args.ema_decay)
            loss_sum += float(loss.detach()) * source.shape[0]
            supervised_loss_sum += float(supervised_loss.detach()) * source.shape[0]
            consistency_loss_sum += float(consistency_loss.detach()) * source.shape[0]
            feature_guidance_loss_sum += (
                float(feature_guidance_loss.detach()) * source.shape[0]
            )
            sample_count += source.shape[0]
            if signal_losses:
                signal_auxiliary_loss_sum += float(signal_losses[0]) * source.shape[0]
            global_batch_index += 1

        group_dro_metrics = None
        if group_q is not None:
            if group_p is None or group_risk_loader is None:
                raise RuntimeError("GroupDRO state is missing p or its risk loader.")
            group_risks, group_risk_counts = evaluate_group_risks(
                model,
                group_risk_loader,
                criterion,
                device,
                group_dro_state["counts"],
                condition_by_roi,
            )
            group_dro_metrics, group_q = group_dro_epoch_report(
                group_p,
                group_q,
                group_risks,
                group_risk_counts,
                args.group_dro_eta,
            )
            group_dro_metrics["risk_pass"] = group_dro_state["risk_pass"]

        evaluation_model = ema_model if ema_model is not None else model
        metrics = (
            validate(
                evaluation_model,
                val_loader,
                device,
                markers,
                args.selection_jpeg_quality,
                condition_by_roi,
                args.validation_output_source_offset_resolved,
                args.selection_ssim_mode,
            )
            if val_loader and epoch in validation_epochs
            else None
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            "encoder_frozen": encoder_frozen,
            "train_supervised_loss": supervised_loss_sum / sample_count,
            "train_d4_consistency_loss": consistency_loss_sum / sample_count,
            "train_feature_guidance_loss": feature_guidance_loss_sum / sample_count,
            "train_signal_auxiliary_loss": signal_auxiliary_loss_sum / sample_count,
            "learning_rate": scheduler.get_last_lr()[0],
            "learning_rates": {
                str(group.get("group_name", index)): learning_rate
                for index, (group, learning_rate) in enumerate(
                    zip(optimizer.param_groups, scheduler.get_last_lr())
                )
            },
            "optimizer_steps": optimizer_steps,
            "seconds": time.perf_counter() - started,
            "validation": metrics,
        }
        if group_dro_metrics is not None:
            row["group_dro"] = group_dro_metrics
        if projector is not None:
            row["primary_gradient_projection"] = projection_stats
        if metrics is not None:
            score_key = (
                "competition_score_mean_image_psnr"
                if args.selection_psnr_aggregation == "mean-image"
                else "competition_score_global_mse_psnr"
            )
            mean_score_key = f"mean_{score_key}"
            row["selection_score"] = (
                metrics["markers"][args.selection_marker][score_key]
                if args.selection_marker is not None
                else metrics[mean_score_key]
            )
            row["selection_metric"] = score_key
            row["selection_scores"] = {
                "mean-image": (
                    metrics["markers"][args.selection_marker][
                        "competition_score_mean_image_psnr"
                    ]
                    if args.selection_marker is not None
                    else metrics["mean_competition_score_mean_image_psnr"]
                ),
                "global-mse": (
                    metrics["markers"][args.selection_marker][
                        "competition_score_global_mse_psnr"
                    ]
                    if args.selection_marker is not None
                    else metrics["mean_competition_score_global_mse_psnr"]
                ),
            }
        if device.type == "cuda":
            row["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if args.save_every_epoch:
            save_model_checkpoint(
                args.output_dir / "epochs" / f"epoch_{epoch:03d}.pt",
                evaluation_model,
                epoch,
                row.get("selection_score"),
                args,
                markers,
                args.selection_psnr_aggregation,
            )

        if metrics is not None:
            for aggregation, score in row["selection_scores"].items():
                if score > alternate_best_scores[aggregation]:
                    alternate_best_scores[aggregation] = score
                    filename = f"best_{aggregation.replace('-', '_')}.pt"
                    save_model_checkpoint(
                        args.output_dir / filename,
                        evaluation_model,
                        epoch,
                        score,
                        args,
                        markers,
                        aggregation,
                    )

        if metrics is not None and (
            best_score is None or row["selection_score"] > best_score
        ):
            best_score = row["selection_score"]
            save_checkpoint(
                args.output_dir / "best.pt",
                evaluation_model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
                markers,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            evaluation_model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            args,
            markers,
        )

    if args.train_all:
        save_checkpoint(
            args.output_dir / "final.pt",
            ema_model if ema_model is not None else model,
            optimizer,
            scheduler,
            args.epochs,
            best_score,
            args,
            markers,
        )


if __name__ == "__main__":
    main()
