from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


MARKERS = ("CD45RO", "CD68", "HLA-DR", "Vimentin")
PATCH_RE = re.compile(r"^(ROI\d+)_(\d+)_(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class PatchId:
    roi: str
    row: int
    col: int


def parse_patch_name(name: str) -> PatchId:
    match = PATCH_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected patch filename: {name}")
    return PatchId(match.group(1).upper(), int(match.group(2)), int(match.group(3)))


def resolve_data_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "train" / "DAPI").is_dir() and (path / "test" / "DAPI").is_dir():
        return path

    matches = sorted(
        candidate.parent
        for candidate in path.rglob("train")
        if (candidate / "DAPI").is_dir()
        and (candidate.parent / "test" / "DAPI").is_dir()
    )
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(
            f"Expected exactly one data root below {path}, found {len(unique)}: {unique}"
        )
    return unique[0]


def image_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.jpg"))


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def load_context_gray(directory: Path, name: str, halo: int = 0) -> np.ndarray:
    center = load_gray(directory / name)
    if halo == 0:
        return center
    if not 0 < halo <= center.shape[0] or center.shape[0] != center.shape[1]:
        raise ValueError(
            f"Context halo must be between 1 and the square patch size, got {halo} "
            f"for {center.shape}."
        )

    patch = parse_patch_name(name)
    size = center.shape[0]
    canvas = np.pad(center, ((halo, halo), (halo, halo)), mode="reflect")

    def slices(offset: int) -> tuple[slice, slice]:
        if offset == -1:
            return slice(0, halo), slice(size - halo, size)
        if offset == 0:
            return slice(halo, halo + size), slice(0, size)
        return slice(halo + size, halo + size + halo), slice(0, halo)

    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            row = patch.row + row_offset
            col = patch.col + col_offset
            if row < 0 or col < 0:
                continue
            neighbor_path = directory / f"{patch.roi}_{row:02d}_{col:02d}.jpg"
            if not neighbor_path.is_file():
                continue
            neighbor = load_gray(neighbor_path)
            destination_rows, source_rows = slices(row_offset)
            destination_cols, source_cols = slices(col_offset)
            canvas[destination_rows, destination_cols] = neighbor[
                source_rows, source_cols
            ]
    return canvas


def split_names_by_roi(
    names: Sequence[str],
    val_rois: set[str],
    train_rois: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    available_rois = {parse_patch_name(name).roi for name in names}
    unknown_val_rois = sorted(val_rois - available_rois)
    if unknown_val_rois:
        raise ValueError(f"Unknown validation ROIs: {unknown_val_rois}.")
    if train_rois is not None:
        if not train_rois:
            raise ValueError("Explicit training ROIs cannot be empty.")
        overlap = sorted(train_rois.intersection(val_rois))
        if overlap:
            raise ValueError(f"Training and validation ROIs overlap: {overlap}.")
        unknown_train_rois = sorted(train_rois - available_rois)
        if unknown_train_rois:
            raise ValueError(f"Unknown training ROIs: {unknown_train_rois}.")

    train = [
        name
        for name in names
        if (
            parse_patch_name(name).roi in train_rois
            if train_rois is not None
            else parse_patch_name(name).roi not in val_rois
        )
    ]
    val = [name for name in names if parse_patch_name(name).roi in val_rois]
    if not train or not val:
        raise ValueError("Both training and validation splits must contain images.")
    return train, val


def choose_evenly(names: Sequence[str], limit: int) -> list[str]:
    if limit <= 0 or len(names) <= limit:
        return list(names)
    indices = np.linspace(0, len(names) - 1, num=limit, dtype=int)
    return [names[index] for index in indices]


def roi_intensity_means(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, float], float]:
    if not names:
        raise ValueError("ROI intensity means require at least one image.")
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "pixels": 0.0}
    )
    total_sum = 0.0
    total_pixels = 0.0
    for name in names:
        image = load_gray(directory / name)
        roi = parse_patch_name(name).roi
        image_sum = float(image.sum())
        pixels = float(image.size)
        totals[roi]["sum"] += image_sum
        totals[roi]["pixels"] += pixels
        total_sum += image_sum
        total_pixels += pixels
    means = {
        roi: values["sum"] / values["pixels"]
        for roi, values in sorted(totals.items())
    }
    return means, total_sum / total_pixels


def fit_roi_mean_conditioning(
    directory: Path,
    names: Sequence[str],
    fit_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, object]]:
    if not fit_names:
        raise ValueError("ROI mean conditioning requires training images.")
    roi_means, _ = roi_intensity_means(directory, names)
    fit_rois = sorted({parse_patch_name(name).roi for name in fit_names})
    missing = sorted(set(fit_rois) - set(roi_means))
    if missing:
        raise ValueError(f"ROI conditioning means are missing for: {missing}")
    fit_values = np.asarray([roi_means[roi] for roi in fit_rois], dtype=np.float64)
    center = float(fit_values.mean())
    scale = float(fit_values.std())
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError(
            f"ROI conditioning statistics are invalid: center={center}, scale={scale}"
        )
    conditions = {roi: (value - center) / scale for roi, value in roi_means.items()}
    report: dict[str, object] = {
        "version": 1,
        "mode": "roi-mean-film",
        "feature": "dapi_roi_mean",
        "injection": "bottleneck",
        "normalization": {
            "fit_scope": "training_rois_only",
            "roi_weighting": "equal",
            "units": "uint8_0_255",
            "center": center,
            "scale": scale,
        },
        "train_rois": fit_rois,
        "train_roi_means": {roi: roi_means[roi] for roi in fit_rois},
    }
    return conditions, report


def apply_roi_mean_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, float]:
    if not isinstance(report, dict):
        raise ValueError("ROI mean conditioning metadata must be a dictionary.")
    try:
        normalization = report["normalization"]
        if not isinstance(normalization, dict):
            raise TypeError
        center = float(normalization["center"])
        scale = float(normalization["scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ROI mean conditioning metadata is incomplete.") from error
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError(
            f"ROI conditioning statistics are invalid: center={center}, scale={scale}"
        )
    roi_means, _ = roi_intensity_means(directory, names)
    return {roi: (value - center) / scale for roi, value in roi_means.items()}


def rbf_roi_weights(
    names: Sequence[str],
    roi_means: dict[str, float],
    target_center: float,
    alpha: float,
    sigma: float,
) -> dict[str, float]:
    if not names:
        raise ValueError("ROI weighting requires at least one training image.")
    if not np.isfinite(target_center):
        raise ValueError("ROI weighting target center must be finite.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ROI weighting alpha must be finite and non-negative.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("ROI weighting sigma must be finite and positive.")

    rois = sorted({parse_patch_name(name).roi for name in names})
    missing = sorted(set(rois) - set(roi_means))
    if missing:
        raise ValueError(f"ROI means are missing for: {missing}")
    non_finite = sorted(roi for roi in rois if not np.isfinite(roi_means[roi]))
    if non_finite:
        raise ValueError(f"ROI means must be finite for: {non_finite}")
    raw = {
        roi: 1.0
        + alpha
        * float(np.exp(-0.5 * ((roi_means[roi] - target_center) / sigma) ** 2))
        for roi in rois
    }
    normalizer = float(
        np.mean([raw[parse_patch_name(name).roi] for name in names])
    )
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError(f"ROI weight normalizer must be positive, got {normalizer}.")
    return {roi: value / normalizer for roi, value in raw.items()}


def linear_fit(inputs: Sequence[float], targets: Sequence[float]) -> tuple[float, float]:
    if len(inputs) != len(targets) or len(inputs) < 2:
        raise ValueError("Linear calibration requires at least two paired values.")
    design = np.column_stack((np.asarray(inputs), np.ones(len(inputs))))
    slope, intercept = np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]
    return float(slope), float(intercept)


def fit_roi_mean_calibration(
    data_root: Path,
    marker: str,
    names: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Fit target ROI mean from DAPI ROI mean using only paired training data."""
    selected_names = (
        list(names)
        if names is not None
        else image_names(data_root / "train" / "DAPI")
    )
    roi_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"source_sum": 0.0, "target_sum": 0.0, "pixels": 0.0}
    )
    for name in selected_names:
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / marker / name)
        roi = parse_patch_name(name).roi
        roi_totals[roi]["source_sum"] += float(source.sum())
        roi_totals[roi]["target_sum"] += float(target.sum())
        roi_totals[roi]["pixels"] += float(source.size)

    source_means = [values["source_sum"] / values["pixels"] for values in roi_totals.values()]
    target_means = [values["target_sum"] / values["pixels"] for values in roi_totals.values()]
    return linear_fit(source_means, target_means)
