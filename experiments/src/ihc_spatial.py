from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ihc_data import (
    PatchId,
    apply_roi_mean_conditioning,
    image_names,
    load_gray,
    parse_patch_name,
)


LOCAL_FEATURE_NAMES = (
    "mean",
    "std",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "mean_abs_gradient",
    "dark_fraction",
)
SPATIAL_CONDITION_MODE = "spatial-stats-film"
SPATIAL_CONDITION_RADIUS = 2
SPATIAL_FILM_HIDDEN_FEATURES = 256
SPATIAL_THUMBNAIL_CONDITION_MODE = "spatial-thumbnail-film"
SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE = "spatial-thumbnail-texture-film"
SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE = (
    "spatial-thumbnail-multiscale-film"
)
SPATIAL_THUMBNAIL_RING_CONDITION_MODE = "spatial-thumbnail-ring-film"
SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE = "spatial-thumbnail-aligned-fusion"
SPATIAL_WHOLE_ROI_CONDITION_MODE = "spatial-thumbnail-whole-roi-film"
SPATIAL_THUMBNAIL_TILE_SIDE = 16
SPATIAL_THUMBNAIL_GRID_SIDE = SPATIAL_CONDITION_RADIUS * 2 + 1
SPATIAL_THUMBNAIL_SIDE = SPATIAL_THUMBNAIL_TILE_SIDE * SPATIAL_THUMBNAIL_GRID_SIDE
SPATIAL_THUMBNAIL_CHANNELS = 2
SPATIAL_THUMBNAIL_TEXTURE_CHANNELS = 4
SPATIAL_THUMBNAIL_TEXTURE_SIDE = SPATIAL_THUMBNAIL_SIDE
SPATIAL_THUMBNAIL_EMBEDDING_FEATURES = 64
SPATIAL_THUMBNAIL_ALIGNED_FEATURES = 32
SPATIAL_THUMBNAIL_ALIGNED_DILATIONS = (1, 2, 4, 8, 16, 8, 4, 2, 1)
SPATIAL_WHOLE_ROI_SIDE = 32
SPATIAL_WHOLE_ROI_CHANNELS = 4
SPATIAL_WHOLE_ROI_EMBEDDING_FEATURES = 64


def local_intensity_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"Expected a non-empty grayscale image, got {image.shape}.")
    if not np.isfinite(image).all():
        raise ValueError("Image features require finite pixels.")

    vertical = np.abs(np.diff(image, axis=0)).mean() if image.shape[0] > 1 else 0.0
    horizontal = np.abs(np.diff(image, axis=1)).mean() if image.shape[1] > 1 else 0.0
    quantiles = np.quantile(image, (0.05, 0.25, 0.5, 0.75, 0.95))
    return np.asarray(
        (
            image.mean(),
            image.std(),
            *quantiles,
            0.5 * (vertical + horizontal),
            np.mean(image <= 5.0) * 255.0,
        ),
        dtype=np.float32,
    ) / 255.0


def downsample_tile_mean(
    image: np.ndarray,
    side: int = SPATIAL_THUMBNAIL_TILE_SIDE,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not image.size:
        raise ValueError(f"Expected a non-empty square grayscale tile, got {image.shape}.")
    if side <= 0 or image.shape[0] % side:
        raise ValueError(
            f"Thumbnail side {side} must evenly divide tile size {image.shape[0]}."
        )
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 255.0:
        raise ValueError("Thumbnail source pixels must be finite uint8-range values.")
    block = image.shape[0] // side
    return (
        image.reshape(side, block, side, block).mean(axis=(1, 3)).astype(np.float32)
        / 255.0
    )


def downsample_tile_texture(
    image: np.ndarray,
    side: int = SPATIAL_THUMBNAIL_TILE_SIDE,
) -> np.ndarray:
    """Return block mean, contrast, and gradient maps for one DAPI tile.

    The maps use the same non-overlapping grid as the existing thumbnail.  All
    channels are normalized to the uint8 range so the condition is fixed and
    independent of the training split.
    """

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not image.size:
        raise ValueError(f"Expected a non-empty square grayscale tile, got {image.shape}.")
    if side <= 0 or image.shape[0] % side:
        raise ValueError(
            f"Thumbnail side {side} must evenly divide tile size {image.shape[0]}."
        )
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 255.0:
        raise ValueError("Thumbnail source pixels must be finite uint8-range values.")

    block = image.shape[0] // side
    blocks = image.reshape(side, block, side, block)
    means = blocks.mean(axis=(1, 3))
    centered = blocks - means[:, None, :, None]
    standard_deviation = np.sqrt(np.mean(centered * centered, axis=(1, 3)))
    vertical = (
        np.abs(np.diff(blocks, axis=1)).mean(axis=(1, 3))
        if block > 1
        else np.zeros_like(means)
    )
    horizontal = (
        np.abs(np.diff(blocks, axis=3)).mean(axis=(1, 3))
        if block > 1
        else np.zeros_like(means)
    )
    gradient = 0.5 * (vertical + horizontal)
    return np.stack(
        (means, standard_deviation, gradient),
        axis=0,
    ).astype(np.float32) / 255.0


@dataclass(frozen=True)
class SpatialFeatureTable:
    values: np.ndarray
    names: tuple[str, ...]


def spatial_context_feature_names(
    radius: int = SPATIAL_CONDITION_RADIUS,
) -> tuple[str, ...]:
    if radius < 2:
        raise ValueError("Spatial context radius must be at least two tiles.")
    offsets = tuple(
        (row_offset, col_offset)
        for row_offset in range(-radius, radius + 1)
        for col_offset in range(-radius, radius + 1)
    )
    return (
        "row_position",
        "col_position",
        "distance_top",
        "distance_bottom",
        "distance_left",
        "distance_right",
        "roi_occupancy",
        "roi_mean_mean",
        "roi_mean_std",
        "roi_std_mean",
        "roi_std_std",
        "radius1_mean",
        "radius1_std",
        "radius1_presence",
        f"radius{radius}_mean",
        f"radius{radius}_std",
        f"radius{radius}_presence",
        *(
            f"grid_mean_r{row_offset:+d}_c{col_offset:+d}"
            for row_offset, col_offset in offsets
        ),
        *(
            f"grid_std_r{row_offset:+d}_c{col_offset:+d}"
            for row_offset, col_offset in offsets
        ),
        *(
            f"grid_present_r{row_offset:+d}_c{col_offset:+d}"
            for row_offset, col_offset in offsets
        ),
    )


SPATIAL_CONDITION_FEATURE_NAMES = (
    *LOCAL_FEATURE_NAMES,
    *spatial_context_feature_names(),
)
SPATIAL_CONDITION_DIMENSION = len(SPATIAL_CONDITION_FEATURE_NAMES)
SPATIAL_THUMBNAIL_PIXEL_DIMENSION = (
    SPATIAL_THUMBNAIL_CHANNELS * SPATIAL_THUMBNAIL_SIDE**2
)
SPATIAL_THUMBNAIL_CONDITION_DIMENSION = (
    SPATIAL_CONDITION_DIMENSION + SPATIAL_THUMBNAIL_PIXEL_DIMENSION
)
SPATIAL_THUMBNAIL_TEXTURE_PIXEL_DIMENSION = (
    SPATIAL_THUMBNAIL_TEXTURE_CHANNELS * SPATIAL_THUMBNAIL_SIDE**2
)
SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION = (
    SPATIAL_CONDITION_DIMENSION + SPATIAL_THUMBNAIL_TEXTURE_PIXEL_DIMENSION
)
SPATIAL_WHOLE_ROI_PIXEL_DIMENSION = (
    SPATIAL_WHOLE_ROI_CHANNELS * SPATIAL_WHOLE_ROI_SIDE**2
)
SPATIAL_WHOLE_ROI_CONDITION_DIMENSION = (
    SPATIAL_THUMBNAIL_CONDITION_DIMENSION + SPATIAL_WHOLE_ROI_PIXEL_DIMENSION
)


def build_spatial_context_features(
    patches: Sequence[PatchId],
    local_features: np.ndarray,
    radius: int = 2,
) -> SpatialFeatureTable:
    if radius < 2:
        raise ValueError("Spatial context radius must be at least two tiles.")
    local_features = np.asarray(local_features, dtype=np.float32)
    if local_features.shape != (len(patches), len(LOCAL_FEATURE_NAMES)):
        raise ValueError(
            "Local feature matrix must have shape "
            f"({len(patches)}, {len(LOCAL_FEATURE_NAMES)}), got {local_features.shape}."
        )
    if not np.isfinite(local_features).all():
        raise ValueError("Local feature matrix must be finite.")

    indices_by_key: dict[tuple[str, int, int], int] = {}
    indices_by_roi: dict[str, list[int]] = defaultdict(list)
    for index, patch in enumerate(patches):
        key = (patch.roi, patch.row, patch.col)
        if key in indices_by_key:
            raise ValueError(f"Duplicate patch coordinates: {key}.")
        indices_by_key[key] = index
        indices_by_roi[patch.roi].append(index)

    offsets = tuple(
        (row_offset, col_offset)
        for row_offset in range(-radius, radius + 1)
        for col_offset in range(-radius, radius + 1)
    )
    names = spatial_context_feature_names(radius)

    output = np.empty((len(patches), len(names)), dtype=np.float32)
    for roi, roi_indices in indices_by_roi.items():
        rows = np.asarray([patches[index].row for index in roi_indices], dtype=np.float32)
        cols = np.asarray([patches[index].col for index in roi_indices], dtype=np.float32)
        row_min, row_max = float(rows.min()), float(rows.max())
        col_min, col_max = float(cols.min()), float(cols.max())
        row_span = max(row_max - row_min, 1.0)
        col_span = max(col_max - col_min, 1.0)
        bounding_tiles = (row_max - row_min + 1.0) * (col_max - col_min + 1.0)
        roi_local = local_features[roi_indices]
        roi_summary = (
            float(roi_local[:, 0].mean()),
            float(roi_local[:, 0].std()),
            float(roi_local[:, 1].mean()),
            float(roi_local[:, 1].std()),
        )

        for index in roi_indices:
            patch = patches[index]
            grid_means: list[float] = []
            grid_stds: list[float] = []
            grid_presence: list[float] = []
            radius_values: dict[int, list[float]] = {1: [], radius: []}
            radius_counts: dict[int, int] = {1: 0, radius: 0}

            for row_offset, col_offset in offsets:
                neighbor_index = indices_by_key.get(
                    (roi, patch.row + row_offset, patch.col + col_offset)
                )
                present = neighbor_index is not None
                grid_presence.append(float(present))
                if present:
                    grid_means.append(float(local_features[neighbor_index, 0]))
                    grid_stds.append(float(local_features[neighbor_index, 1]))
                else:
                    grid_means.append(0.0)
                    grid_stds.append(0.0)

                distance = max(abs(row_offset), abs(col_offset))
                for neighborhood_radius in radius_values:
                    if distance <= neighborhood_radius:
                        radius_counts[neighborhood_radius] += 1
                        if present:
                            radius_values[neighborhood_radius].append(
                                float(local_features[neighbor_index, 0])
                            )

            neighborhood_summary: list[float] = []
            for neighborhood_radius in (1, radius):
                values = radius_values[neighborhood_radius]
                neighborhood_summary.extend(
                    (
                        float(np.mean(values)),
                        float(np.std(values)),
                        len(values) / radius_counts[neighborhood_radius],
                    )
                )

            output[index] = np.asarray(
                (
                    2.0 * (patch.row - row_min) / row_span - 1.0,
                    2.0 * (patch.col - col_min) / col_span - 1.0,
                    (patch.row - row_min) / row_span,
                    (row_max - patch.row) / row_span,
                    (patch.col - col_min) / col_span,
                    (col_max - patch.col) / col_span,
                    len(roi_indices) / bounding_tiles,
                    *roi_summary,
                    *neighborhood_summary,
                    *grid_means,
                    *grid_stds,
                    *grid_presence,
                ),
                dtype=np.float32,
            )

    return SpatialFeatureTable(output, tuple(names))


def build_neighborhood_thumbnails(
    patches: Sequence[PatchId],
    tile_thumbnails: np.ndarray,
    radius: int = SPATIAL_CONDITION_RADIUS,
) -> np.ndarray:
    if not patches:
        raise ValueError("Neighborhood thumbnails require at least one patch.")
    if radius != SPATIAL_CONDITION_RADIUS:
        raise ValueError(
            f"Neighborhood thumbnail radius must be {SPATIAL_CONDITION_RADIUS}."
        )
    tile_thumbnails = np.asarray(tile_thumbnails, dtype=np.float32)
    if (
        tile_thumbnails.ndim != 3
        or tile_thumbnails.shape[0] != len(patches)
        or tile_thumbnails.shape[1] != tile_thumbnails.shape[2]
    ):
        raise ValueError(
            "Tile thumbnails must have shape (patches, side, side), got "
            f"{tile_thumbnails.shape}."
        )
    if not np.isfinite(tile_thumbnails).all() or (
        tile_thumbnails.min() < 0.0 or tile_thumbnails.max() > 1.0
    ):
        raise ValueError("Tile thumbnails must contain finite normalized pixels.")

    indices_by_key: dict[tuple[str, int, int], int] = {}
    for index, patch in enumerate(patches):
        key = (patch.roi, patch.row, patch.col)
        if key in indices_by_key:
            raise ValueError(f"Duplicate patch coordinates: {key}.")
        indices_by_key[key] = index

    tile_side = tile_thumbnails.shape[1]
    grid_side = radius * 2 + 1
    output = np.zeros(
        (len(patches), SPATIAL_THUMBNAIL_CHANNELS, grid_side * tile_side, grid_side * tile_side),
        dtype=np.float32,
    )
    for index, patch in enumerate(patches):
        for grid_row, row_offset in enumerate(range(-radius, radius + 1)):
            row_slice = slice(grid_row * tile_side, (grid_row + 1) * tile_side)
            for grid_col, col_offset in enumerate(range(-radius, radius + 1)):
                neighbor_index = indices_by_key.get(
                    (patch.roi, patch.row + row_offset, patch.col + col_offset)
                )
                if neighbor_index is None:
                    continue
                col_slice = slice(grid_col * tile_side, (grid_col + 1) * tile_side)
                output[index, 0, row_slice, col_slice] = tile_thumbnails[neighbor_index]
                output[index, 1, row_slice, col_slice] = 1.0
    return output


def build_neighborhood_texture_maps(
    patches: Sequence[PatchId],
    tile_features: np.ndarray,
    radius: int = SPATIAL_CONDITION_RADIUS,
) -> np.ndarray:
    """Place fixed per-tile texture maps into a zero-padded ROI neighborhood."""

    if not patches:
        raise ValueError("Neighborhood texture maps require at least one patch.")
    if radius != SPATIAL_CONDITION_RADIUS:
        raise ValueError(
            f"Neighborhood texture radius must be {SPATIAL_CONDITION_RADIUS}."
        )
    tile_features = np.asarray(tile_features, dtype=np.float32)
    if (
        tile_features.ndim != 4
        or tile_features.shape[0] != len(patches)
        or tile_features.shape[1] != SPATIAL_THUMBNAIL_TEXTURE_CHANNELS - 1
        or tile_features.shape[2] != tile_features.shape[3]
    ):
        raise ValueError(
            "Tile texture maps must have shape "
            f"(patches, {SPATIAL_THUMBNAIL_TEXTURE_CHANNELS - 1}, side, side), "
            f"got {tile_features.shape}."
        )
    if not np.isfinite(tile_features).all() or (
        tile_features.min() < 0.0 or tile_features.max() > 1.0
    ):
        raise ValueError("Tile texture maps must contain finite normalized pixels.")

    indices_by_key: dict[tuple[str, int, int], int] = {}
    for index, patch in enumerate(patches):
        key = (patch.roi, patch.row, patch.col)
        if key in indices_by_key:
            raise ValueError(f"Duplicate patch coordinates: {key}.")
        indices_by_key[key] = index

    tile_side = tile_features.shape[2]
    grid_side = radius * 2 + 1
    output = np.zeros(
        (
            len(patches),
            SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
            grid_side * tile_side,
            grid_side * tile_side,
        ),
        dtype=np.float32,
    )
    for index, patch in enumerate(patches):
        for grid_row, row_offset in enumerate(range(-radius, radius + 1)):
            row_slice = slice(grid_row * tile_side, (grid_row + 1) * tile_side)
            for grid_col, col_offset in enumerate(range(-radius, radius + 1)):
                neighbor_index = indices_by_key.get(
                    (patch.roi, patch.row + row_offset, patch.col + col_offset)
                )
                if neighbor_index is None:
                    continue
                col_slice = slice(grid_col * tile_side, (grid_col + 1) * tile_side)
                output[index, :-1, row_slice, col_slice] = tile_features[neighbor_index]
                output[index, -1, row_slice, col_slice] = 1.0
    return output


def build_whole_roi_thumbnails(
    patches: Sequence[PatchId],
    local_features: np.ndarray,
    side: int = SPATIAL_WHOLE_ROI_SIDE,
) -> np.ndarray:
    if not patches:
        raise ValueError("Whole-ROI thumbnails require at least one patch.")
    if side <= 0:
        raise ValueError("Whole-ROI thumbnail side must be positive.")
    local_features = np.asarray(local_features, dtype=np.float32)
    expected_shape = (len(patches), len(LOCAL_FEATURE_NAMES))
    if local_features.shape != expected_shape:
        raise ValueError(
            f"Local feature matrix must have shape {expected_shape}, "
            f"got {local_features.shape}."
        )
    if not np.isfinite(local_features).all() or (
        local_features.min() < 0.0 or local_features.max() > 1.0
    ):
        raise ValueError("Whole-ROI local features must be finite and normalized.")

    indices_by_roi: dict[str, list[int]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for index, patch in enumerate(patches):
        key = (patch.roi, patch.row, patch.col)
        if key in seen:
            raise ValueError(f"Duplicate patch coordinates: {key}.")
        seen.add(key)
        indices_by_roi[patch.roi].append(index)

    output = np.zeros(
        (len(patches), SPATIAL_WHOLE_ROI_CHANNELS, side, side),
        dtype=np.float32,
    )
    for roi_indices in indices_by_roi.values():
        row_min = min(patches[index].row for index in roi_indices)
        row_max = max(patches[index].row for index in roi_indices)
        col_min = min(patches[index].col for index in roi_indices)
        col_max = max(patches[index].col for index in roi_indices)
        grid_height = row_max - row_min + 1
        grid_width = col_max - col_min + 1

        grid = np.zeros((3, grid_height, grid_width), dtype=np.float32)
        for index in roi_indices:
            patch = patches[index]
            row = patch.row - row_min
            col = patch.col - col_min
            grid[0, row, col] = local_features[index, 0]
            grid[1, row, col] = local_features[index, 1]
            grid[2, row, col] = 1.0

        scale = min(side / grid_height, side / grid_width)
        resized_height = min(side, max(1, int(round(grid_height * scale))))
        resized_width = min(side, max(1, int(round(grid_width * scale))))
        source_rows = np.minimum(
            ((np.arange(resized_height) + 0.5) * grid_height / resized_height).astype(int),
            grid_height - 1,
        )
        source_cols = np.minimum(
            ((np.arange(resized_width) + 0.5) * grid_width / resized_width).astype(int),
            grid_width - 1,
        )
        resized = grid[:, source_rows][:, :, source_cols]
        row_offset = (side - resized_height) // 2
        col_offset = (side - resized_width) // 2
        row_slice = slice(row_offset, row_offset + resized_height)
        col_slice = slice(col_offset, col_offset + resized_width)

        for index in roi_indices:
            output[index, :3, row_slice, col_slice] = resized
            patch = patches[index]
            target_rows = np.flatnonzero(source_rows == patch.row - row_min)
            target_cols = np.flatnonzero(source_cols == patch.col - col_min)
            if target_rows.size and target_cols.size:
                output[index, 3][
                    row_offset + target_rows[:, None],
                    col_offset + target_cols[None, :],
                ] = 1.0
            else:
                target_row = min(
                    resized_height - 1,
                    int((patch.row - row_min + 0.5) * resized_height / grid_height),
                )
                target_col = min(
                    resized_width - 1,
                    int((patch.col - col_min + 0.5) * resized_width / grid_width),
                )
                output[index, 3, row_offset + target_row, col_offset + target_col] = 1.0
    return output


def spatial_conditioning_specification() -> dict[str, object]:
    return {
        "version": 1,
        "mode": SPATIAL_CONDITION_MODE,
        "feature": "dapi_patch_and_roi_spatial_stats",
        "condition_key": "patch_name",
        "condition_dimension": SPATIAL_CONDITION_DIMENSION,
        "radius_tiles": SPATIAL_CONDITION_RADIUS,
        "feature_names": list(SPATIAL_CONDITION_FEATURE_NAMES),
        "normalization": {
            "fit_scope": "none_input_fixed",
            "intensity_units": "uint8_div_255",
            "geometry_units": "roi_grid_normalized",
        },
        "injection": "bottleneck_mlp_film",
        "film_hidden_features": SPATIAL_FILM_HIDDEN_FEATURES,
        "film_output_initialization": "zeros",
    }


def spatial_thumbnail_conditioning_specification() -> dict[str, object]:
    return {
        "version": 1,
        "mode": SPATIAL_THUMBNAIL_CONDITION_MODE,
        "feature": "dapi_patch_roi_stats_and_neighborhood_thumbnail",
        "condition_key": "patch_name",
        "condition_dimension": SPATIAL_THUMBNAIL_CONDITION_DIMENSION,
        "stats_dimension": SPATIAL_CONDITION_DIMENSION,
        "stats_feature_names": list(SPATIAL_CONDITION_FEATURE_NAMES),
        "radius_tiles": SPATIAL_CONDITION_RADIUS,
        "thumbnail": {
            "channels": ["dapi_mean", "presence"],
            "shape": [
                SPATIAL_THUMBNAIL_CHANNELS,
                SPATIAL_THUMBNAIL_SIDE,
                SPATIAL_THUMBNAIL_SIDE,
            ],
            "tile_side": SPATIAL_THUMBNAIL_TILE_SIDE,
            "grid_side": SPATIAL_THUMBNAIL_GRID_SIDE,
            "downsample": "nonoverlapping_block_mean",
            "missing_intensity": 0.0,
            "missing_presence": 0.0,
        },
        "normalization": {
            "fit_scope": "none_input_fixed",
            "intensity_units": "uint8_div_255",
            "geometry_units": "roi_grid_normalized",
        },
        "injection": "bottleneck_stats_plus_thumbnail_cnn_film",
        "stats_film_hidden_features": SPATIAL_FILM_HIDDEN_FEATURES,
        "thumbnail_encoder_channels": [16, 32, 64, 64],
        "thumbnail_embedding_features": SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
        "thumbnail_d4_aggregation": "mean",
        "thumbnail_film_output_initialization": "zeros",
    }


def spatial_thumbnail_texture_conditioning_specification() -> dict[str, object]:
    return {
        "version": 1,
        "mode": SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE,
        "feature": "dapi_patch_roi_stats_and_neighborhood_texture_maps",
        "condition_key": "patch_name",
        "condition_dimension": SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION,
        "stats_dimension": SPATIAL_CONDITION_DIMENSION,
        "stats_feature_names": list(SPATIAL_CONDITION_FEATURE_NAMES),
        "radius_tiles": SPATIAL_CONDITION_RADIUS,
        "thumbnail": {
            "channels": [
                "dapi_mean",
                "dapi_block_std",
                "dapi_block_mean_abs_gradient",
                "presence",
            ],
            "shape": [
                SPATIAL_THUMBNAIL_TEXTURE_CHANNELS,
                SPATIAL_THUMBNAIL_SIDE,
                SPATIAL_THUMBNAIL_SIDE,
            ],
            "tile_side": SPATIAL_THUMBNAIL_TILE_SIDE,
            "grid_side": SPATIAL_THUMBNAIL_GRID_SIDE,
            "downsample": "nonoverlapping_block_mean_std_gradient",
            "missing_value": 0.0,
            "missing_presence": 0.0,
        },
        "normalization": {
            "fit_scope": "none_input_fixed",
            "intensity_units": "uint8_div_255",
            "geometry_units": "roi_grid_normalized",
        },
        "injection": "bottleneck_stats_plus_texture_thumbnail_cnn_film",
        "stats_film_hidden_features": SPATIAL_FILM_HIDDEN_FEATURES,
        "thumbnail_encoder_channels": [16, 32, 64, 64],
        "thumbnail_embedding_features": SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
        "thumbnail_d4_aggregation": "mean",
        "thumbnail_film_output_initialization": "zeros",
    }


def spatial_thumbnail_multiscale_conditioning_specification() -> dict[str, object]:
    specification = spatial_thumbnail_conditioning_specification()
    specification["mode"] = SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE
    specification["injection"] = (
        "bottleneck_and_decoder_stats_plus_thumbnail_cnn_film"
    )
    specification["decoder_film"] = {
        "channel_multipliers": [8, 4, 2, 1],
        "condition_features": (
            SPATIAL_FILM_HIDDEN_FEATURES
            + SPATIAL_THUMBNAIL_EMBEDDING_FEATURES
        ),
        "output_initialization": "zeros",
    }
    return specification


def spatial_thumbnail_ring_conditioning_specification() -> dict[str, object]:
    specification = spatial_thumbnail_conditioning_specification()
    specification["mode"] = SPATIAL_THUMBNAIL_RING_CONDITION_MODE
    specification["injection"] = (
        "bottleneck_stats_plus_ring_aware_thumbnail_cnn_film"
    )
    specification["thumbnail_spatial_pooling"] = {
        "feature_grid_side": SPATIAL_THUMBNAIL_GRID_SIDE,
        "regions": ["center", "inner_ring", "outer_ring"],
        "region_pixels": [1, 8, 16],
        "projection_input_features": 3 * SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
        "projection_output_features": SPATIAL_THUMBNAIL_EMBEDDING_FEATURES,
        "projection_mode": "residual_to_global_average",
        "projection_initialization": "zeros",
    }
    return specification


def spatial_thumbnail_aligned_conditioning_specification() -> dict[str, object]:
    specification = spatial_thumbnail_conditioning_specification()
    specification["mode"] = SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE
    specification["injection"] = (
        "bottleneck_film_plus_spatially_aligned_thumbnail_feature_add"
    )
    specification["aligned_context"] = {
        "input_channels": ["dapi_mean", "presence"],
        "input_shape": [
            SPATIAL_THUMBNAIL_CHANNELS,
            SPATIAL_THUMBNAIL_SIDE,
            SPATIAL_THUMBNAIL_SIDE,
        ],
        "source_pixels_per_thumbnail_pixel": (
            256 // SPATIAL_THUMBNAIL_TILE_SIDE
        ),
        "joint_geometric_transform": "same_d4_as_source",
        "encoder_features": SPATIAL_THUMBNAIL_ALIGNED_FEATURES,
        "encoder_dilations": list(SPATIAL_THUMBNAIL_ALIGNED_DILATIONS),
        "fusion_grid": "center_crop_to_unet_bottleneck",
        "fusion": "zero_initialized_1x1_add",
    }
    return specification


def spatial_whole_roi_conditioning_specification() -> dict[str, object]:
    specification = spatial_thumbnail_conditioning_specification()
    specification["mode"] = SPATIAL_WHOLE_ROI_CONDITION_MODE
    specification["feature"] = (
        "dapi_patch_roi_stats_neighborhood_and_whole_roi_thumbnail"
    )
    specification["condition_dimension"] = SPATIAL_WHOLE_ROI_CONDITION_DIMENSION
    specification["injection"] = (
        "bottleneck_stats_plus_local_and_whole_roi_thumbnail_cnn_film"
    )
    specification["whole_roi_thumbnail"] = {
        "channels": ["dapi_mean", "dapi_std", "presence", "target_patch"],
        "shape": [
            SPATIAL_WHOLE_ROI_CHANNELS,
            SPATIAL_WHOLE_ROI_SIDE,
            SPATIAL_WHOLE_ROI_SIDE,
        ],
        "source_grid": "roi_patch_bounding_grid",
        "resize": "aspect_preserving_nearest_letterbox",
        "missing_value": 0.0,
        "target_patch_value": 1.0,
    }
    specification["whole_roi_thumbnail_encoder_channels"] = [16, 32, 64, 64]
    specification["whole_roi_thumbnail_embedding_features"] = (
        SPATIAL_WHOLE_ROI_EMBEDDING_FEATURES
    )
    specification["whole_roi_thumbnail_d4_aggregation"] = "mean"
    specification["whole_roi_film_output_initialization"] = "zeros"
    return specification


def validate_spatial_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError("Unsupported spatial-statistics conditioning specification.")
    return report


def validate_spatial_thumbnail_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_thumbnail_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError("Unsupported spatial-thumbnail conditioning specification.")
    return report


def validate_spatial_thumbnail_texture_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_thumbnail_texture_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError(
            "Unsupported spatial-thumbnail texture conditioning specification."
        )
    return report


def validate_spatial_thumbnail_multiscale_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_thumbnail_multiscale_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError(
            "Unsupported spatial-thumbnail multiscale conditioning specification."
        )
    return report


def validate_spatial_thumbnail_ring_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_thumbnail_ring_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError(
            "Unsupported spatial-thumbnail ring conditioning specification."
        )
    return report


def validate_spatial_thumbnail_aligned_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_thumbnail_aligned_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError(
            "Unsupported spatial-thumbnail aligned conditioning specification."
        )
    return report


def validate_spatial_whole_roi_conditioning_specification(
    report: object,
) -> dict[str, object]:
    expected = spatial_whole_roi_conditioning_specification()
    if not isinstance(report, dict) or report != expected:
        raise ValueError("Unsupported whole-ROI conditioning specification.")
    return report


def _spatial_inputs_and_stats(
    directory: Path,
    names: Sequence[str],
    *,
    include_tile_thumbnails: bool = False,
    include_tile_texture: bool = False,
) -> tuple[list[PatchId], np.ndarray, np.ndarray | None]:
    if not names:
        raise ValueError("Spatial conditioning requires at least one image.")
    if len(set(names)) != len(names):
        raise ValueError("Spatial conditioning names must be unique.")
    if include_tile_thumbnails and include_tile_texture:
        raise ValueError("Choose either tile thumbnails or tile texture maps.")
    patches = [parse_patch_name(name) for name in names]
    local_rows: list[np.ndarray] = []
    thumbnail_rows: list[np.ndarray] | None = (
        [] if include_tile_thumbnails or include_tile_texture else None
    )
    for name in names:
        image = load_gray(directory / name)
        local_rows.append(local_intensity_features(image))
        if thumbnail_rows is not None:
            if include_tile_texture:
                thumbnail_rows.append(
                    downsample_tile_texture(image, SPATIAL_THUMBNAIL_TILE_SIDE)
                )
            else:
                thumbnail_rows.append(
                    downsample_tile_mean(image, SPATIAL_THUMBNAIL_TILE_SIDE)
                )
    local = np.stack(local_rows)
    context = build_spatial_context_features(
        patches,
        local,
        radius=SPATIAL_CONDITION_RADIUS,
    )
    values = np.concatenate((local, context.values), axis=1).astype(np.float32)
    if values.shape[1] != SPATIAL_CONDITION_DIMENSION:
        raise RuntimeError(
            "Spatial condition dimension changed unexpectedly: "
            f"{values.shape[1]} != {SPATIAL_CONDITION_DIMENSION}."
        )
    tile_thumbnails = (
        np.stack(thumbnail_rows) if thumbnail_rows is not None else None
    )
    return patches, values, tile_thumbnails


def build_spatial_stats_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    _, values, _ = _spatial_inputs_and_stats(directory, names)
    return (
        {name: values[index] for index, name in enumerate(names)},
        spatial_conditioning_specification(),
    )


def build_spatial_thumbnail_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    patches, stats, tiles = _spatial_inputs_and_stats(
        directory,
        names,
        include_tile_thumbnails=True,
    )
    if tiles is None:
        raise RuntimeError("Thumbnail extraction did not produce tile thumbnails.")
    thumbnails = build_neighborhood_thumbnails(
        patches,
        tiles,
        radius=SPATIAL_CONDITION_RADIUS,
    )
    values = np.concatenate(
        (stats, thumbnails.reshape(len(names), -1)),
        axis=1,
    ).astype(np.float32)
    if values.shape[1] != SPATIAL_THUMBNAIL_CONDITION_DIMENSION:
        raise RuntimeError(
            "Spatial-thumbnail condition dimension changed unexpectedly: "
            f"{values.shape[1]} != {SPATIAL_THUMBNAIL_CONDITION_DIMENSION}."
        )
    return (
        {name: values[index] for index, name in enumerate(names)},
        spatial_thumbnail_conditioning_specification(),
    )


def build_spatial_thumbnail_texture_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    patches, stats, tiles = _spatial_inputs_and_stats(
        directory,
        names,
        include_tile_texture=True,
    )
    if tiles is None:
        raise RuntimeError("Texture extraction did not produce tile maps.")
    thumbnails = build_neighborhood_texture_maps(
        patches,
        tiles,
        radius=SPATIAL_CONDITION_RADIUS,
    )
    values = np.concatenate(
        (stats, thumbnails.reshape(len(names), -1)),
        axis=1,
    ).astype(np.float32)
    if values.shape[1] != SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION:
        raise RuntimeError(
            "Spatial-thumbnail texture condition dimension changed unexpectedly: "
            f"{values.shape[1]} != {SPATIAL_THUMBNAIL_TEXTURE_CONDITION_DIMENSION}."
        )
    return (
        {name: values[index] for index, name in enumerate(names)},
        spatial_thumbnail_texture_conditioning_specification(),
    )


def build_spatial_thumbnail_multiscale_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    conditions, _ = build_spatial_thumbnail_conditioning(directory, names)
    return conditions, spatial_thumbnail_multiscale_conditioning_specification()


def build_spatial_thumbnail_ring_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    conditions, _ = build_spatial_thumbnail_conditioning(directory, names)
    return conditions, spatial_thumbnail_ring_conditioning_specification()


def build_spatial_thumbnail_aligned_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    conditions, _ = build_spatial_thumbnail_conditioning(directory, names)
    return conditions, spatial_thumbnail_aligned_conditioning_specification()


def build_spatial_whole_roi_conditioning(
    directory: Path,
    names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    patches, stats, tiles = _spatial_inputs_and_stats(
        directory,
        names,
        include_tile_thumbnails=True,
    )
    if tiles is None:
        raise RuntimeError("Thumbnail extraction did not produce tile thumbnails.")
    local_thumbnails = build_neighborhood_thumbnails(
        patches,
        tiles,
        radius=SPATIAL_CONDITION_RADIUS,
    )
    whole_roi_thumbnails = build_whole_roi_thumbnails(
        patches,
        stats[:, : len(LOCAL_FEATURE_NAMES)],
    )
    values = np.concatenate(
        (
            stats,
            local_thumbnails.reshape(len(names), -1),
            whole_roi_thumbnails.reshape(len(names), -1),
        ),
        axis=1,
    ).astype(np.float32)
    if values.shape[1] != SPATIAL_WHOLE_ROI_CONDITION_DIMENSION:
        raise RuntimeError(
            "Whole-ROI condition dimension changed unexpectedly: "
            f"{values.shape[1]} != {SPATIAL_WHOLE_ROI_CONDITION_DIMENSION}."
        )
    return (
        {name: values[index] for index, name in enumerate(names)},
        spatial_whole_roi_conditioning_specification(),
    )


def apply_spatial_stats_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_conditioning_specification(report)
    conditions, generated = build_spatial_stats_conditioning(directory, names)
    if generated != report:
        raise RuntimeError("Generated spatial conditioning metadata changed unexpectedly.")
    return conditions


def apply_spatial_thumbnail_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_thumbnail_conditioning_specification(report)
    conditions, generated = build_spatial_thumbnail_conditioning(directory, names)
    if generated != report:
        raise RuntimeError(
            "Generated spatial-thumbnail conditioning metadata changed unexpectedly."
        )
    return conditions


def apply_spatial_thumbnail_texture_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_thumbnail_texture_conditioning_specification(report)
    conditions, generated = build_spatial_thumbnail_texture_conditioning(
        directory,
        names,
    )
    if generated != report:
        raise RuntimeError(
            "Generated spatial-thumbnail texture conditioning metadata changed "
            "unexpectedly."
        )
    return conditions


def apply_spatial_thumbnail_multiscale_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_thumbnail_multiscale_conditioning_specification(report)
    conditions, generated = build_spatial_thumbnail_multiscale_conditioning(
        directory,
        names,
    )
    if generated != report:
        raise RuntimeError(
            "Generated spatial-thumbnail multiscale conditioning metadata "
            "changed unexpectedly."
        )
    return conditions


def apply_spatial_thumbnail_ring_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_thumbnail_ring_conditioning_specification(report)
    conditions, generated = build_spatial_thumbnail_ring_conditioning(
        directory,
        names,
    )
    if generated != report:
        raise RuntimeError(
            "Generated spatial-thumbnail ring conditioning metadata changed "
            "unexpectedly."
        )
    return conditions


def apply_spatial_thumbnail_aligned_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_thumbnail_aligned_conditioning_specification(report)
    conditions, generated = build_spatial_thumbnail_aligned_conditioning(
        directory,
        names,
    )
    if generated != report:
        raise RuntimeError(
            "Generated spatial-thumbnail aligned conditioning metadata "
            "changed unexpectedly."
        )
    return conditions


def apply_spatial_whole_roi_conditioning(
    directory: Path,
    names: Sequence[str],
    report: object,
) -> dict[str, np.ndarray]:
    validate_spatial_whole_roi_conditioning_specification(report)
    conditions, generated = build_spatial_whole_roi_conditioning(
        directory,
        names,
    )
    if generated != report:
        raise RuntimeError(
            "Generated whole-ROI conditioning metadata changed unexpectedly."
        )
    return conditions


def apply_complete_roi_conditioning(
    directory: Path,
    names: Sequence[str],
    report: Mapping[str, object],
) -> dict[str, float] | dict[str, np.ndarray] | None:
    """Compute input-only conditions from complete ROIs, then select query images.

    Sampling which predictions to evaluate must not sample the DAPI context.
    Low-level builders retain their explicit file-universe semantics.
    """
    if report.get("mode") == "none":
        return None
    if not names or len(set(names)) != len(names):
        raise ValueError("Condition queries must contain unique image names.")
    rois = {parse_patch_name(name).roi for name in names}
    universe = [name for name in image_names(directory) if parse_patch_name(name).roi in rois]
    missing = set(names) - set(universe)
    if missing:
        raise ValueError(f"Condition queries missing from DAPI directory: {sorted(missing)}")
    conditions = apply_input_conditioning(directory, universe, report)
    if conditions is None:
        return None
    if report.get("mode") == "roi-mean-film":
        return conditions  # Legacy conditioning uses ROI keys rather than image keys.
    return {name: conditions[name] for name in names}


def apply_input_conditioning(
    directory: Path,
    names: Sequence[str],
    report: Mapping[str, object],
) -> dict[str, float] | dict[str, np.ndarray] | None:
    mode = report.get("mode")
    if mode == "none":
        return None
    if mode == "roi-mean-film":
        return apply_roi_mean_conditioning(directory, names, report)
    if mode == SPATIAL_CONDITION_MODE:
        return apply_spatial_stats_conditioning(directory, names, report)
    if mode == SPATIAL_THUMBNAIL_CONDITION_MODE:
        return apply_spatial_thumbnail_conditioning(directory, names, report)
    if mode == SPATIAL_THUMBNAIL_TEXTURE_CONDITION_MODE:
        return apply_spatial_thumbnail_texture_conditioning(directory, names, report)
    if mode == SPATIAL_THUMBNAIL_MULTISCALE_CONDITION_MODE:
        return apply_spatial_thumbnail_multiscale_conditioning(
            directory,
            names,
            report,
        )
    if mode == SPATIAL_THUMBNAIL_RING_CONDITION_MODE:
        return apply_spatial_thumbnail_ring_conditioning(
            directory,
            names,
            report,
        )
    if mode == SPATIAL_THUMBNAIL_ALIGNED_CONDITION_MODE:
        return apply_spatial_thumbnail_aligned_conditioning(
            directory,
            names,
            report,
        )
    if mode == SPATIAL_WHOLE_ROI_CONDITION_MODE:
        return apply_spatial_whole_roi_conditioning(
            directory,
            names,
            report,
        )
    raise ValueError(f"Unsupported input conditioning mode: {mode}")


def conditioning_values_report(
    values: Mapping[str, float | Sequence[float] | np.ndarray] | None,
) -> dict[str, float] | dict[str, object] | None:
    if values is None:
        return None
    keys = sorted(values)
    rows = [np.atleast_1d(np.asarray(values[key], dtype=np.float32)) for key in keys]
    if all(row.shape == (1,) for row in rows):
        return {key: float(row[0]) for key, row in zip(keys, rows)}
    dimensions = {row.shape for row in rows}
    if len(dimensions) != 1 or any(row.ndim != 1 for row in rows):
        raise ValueError(f"Conditioning values have inconsistent shapes: {dimensions}")
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    for row in rows:
        digest.update(row.astype("<f4", copy=False).tobytes(order="C"))
    dimension = int(rows[0].shape[0])
    return {
        "key_scope": "patch_name",
        "count": len(keys),
        "dimension": dimension,
        "sha256": digest.hexdigest(),
    }
