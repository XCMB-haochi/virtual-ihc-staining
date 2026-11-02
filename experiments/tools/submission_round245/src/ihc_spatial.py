from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping
import numpy as np
from ihc_data import PatchId,image_names,load_gray,parse_patch_name


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


SPATIAL_CONDITION_RADIUS = 2


SPATIAL_FILM_HIDDEN_FEATURES = 256


SPATIAL_THUMBNAIL_CONDITION_MODE = "spatial-thumbnail-film"


SPATIAL_THUMBNAIL_TILE_SIDE = 16


SPATIAL_THUMBNAIL_GRID_SIDE = SPATIAL_CONDITION_RADIUS * 2 + 1


SPATIAL_THUMBNAIL_SIDE = SPATIAL_THUMBNAIL_TILE_SIDE * SPATIAL_THUMBNAIL_GRID_SIDE


SPATIAL_THUMBNAIL_CHANNELS = 2


SPATIAL_THUMBNAIL_EMBEDDING_FEATURES = 64


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


def _spatial_inputs_and_stats(directory,names,include_tile_thumbnails=False):
    if not names or len(names)!=len(set(names)):
        raise ValueError('Need nonempty unique image names')
    patches=[parse_patch_name(n) for n in names]
    local=[];thumb=[]
    for n in names:
        image=load_gray(directory/n)
        local.append(local_intensity_features(image))
        if include_tile_thumbnails:thumb.append(downsample_tile_mean(image,SPATIAL_THUMBNAIL_TILE_SIDE))
    local=np.stack(local)
    context=build_spatial_context_features(patches,local,radius=SPATIAL_CONDITION_RADIUS)
    values=np.concatenate((local,context.values),axis=1).astype(np.float32)
    assert values.shape[1]==SPATIAL_CONDITION_DIMENSION
    return patches,values,np.stack(thumb) if thumb else None

def apply_complete_roi_conditioning(directory,names,report):
    if report!=spatial_thumbnail_conditioning_specification():
        raise ValueError('Unexpected condition specification')
    rois={parse_patch_name(n).roi for n in names}
    universe=[n for n in image_names(directory) if parse_patch_name(n).roi in rois]
    if not set(names)<=set(universe):raise ValueError('Missing input tiles')
    values,_=build_spatial_thumbnail_conditioning(directory,universe)
    return {n:values[n] for n in names}
