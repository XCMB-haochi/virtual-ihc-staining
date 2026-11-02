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

