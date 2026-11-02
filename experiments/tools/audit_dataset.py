from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import MARKERS, image_names, parse_patch_name, resolve_data_root


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_images(paths: list[Path]) -> dict[str, object]:
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    grayscale_rgb = 0
    channel_means = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image)
            modes[image.mode] += 1
            sizes[f"{image.width}x{image.height}"] += 1
        if array.ndim == 3 and array.shape[2] == 3:
            grayscale_rgb += int(
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2])
            )
            channel_means += array.reshape(-1, 3).sum(axis=0)
            pixel_count += array.shape[0] * array.shape[1]

    return {
        "count": len(paths),
        "modes": dict(modes),
        "sizes": dict(sizes),
        "grayscale_rgb_files": grayscale_rgb,
        "channel_means": (channel_means / max(pixel_count, 1)).round(4).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the paired virtual-staining dataset.")
    parser.add_argument("dataset", type=Path, help="Dataset directory or its parent.")
    parser.add_argument(
        "--hash-splits",
        action="store_true",
        help="Hash train/test DAPI files to find exact cross-split duplicates.",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=256,
        help="Maximum images inspected per directory; use 0 for all.",
    )
    args = parser.parse_args()

    root = resolve_data_root(args.dataset)
    train_names = image_names(root / "train" / "DAPI")
    test_names = image_names(root / "test" / "DAPI")
    train_set = set(train_names)

    pairing = {}
    for marker in MARKERS:
        marker_names = set(image_names(root / "train" / marker))
        pairing[marker] = {
            "count": len(marker_names),
            "missing_vs_dapi": sorted(train_set - marker_names)[:20],
            "extra_vs_dapi": sorted(marker_names - train_set)[:20],
        }

    roi_counts: dict[str, dict[str, int]] = {}
    for split, names in (("train", train_names), ("test", test_names)):
        counts: defaultdict[str, int] = defaultdict(int)
        bounds: dict[str, list[int]] = {}
        for name in names:
            patch = parse_patch_name(name)
            counts[patch.roi] += 1
            current = bounds.setdefault(patch.roi, [patch.row, patch.row, patch.col, patch.col])
            current[0] = min(current[0], patch.row)
            current[1] = max(current[1], patch.row)
            current[2] = min(current[2], patch.col)
            current[3] = max(current[3], patch.col)
        roi_counts[split] = {
            roi: {
                "patches": counts[roi],
                "row_min": bounds[roi][0],
                "row_max": bounds[roi][1],
                "col_min": bounds[roi][2],
                "col_max": bounds[roi][3],
            }
            for roi in sorted(counts)
        }

    image_stats = {}
    directories = [root / "test" / "DAPI"] + [
        root / "train" / name for name in ("DAPI", *MARKERS)
    ]
    for directory in directories:
        paths = sorted(directory.glob("*.jpg"))
        if args.sample_images > 0:
            paths = paths[: args.sample_images]
        image_stats[str(directory.relative_to(root))] = inspect_images(paths)

    duplicates: dict[str, object] = {"checked": False}
    if args.hash_splits:
        train_hashes: defaultdict[str, list[str]] = defaultdict(list)
        for name in train_names:
            train_hashes[file_digest(root / "train" / "DAPI" / name)].append(name)
        matches = []
        for name in test_names:
            digest = file_digest(root / "test" / "DAPI" / name)
            if digest in train_hashes:
                matches.append({"test": name, "train": train_hashes[digest]})
        duplicates = {"checked": True, "cross_split_exact_matches": matches}

    result = {
        "data_root": str(root),
        "train_dapi_count": len(train_names),
        "test_dapi_count": len(test_names),
        "train_test_filename_overlap": sorted(set(train_names) & set(test_names)),
        "pairing": pairing,
        "roi_counts": roi_counts,
        "image_stats": image_stats,
        "duplicates": duplicates,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
