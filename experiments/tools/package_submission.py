from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and package virtual-staining predictions."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("test_dapi", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument(
        "--marker",
        required=True,
        choices=("CD45RO", "CD68", "HLA-DR", "Vimentin"),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix already present in prediction filenames.",
    )
    parser.add_argument(
        "--archive-suffix",
        help="Suffix to use inside the ZIP; defaults to --suffix.",
    )
    args = parser.parse_args()

    expected_inputs = sorted(args.test_dapi.glob("*.jpg"))
    if not expected_inputs:
        raise ValueError(f"No test DAPI JPEG files found in {args.test_dapi}")
    expected_outputs = {
        f"{path.stem}{args.suffix}.jpg": path.name for path in expected_inputs
    }
    archive_suffix = args.suffix if args.archive_suffix is None else args.archive_suffix
    actual = {path.name: path for path in args.predictions.glob("*.jpg")}
    missing = sorted(set(expected_outputs) - set(actual))
    extra = sorted(set(actual) - set(expected_outputs))
    errors: list[str] = []
    modes: dict[str, int] = {}
    for name, path in actual.items():
        with Image.open(path) as image:
            modes[image.mode] = modes.get(image.mode, 0) + 1
            if image.size != (256, 256):
                errors.append(f"{name}: unexpected size {image.size}")
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3:
                errors.append(f"{name}: expected RGB array, got {array.shape}")
            elif not (
                np.array_equal(array[..., 0], array[..., 1])
                and np.array_equal(array[..., 1], array[..., 2])
            ):
                errors.append(f"{name}: RGB channels differ")

    report = {
        "expected": len(expected_outputs),
        "actual": len(actual),
        "missing": missing,
        "extra": extra,
        "image_errors": errors,
        "modes": modes,
    }
    if missing or extra or errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(actual):
            source_stem = Path(name).stem
            if args.suffix:
                source_stem = source_stem[: -len(args.suffix)]
            archive_name = f"{source_stem}{archive_suffix}.jpg"
            archive.write(
                actual[name],
                arcname=f"results/test/{args.marker}/{archive_name}",
            )
    report["archive_suffix"] = archive_suffix
    report["zip"] = str(args.output_zip.resolve())
    report["zip_bytes"] = args.output_zip.stat().st_size
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
