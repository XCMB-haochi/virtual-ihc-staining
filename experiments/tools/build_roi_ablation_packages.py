from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import parse_patch_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def archive_entries(archive: zipfile.ZipFile, marker: str) -> dict[str, str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"Archive contains duplicate paths: {archive.filename}")
    if archive.testzip() is not None:
        raise ValueError(f"Archive failed CRC validation: {archive.filename}")

    prefix = f"results/test/{marker}/"
    entries: dict[str, str] = {}
    for name in names:
        if not name.startswith(prefix) or not name.lower().endswith(".jpg"):
            raise ValueError(f"Unexpected archive path: {name}")
        patch_name = Path(name).name
        if patch_name in entries:
            raise ValueError(f"Duplicate patch filename in archive: {patch_name}")
        if name != f"{prefix}{patch_name}":
            raise ValueError(f"Archive path is not canonical: {name}")
        parse_patch_name(patch_name)
        entries[patch_name] = name
    if not entries:
        raise ValueError(f"Archive contains no {marker} JPEG files: {archive.filename}")
    return entries


def build_leave_one_roi_out(
    primary_zip: Path,
    fallback_zip: Path,
    output_zip: Path,
    marker: str,
    fallback_roi: str,
) -> dict[str, object]:
    fallback_roi = fallback_roi.upper()
    primary_zip = primary_zip.resolve()
    fallback_zip = fallback_zip.resolve()
    output_zip = output_zip.resolve()
    if output_zip in {primary_zip, fallback_zip}:
        raise ValueError("Output archive must not overwrite an input archive.")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_zip.name}.",
        suffix=".tmp",
        dir=output_zip.parent,
    )
    os.close(file_descriptor)
    temporary_zip = Path(temporary_name)
    try:
        with (
            zipfile.ZipFile(primary_zip) as primary,
            zipfile.ZipFile(fallback_zip) as fallback,
        ):
            primary_entries = archive_entries(primary, marker)
            fallback_entries = archive_entries(fallback, marker)
            if set(primary_entries) != set(fallback_entries):
                missing = sorted(set(primary_entries) - set(fallback_entries))
                extra = sorted(set(fallback_entries) - set(primary_entries))
                raise ValueError(
                    f"Archive file sets differ; fallback missing={missing[:5]}, "
                    f"fallback extra={extra[:5]}"
                )

            selected_count = 0
            with zipfile.ZipFile(
                temporary_zip,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as output:
                for patch_name in sorted(primary_entries):
                    roi = parse_patch_name(patch_name).roi
                    source = fallback if roi == fallback_roi else primary
                    source_entries = (
                        fallback_entries if roi == fallback_roi else primary_entries
                    )
                    if roi == fallback_roi:
                        selected_count += 1
                    archive_name = primary_entries[patch_name]
                    output.writestr(
                        archive_name,
                        source.read(source_entries[patch_name]),
                    )

        if selected_count == 0:
            raise ValueError(f"ROI {fallback_roi} is absent from the archives.")

        with zipfile.ZipFile(temporary_zip) as output:
            bad_crc = output.testzip()
            entry_count = len(output.namelist())
        if bad_crc is not None:
            raise ValueError(f"Generated archive failed CRC validation at {bad_crc}")
        temporary_zip.replace(output_zip)
    finally:
        temporary_zip.unlink(missing_ok=True)

    return {
        "fallback_roi": fallback_roi,
        "fallback_images": selected_count,
        "entries": entry_count,
        "bytes": output_zip.stat().st_size,
        "sha256": file_sha256(output_zip),
        "primary": str(primary_zip),
        "primary_sha256": file_sha256(primary_zip),
        "fallback": str(fallback_zip),
        "fallback_sha256": file_sha256(fallback_zip),
        "output": str(output_zip.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leave-one-ROI-out submission probes from two validated ZIPs."
        )
    )
    parser.add_argument("primary_zip", type=Path)
    parser.add_argument("fallback_zip", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--marker", default="CD68")
    parser.add_argument("--prefix", default="roi_scale_ablation")
    args = parser.parse_args()

    with zipfile.ZipFile(args.primary_zip) as primary:
        entries = archive_entries(primary, args.marker)
    rois = sorted({parse_patch_name(name).roi for name in entries})

    reports = []
    for roi in rois:
        output_zip = args.output_dir / f"{args.prefix}_disable_{roi}.zip"
        report = build_leave_one_roi_out(
            args.primary_zip,
            args.fallback_zip,
            output_zip,
            args.marker,
            roi,
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    print(json.dumps({"packages": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
