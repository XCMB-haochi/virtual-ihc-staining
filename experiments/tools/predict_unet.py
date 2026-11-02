from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    MARKERS,
    fit_roi_mean_calibration,
    image_names,
    load_context_gray,
    load_gray,
    parse_patch_name,
    resolve_data_root,
)
from ihc_torch import (
    batch_condition_tensor,
    build_model_from_checkpoint,
    center_crop_like,
    checkpoint_conditioning,
    predict_with_tta,
    save_grayscale_rgb,
    shift_tensor_edge,
)
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning


def parse_source_offset(raw: str) -> tuple[int, int]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "Output source offset must contain two comma-separated integers."
        ) from error
    if len(values) != 2 or any(not -8 <= value <= 8 for value in values):
        raise ValueError(
            "Output source offset must contain two integers from -8 to 8."
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate virtual-staining JPEG files.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--marker", choices=MARKERS)
    parser.add_argument("--all-markers", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--filename-suffix", default="")
    parser.add_argument("--tta", choices=("none", "d4"), default="none")
    parser.add_argument("--context-halo", type=int)
    parser.add_argument(
        "--output-source-offset",
        default="0,0",
        help="Sample the predicted output at this fixed row,column source offset.",
    )
    parser.add_argument("--blend-checkpoint", type=Path)
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument(
        "--roi-scale",
        action="store_true",
        help="Match each test ROI mean to a DAPI-to-target relation fitted on training ROIs.",
    )
    parser.add_argument(
        "--roi-scale-strength",
        type=float,
        default=1.0,
        help="Shrink ROI scaling toward 1.0; 0 disables its effect and 1 applies it fully.",
    )
    parser.add_argument(
        "--roi-scale-mode",
        choices=("linear", "log"),
        default="linear",
        help="Interpolate the ROI correction in linear or log scale space.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.marker is None) == (not args.all_markers):
        raise ValueError("Choose exactly one of --marker or --all-markers.")
    if args.roi_scale and args.max_images > 0:
        raise ValueError("ROI scaling requires complete test ROIs; do not use --max-images.")
    if args.roi_scale_strength < 0:
        raise ValueError("--roi-scale-strength must be non-negative.")
    if not 0 <= args.blend_weight <= 1:
        raise ValueError("--blend-weight must be in [0, 1].")
    if args.blend_checkpoint is not None and args.all_markers:
        raise ValueError("Checkpoint blending currently supports one --marker only.")
    output_source_offset = parse_source_offset(args.output_source_offset)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    markers = tuple(checkpoint.get("markers", MARKERS))
    conditioning = checkpoint_conditioning(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    blend_model = None
    blend_index = None
    blend_checkpoint_args = None
    blend_conditioning: dict[str, object] | None = None
    if args.blend_checkpoint is not None:
        blend_checkpoint = torch.load(
            args.blend_checkpoint, map_location=device, weights_only=False
        )
        blend_markers = tuple(blend_checkpoint.get("markers", MARKERS))
        if args.marker not in blend_markers:
            raise ValueError(
                f"Marker {args.marker} is absent from blend checkpoint: {blend_markers}"
            )
        blend_index = blend_markers.index(args.marker)
        blend_conditioning = checkpoint_conditioning(blend_checkpoint)
        blend_model = build_model_from_checkpoint(blend_checkpoint, device)
        blend_model.eval()
        blend_checkpoint_args = blend_checkpoint.get("args", {})
    checkpoint_args = checkpoint.get("args", {})
    context_halo = (
        args.context_halo
        if args.context_halo is not None
        else int(checkpoint_args.get("context_halo", 0))
    )
    if args.context_halo is None and blend_checkpoint_args is not None:
        blend_context_halo = int(blend_checkpoint_args.get("context_halo", 0))
        if blend_context_halo != context_halo:
            raise ValueError(
                "Checkpoints use different context halos; pass --context-halo "
                "explicitly to define shared blend inference."
            )

    data_root = resolve_data_root(args.dataset)
    names = image_names(data_root / "test" / "DAPI")
    condition_by_roi = apply_input_conditioning(
        data_root / "test" / "DAPI",
        names,
        conditioning,
    )
    blend_condition_by_roi = (
        apply_input_conditioning(
            data_root / "test" / "DAPI", names, blend_conditioning
        )
        if blend_conditioning is not None
        else None
    )
    if args.max_images > 0:
        names = names[: args.max_images]
    selected = markers if args.all_markers else (args.marker,)
    marker_indices = {marker: markers.index(marker) for marker in selected}
    if args.blend_checkpoint is not None:
        marker_indices = {args.marker: 0}

    def infer(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        batch_names: list[str],
    ) -> torch.Tensor:
        condition = batch_condition_tensor(batch_names, condition_by_roi, device)
        primary = center_crop_like(
            predict_with_tta(model, tensor, args.tta, condition), reference
        )
        if blend_model is None or blend_index is None:
            return primary
        primary_index = markers.index(args.marker)
        blend_condition = batch_condition_tensor(
            batch_names, blend_condition_by_roi, device
        )
        secondary = center_crop_like(
            predict_with_tta(blend_model, tensor, args.tta, blend_condition),
            reference,
        )
        return (
            primary[:, primary_index : primary_index + 1] * (1.0 - args.blend_weight)
            + secondary[:, blend_index : blend_index + 1] * args.blend_weight
        )

    def infer_with_offset(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        batch_names: list[str],
    ) -> torch.Tensor:
        return shift_tensor_edge(infer(tensor, reference, batch_names), output_source_offset)
    output_directories = {
        marker: args.output / marker if args.all_markers else args.output
        for marker in selected
    }
    for marker, directory in output_directories.items():
        existing = list(directory.glob("*.jpg")) if directory.is_dir() else []
        if existing:
            raise FileExistsError(
                f"Refusing to mix new {marker} predictions with {len(existing)} "
                f"existing JPEG files in {directory}. Use an empty output directory."
            )

    roi_scales: dict[str, dict[str, float]] = {
        marker: defaultdict(lambda: 1.0) for marker in selected
    }
    calibration_report: dict[str, object] = {}
    if args.roi_scale:
        source_totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {"sum": 0.0, "pixels": 0.0}
        )
        prediction_totals: dict[str, dict[str, float]] = {
            marker: defaultdict(float) for marker in selected
        }
        with torch.no_grad():
            for start in range(0, len(names), args.batch_size):
                batch_names = names[start : start + args.batch_size]
                source_center = np.stack(
                    [load_gray(data_root / "test" / "DAPI" / name) for name in batch_names]
                )
                source_context = np.stack(
                    [
                        load_context_gray(
                            data_root / "test" / "DAPI", name, context_halo
                        )
                        for name in batch_names
                    ]
                )
                tensor = torch.from_numpy(source_context[:, None, ...] / 255.0).float().to(device)
                reference = torch.from_numpy(source_center[:, None, ...])
                prediction = (
                    infer_with_offset(tensor, reference, batch_names).cpu().numpy()
                    * 255.0
                )
                for batch_index, name in enumerate(batch_names):
                    roi = parse_patch_name(name).roi
                    source_totals[roi]["sum"] += float(source_center[batch_index].sum())
                    source_totals[roi]["pixels"] += float(source_center[batch_index].size)
                    for marker, index in marker_indices.items():
                        prediction_totals[marker][roi] += float(
                            prediction[batch_index, index].sum()
                        )

        for marker in selected:
            slope, intercept = fit_roi_mean_calibration(data_root, marker)
            marker_report = {
                "fit": {"slope": slope, "intercept": intercept},
                "rois": {},
            }
            for roi, values in source_totals.items():
                source_mean = values["sum"] / values["pixels"]
                prediction_mean = prediction_totals[marker][roi] / values["pixels"]
                expected_mean = slope * source_mean + intercept
                scale = expected_mean / max(prediction_mean, 1e-6)
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        f"Invalid ROI scale for {marker}/{roi}: {scale} "
                        f"(expected_mean={expected_mean}, prediction_mean={prediction_mean})"
                    )
                adjusted_scale = (
                    scale**args.roi_scale_strength
                    if args.roi_scale_mode == "log"
                    else 1.0 + args.roi_scale_strength * (scale - 1.0)
                )
                roi_scales[marker][roi] = adjusted_scale
                marker_report["rois"][roi] = {
                    "source_mean": source_mean,
                    "prediction_mean": prediction_mean,
                    "expected_mean": expected_mean,
                    "scale": scale,
                    "strength": args.roi_scale_strength,
                    "mode": args.roi_scale_mode,
                    "adjusted_scale": adjusted_scale,
                }
            calibration_report[marker] = marker_report
        print(json.dumps({"roi_scale_calibration": calibration_report}, indent=2))

    with torch.no_grad():
        for start in range(0, len(names), args.batch_size):
            batch_names = names[start : start + args.batch_size]
            source_center = np.stack(
                [load_gray(data_root / "test" / "DAPI" / name) for name in batch_names]
            )
            source_context = np.stack(
                [
                    load_context_gray(data_root / "test" / "DAPI", name, context_halo)
                    for name in batch_names
                ]
            )
            tensor = torch.from_numpy(source_context[:, None, ...] / 255.0).float().to(device)
            reference = torch.from_numpy(source_center[:, None, ...])
            prediction = infer_with_offset(tensor, reference, batch_names).cpu().numpy()
            for batch_index, name in enumerate(batch_names):
                output_name = f"{Path(name).stem}{args.filename_suffix}.jpg"
                roi = parse_patch_name(name).roi
                for marker, index in marker_indices.items():
                    save_grayscale_rgb(
                        prediction[batch_index, index] * roi_scales[marker][roi],
                        output_directories[marker] / output_name,
                        quality=args.jpeg_quality,
                    )


if __name__ == "__main__":
    main()
