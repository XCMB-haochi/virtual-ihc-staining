from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from ihc_data import (
    image_names,
    load_gray,
    parse_patch_name,
    resolve_data_root,
)
from ihc_torch import (
    PairedPatchDataset,
    build_model_from_checkpoint,
    checkpoint_conditioning,
)
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning
from search_postprocessing import evaluate, evaluate_by_roi, predict_validation


def roi_image_moments(
    data_root: Path,
    marker: str,
    names: list[str],
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "source_sum": 0.0,
            "source_squared_sum": 0.0,
            "target_sum": 0.0,
            "target_squared_sum": 0.0,
            "pixels": 0.0,
        }
    )
    for name in names:
        source = load_gray(data_root / "train" / "DAPI" / name).astype(np.float64)
        target = load_gray(data_root / "train" / marker / name).astype(np.float64)
        roi = parse_patch_name(name).roi
        totals[roi]["source_sum"] += float(source.sum())
        totals[roi]["source_squared_sum"] += float(np.square(source).sum())
        totals[roi]["target_sum"] += float(target.sum())
        totals[roi]["target_squared_sum"] += float(np.square(target).sum())
        totals[roi]["pixels"] += float(source.size)

    moments = {}
    for roi, values in sorted(totals.items()):
        pixels = values["pixels"]
        source_mean = values["source_sum"] / pixels
        target_mean = values["target_sum"] / pixels
        source_variance = max(
            values["source_squared_sum"] / pixels - source_mean**2, 0.0
        )
        target_variance = max(
            values["target_squared_sum"] / pixels - target_mean**2, 0.0
        )
        moments[roi] = {
            "source_mean": source_mean,
            "source_std": float(np.sqrt(source_variance)),
            "target_mean": target_mean,
            "target_std": float(np.sqrt(target_variance)),
        }
    return moments


def fit_regression(
    rows: list[dict[str, float]],
    feature_names: tuple[str, ...],
    target_name: str,
) -> dict[str, object]:
    features = np.asarray(
        [[row[name] for name in feature_names] for row in rows], dtype=np.float64
    )
    targets = np.asarray([row[target_name] for row in rows], dtype=np.float64)
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack(((features - center) / scale, np.ones(len(features))))
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return {
        "features": feature_names,
        "center": center,
        "scale": scale,
        "coefficients": coefficients,
    }


def apply_regression(model: dict[str, object], row: dict[str, float]) -> float:
    feature_names = model["features"]
    features = np.asarray([row[name] for name in feature_names], dtype=np.float64)
    normalized = (features - model["center"]) / model["scale"]
    design = np.append(normalized, 1.0)
    return float(design @ model["coefficients"])


def validation_roi_moments(
    names: list[str],
    sources: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "source_sum": 0.0,
            "source_squared_sum": 0.0,
            "prediction_sum": 0.0,
            "prediction_squared_sum": 0.0,
            "pixels": 0.0,
        }
    )
    for index, name in enumerate(names):
        roi = parse_patch_name(name).roi
        source = sources[index].astype(np.float64)
        prediction = predictions[index].astype(np.float64)
        totals[roi]["source_sum"] += float(source.sum())
        totals[roi]["source_squared_sum"] += float(np.square(source).sum())
        totals[roi]["prediction_sum"] += float(prediction.sum())
        totals[roi]["prediction_squared_sum"] += float(np.square(prediction).sum())
        totals[roi]["pixels"] += float(source.size)

    moments = {}
    for roi, values in sorted(totals.items()):
        pixels = values["pixels"]
        source_mean = values["source_sum"] / pixels
        prediction_mean = values["prediction_sum"] / pixels
        moments[roi] = {
            "source_mean": source_mean,
            "source_std": float(
                np.sqrt(
                    max(
                        values["source_squared_sum"] / pixels - source_mean**2,
                        0.0,
                    )
                )
            ),
            "prediction_mean": prediction_mean,
            "prediction_std": float(
                np.sqrt(
                    max(
                        values["prediction_squared_sum"] / pixels
                        - prediction_mean**2,
                        0.0,
                    )
                )
            ),
        }
    return moments


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ROI contrast calibration fitted only on paired training ROIs."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--val-rois", default="ROI020,ROI022,ROI024")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--context-halo", type=int, default=64)
    parser.add_argument("--mean-strength", type=float, default=0.75)
    parser.add_argument(
        "--contrast-strengths", default="0.25,0.5,0.75,1.0"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ranking-psnr-aggregation",
        choices=("mean-image", "global-mse"),
        default="mean-image",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    markers = tuple(checkpoint["markers"])
    if len(markers) != 1:
        raise ValueError("Affine calibration expects a single-marker checkpoint.")
    marker = markers[0]
    conditioning = checkpoint_conditioning(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)

    data_root = resolve_data_root(args.dataset)
    val_rois = {item.strip().upper() for item in args.val_rois.split(",") if item.strip()}
    all_names = image_names(data_root / "train" / "DAPI")
    train_names = [name for name in all_names if parse_patch_name(name).roi not in val_rois]
    val_names = [name for name in all_names if parse_patch_name(name).roi in val_rois]
    if not train_names or not val_names:
        raise ValueError("Training and validation ROI sets must both be non-empty.")
    condition_by_roi = apply_input_conditioning(
        data_root / "train" / "DAPI",
        val_names,
        conditioning,
    )

    train_moments = roi_image_moments(data_root, marker, train_names)
    training_rows = list(train_moments.values())
    mean_model = fit_regression(training_rows, ("source_mean",), "target_mean")
    std_models = {
        "source_std": fit_regression(training_rows, ("source_std",), "target_std"),
        "source_mean_std": fit_regression(
            training_rows, ("source_mean", "source_std"), "target_std"
        ),
    }

    loader = DataLoader(
        PairedPatchDataset(
            data_root,
            val_names,
            markers=(marker,),
            augment=False,
            context_halo=args.context_halo,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    started = time.perf_counter()
    sources, raw, targets, predicted_names = predict_validation(
        model, loader, device, args.tta, condition_by_roi
    )
    roi_moments = validation_roi_moments(predicted_names, sources, raw)

    mean_scales = {}
    for roi, row in roi_moments.items():
        expected_mean = apply_regression(mean_model, row)
        mean_scales[roi] = expected_mean / max(row["prediction_mean"], 1e-6)
        row["expected_mean"] = expected_mean
        row["mean_scale"] = mean_scales[roi]

    mean_multipliers = np.asarray(
        [
            mean_scales[parse_patch_name(name).roi] ** args.mean_strength
            for name in predicted_names
        ],
        dtype=np.float32,
    )[:, None, None]
    mean_calibrated = raw * mean_multipliers
    mean_calibrated_moments = validation_roi_moments(
        predicted_names, sources, mean_calibrated
    )

    strengths = [
        float(item.strip())
        for item in args.contrast_strengths.split(",")
        if item.strip()
    ]
    variants: list[tuple[str, np.ndarray]] = [("raw", raw), ("mean", mean_calibrated)]
    fit_diagnostics = {}
    for model_name, std_model in std_models.items():
        contrast_ratios = {}
        for roi, row in roi_moments.items():
            expected_std = apply_regression(std_model, row)
            current_std = mean_calibrated_moments[roi]["prediction_std"]
            ratio = expected_std / max(current_std, 1e-6)
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"Invalid contrast ratio for {roi}: {ratio}")
            contrast_ratios[roi] = ratio
            row[f"expected_std_{model_name}"] = expected_std
            row[f"contrast_ratio_{model_name}"] = ratio

        for strength in strengths:
            candidate = mean_calibrated.copy()
            for index, name in enumerate(predicted_names):
                roi = parse_patch_name(name).roi
                center = mean_calibrated_moments[roi]["prediction_mean"]
                multiplier = 1.0 + strength * (contrast_ratios[roi] - 1.0)
                candidate[index] = center + multiplier * (candidate[index] - center)
            variants.append((f"mean_contrast_{model_name}_{strength:g}", candidate))

        validation_errors = []
        for roi in sorted(val_rois):
            truth = roi_image_moments(
                data_root,
                marker,
                [name for name in val_names if parse_patch_name(name).roi == roi],
            )[roi]["target_std"]
            predicted = apply_regression(std_model, roi_moments[roi])
            validation_errors.append(abs(predicted - truth))
        fit_diagnostics[model_name] = {
            "features": list(std_model["features"]),
            "coefficients": std_model["coefficients"].tolist(),
            "center": std_model["center"].tolist(),
            "scale": std_model["scale"].tolist(),
            "validation_target_std_mae": float(np.mean(validation_errors)),
        }

    metrics = {}
    roi_metrics = {}
    for name, prediction in variants:
        metrics[name] = evaluate(prediction, targets, args.jpeg_quality)
        roi_metrics[name] = evaluate_by_roi(
            prediction, targets, predicted_names, args.jpeg_quality
        )
        print(json.dumps({"name": name, **metrics[name]}), flush=True)

    ranking_key = (
        "competition_score_mean_image_psnr"
        if args.ranking_psnr_aggregation == "mean-image"
        else "competition_score_global_mse_psnr"
    )
    ranking = sorted(
        ({"name": name, **values} for name, values in metrics.items()),
        key=lambda item: item[ranking_key],
        reverse=True,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "marker": marker,
        "validation_rois": sorted(val_rois),
        "validation_images": len(val_names),
        "context_halo": args.context_halo,
        "tta": args.tta,
        "jpeg_quality": args.jpeg_quality,
        "mean_strength": args.mean_strength,
        "ranking_psnr_aggregation": args.ranking_psnr_aggregation,
        "ranking_key": ranking_key,
        "training_rois": sorted(train_moments),
        "fit_diagnostics": fit_diagnostics,
        "roi_details": roi_moments,
        "ranking": ranking,
        "roi_metrics": roi_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
