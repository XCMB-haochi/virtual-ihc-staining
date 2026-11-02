from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    image_names,
    linear_fit,
    load_gray,
    parse_patch_name,
    resolve_data_root,
)
from ihc_torch import (
    PairedPatchDataset,
    batch_condition_tensor,
    build_model_from_checkpoint,
    center_crop_like,
    checkpoint_conditioning,
    competition_score,
    predict_with_tta,
    psnr_from_global_error,
)
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").convert("RGB").save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def fit_training_statistics(
    data_root: Path,
    marker: str,
    names: list[str],
) -> dict[str, object]:
    source_hist = np.zeros(256, dtype=np.int64)
    target_sums = np.zeros(256, dtype=np.float64)
    roi_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"source_sum": 0.0, "target_sum": 0.0, "pixels": 0.0}
    )
    for name in names:
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / marker / name)
        source_u8 = source.astype(np.uint8).ravel()
        source_hist += np.bincount(source_u8, minlength=256)
        target_sums += np.bincount(
            source_u8, weights=target.ravel(), minlength=256
        )
        roi = parse_patch_name(name).roi
        roi_totals[roi]["source_sum"] += float(source.sum())
        roi_totals[roi]["target_sum"] += float(target.sum())
        roi_totals[roi]["pixels"] += float(source.size)

    observed = source_hist > 0
    conditional_lut = np.zeros(256, dtype=np.float32)
    conditional_lut[observed] = target_sums[observed] / source_hist[observed]
    missing = np.flatnonzero(~observed)
    present = np.flatnonzero(observed)
    if len(missing):
        conditional_lut[missing] = np.interp(missing, present, conditional_lut[present])

    source_means = [
        values["source_sum"] / values["pixels"] for values in roi_totals.values()
    ]
    target_means = [
        values["target_sum"] / values["pixels"] for values in roi_totals.values()
    ]
    slope, intercept = linear_fit(source_means, target_means)
    return {
        "conditional_lut": conditional_lut,
        "roi_fit": (slope, intercept),
    }


@torch.no_grad()
def predict_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: str,
    condition_by_roi: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sources: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    model.eval()
    for source, target, batch_names in loader:
        condition = batch_condition_tensor(
            batch_names, condition_by_roi, device
        )
        prediction = center_crop_like(
            predict_with_tta(model, source.to(device), tta, condition), target
        )
        source_center = center_crop_like(source, target)
        sources.append(np.rint(source_center.numpy()[:, 0] * 255.0).astype(np.uint8))
        predictions.append(prediction.cpu().numpy()[:, 0].astype(np.float32) * 255.0)
        targets.append(np.rint(target.numpy()[:, 0] * 255.0).astype(np.uint8))
        names.extend(batch_names)
    return (
        np.concatenate(sources),
        np.concatenate(predictions),
        np.concatenate(targets),
        names,
    )


def roi_scale_multipliers(
    names: list[str],
    sources: np.ndarray,
    predictions: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"source_sum": 0.0, "prediction_sum": 0.0, "pixels": 0.0}
    )
    for index, name in enumerate(names):
        roi = parse_patch_name(name).roi
        totals[roi]["source_sum"] += float(sources[index].sum())
        totals[roi]["prediction_sum"] += float(predictions[index].sum())
        totals[roi]["pixels"] += float(sources[index].size)

    details = {}
    scales = {}
    for roi, values in totals.items():
        source_mean = values["source_sum"] / values["pixels"]
        prediction_mean = values["prediction_sum"] / values["pixels"]
        expected_mean = slope * source_mean + intercept
        scale = expected_mean / max(prediction_mean, 1e-6)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid ROI scale for {roi}: {scale}")
        scales[roi] = scale
        details[roi] = {
            "source_mean": source_mean,
            "prediction_mean": prediction_mean,
            "expected_mean": expected_mean,
            "scale": scale,
        }
    multipliers = np.asarray(
        [scales[parse_patch_name(name).roi] for name in names], dtype=np.float32
    )[:, None, None]
    return multipliers, details


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    jpeg_quality: int,
) -> dict[str, float]:
    ssim_sum = 0.0
    psnr_sum = 0.0
    official_score_sum = 0.0
    normalized_psnr_sum = 0.0
    squared_error_sum = 0.0
    pixel_count = 0
    for index in range(len(prediction)):
        candidate = np.clip(prediction[index], 0, 255)
        if jpeg_quality > 0:
            candidate = jpeg_roundtrip(candidate, jpeg_quality)
        truth = target[index].astype(np.float32)
        image_ssim = float(
            structural_similarity(
                truth,
                candidate,
                data_range=255,
                win_size=7,
            )
        )
        image_psnr = float(
            peak_signal_noise_ratio(truth, candidate, data_range=255)
        )
        ssim_sum += image_ssim
        psnr_sum += image_psnr
        official_score_sum += competition_score(image_ssim, image_psnr)
        normalized_psnr_sum += float(np.clip(image_psnr / 50.0, 0.0, 1.0))
        error = truth.astype(np.float64) - candidate.astype(np.float64)
        squared_error_sum += float(np.square(error).sum())
        pixel_count += int(error.size)
    ssim = ssim_sum / len(prediction)
    psnr_mean_image = psnr_sum / len(prediction)
    official_score = official_score_sum / len(prediction)
    psnr_global_mse = psnr_from_global_error(squared_error_sum, pixel_count)
    return {
        "ssim": ssim,
        "psnr": psnr_mean_image,
        "competition_score": official_score,
        "competition_score_official": official_score,
        "normalized_psnr_mean": normalized_psnr_sum / len(prediction),
        "psnr_mean_image": psnr_mean_image,
        "competition_score_mean_image_psnr": competition_score(
            ssim, psnr_mean_image
        ),
        "psnr_global_mse": psnr_global_mse,
        "competition_score_global_mse_psnr": competition_score(
            ssim, psnr_global_mse
        ),
    }


def evaluate_by_roi(
    prediction: np.ndarray,
    target: np.ndarray,
    names: list[str],
    jpeg_quality: int,
) -> dict[str, dict[str, float]]:
    roi_indices: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        roi_indices[parse_patch_name(name).roi].append(index)
    return {
        roi: evaluate(prediction[indices], target[indices], jpeg_quality)
        for roi, indices in sorted(roi_indices.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search deterministic post-processing on a held-out ROI fold."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--val-rois", default="ROI020,ROI021,ROI022,ROI023,ROI024")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--context-halo", type=int)
    parser.add_argument(
        "--profile",
        choices=("calibration", "full"),
        default="full",
        help="Calibration-only avoids re-testing discarded post-processing families.",
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
        raise ValueError("Post-processing search expects a single-marker checkpoint.")
    marker = markers[0]
    conditioning = checkpoint_conditioning(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)
    checkpoint_args = checkpoint.get("args", {})
    context_halo = (
        args.context_halo
        if args.context_halo is not None
        else int(checkpoint_args.get("context_halo", 0))
    )

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

    statistics = fit_training_statistics(data_root, marker, train_names)
    loader = DataLoader(
        PairedPatchDataset(
            data_root,
            val_names,
            markers=(marker,),
            augment=False,
            context_halo=context_halo,
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
    slope, intercept = statistics["roi_fit"]
    roi_scales, roi_details = roi_scale_multipliers(
        predicted_names, sources, raw, slope, intercept
    )
    lut_prediction = statistics["conditional_lut"][sources]

    variants: list[tuple[str, np.ndarray]] = [("raw", raw)]
    for strength in (0.25, 0.5, 0.75, 1.0, 1.25):
        multiplier = 1.0 + strength * (roi_scales - 1.0)
        variants.append((f"roi_scale_{strength:g}", raw * multiplier))
    for strength in (0.25, 0.5, 0.75):
        variants.append(
            (f"roi_scale_log_{strength:g}", raw * np.power(roi_scales, strength))
        )
    variants.append(("roi_scale_sqrt", raw * np.sqrt(roi_scales)))

    if args.profile == "full":
        roi_scaled = raw * roi_scales
        for weight in (0.01, 0.025, 0.05, 0.075, 0.1):
            variants.append(
                (
                    f"roi_scale_1_lut_{weight:g}",
                    roi_scaled * (1.0 - weight) + lut_prediction * weight,
                )
            )

        for sigma in (0.5, 0.8, 1.0):
            blurred = gaussian_filter(roi_scaled, sigma=(0.0, sigma, sigma))
            for strength in (-0.25, -0.1, 0.1, 0.25, 0.4):
                candidate = roi_scaled + strength * (blurred - roi_scaled)
                variants.append((f"roi_scale_1_blur_{sigma:g}_{strength:g}", candidate))

        for threshold in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            candidate = roi_scaled.copy()
            candidate[candidate < threshold] = 0.0
            variants.append((f"roi_scale_1_hard_threshold_{threshold}", candidate))

        normalized = np.clip(roi_scaled / 255.0, 0.0, 1.0)
        for gamma in (0.96, 0.98, 1.02, 1.04):
            variants.append((f"roi_scale_1_gamma_{gamma:g}", normalized**gamma * 255.0))

    metrics = {}
    for index, (name, prediction) in enumerate(variants, start=1):
        variant_started = time.perf_counter()
        metrics[name] = evaluate(prediction, targets, args.jpeg_quality)
        metrics[name]["seconds"] = time.perf_counter() - variant_started
        print(
            json.dumps(
                {"variant": index, "total": len(variants), "name": name, **metrics[name]}
            ),
            flush=True,
        )

    ranking_key = (
        "competition_score_mean_image_psnr"
        if args.ranking_psnr_aggregation == "mean-image"
        else "competition_score_global_mse_psnr"
    )
    ranking = sorted(
        (
            {"name": name, **values}
            for name, values in metrics.items()
        ),
        key=lambda item: item[ranking_key],
        reverse=True,
    )
    key_names = {
        "raw",
        "roi_scale_0.5",
        "roi_scale_log_0.5",
        "roi_scale_log_0.75",
        "roi_scale_sqrt",
        "roi_scale_1",
    }
    roi_metrics = {
        name: evaluate_by_roi(prediction, targets, predicted_names, args.jpeg_quality)
        for name, prediction in variants
        if name in key_names
    }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "marker": marker,
        "validation_rois": sorted(val_rois),
        "validation_images": len(val_names),
        "tta": args.tta,
        "context_halo": context_halo,
        "jpeg_quality": args.jpeg_quality,
        "profile": args.profile,
        "ranking_psnr_aggregation": args.ranking_psnr_aggregation,
        "ranking_key": ranking_key,
        "aggregation_note": (
            "Legacy psnr/competition_score aliases use mean per-image PSNR; "
            "global_mse fields pool squared error before computing PSNR."
        ),
        "roi_fit": {"slope": slope, "intercept": intercept},
        "roi_details": roi_details,
        "ranking": ranking,
        "key_roi_metrics": roi_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
