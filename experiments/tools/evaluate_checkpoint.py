from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    MARKERS,
    image_names,
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
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning, conditioning_values_report


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(gray, mode="L").convert("RGB").save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def update_metrics(
    totals: dict[str, dict[str, float]],
    key: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    totals[key]["ssim"] += float(
        structural_similarity(target, prediction, data_range=255)
    )
    totals[key]["psnr"] += float(
        peak_signal_noise_ratio(target, prediction, data_range=255)
    )
    error = target.astype(np.float64) - prediction.astype(np.float64)
    totals[key]["squared_error"] += float(np.square(error).sum())
    totals[key]["pixels"] += int(error.size)
    totals[key]["count"] += 1


def summarize(values: dict[str, float]) -> dict[str, float]:
    ssim = values["ssim"] / values["count"]
    psnr_mean_image = values["psnr"] / values["count"]
    psnr_global_mse = psnr_from_global_error(
        values["squared_error"], int(values["pixels"])
    )
    return {
        "ssim": ssim,
        "psnr": psnr_mean_image,
        "competition_score": competition_score(ssim, psnr_mean_image),
        "psnr_mean_image": psnr_mean_image,
        "competition_score_mean_image_psnr": competition_score(
            ssim, psnr_mean_image
        ),
        "psnr_global_mse": psnr_global_mse,
        "competition_score_global_mse_psnr": competition_score(
            ssim, psnr_global_mse
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with scikit-image and JPEG round trips."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--val-rois", default="ROI020,ROI021,ROI022,ROI023,ROI024")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--jpeg-qualities", default="75,95,100")
    parser.add_argument("--tta", choices=("none", "d4"), default="none")
    parser.add_argument("--context-halo", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    markers = tuple(checkpoint.get("markers", MARKERS))
    conditioning = checkpoint_conditioning(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    checkpoint_args = checkpoint.get("args", {})
    context_halo = (
        args.context_halo
        if args.context_halo is not None
        else int(checkpoint_args.get("context_halo", 0))
    )

    data_root = resolve_data_root(args.dataset)
    val_rois = {item.strip().upper() for item in args.val_rois.split(",") if item.strip()}
    names = [
        name
        for name in image_names(data_root / "train" / "DAPI")
        if parse_patch_name(name).roi in val_rois
    ]
    condition_by_roi = apply_input_conditioning(
        data_root / "train" / "DAPI",
        names,
        conditioning,
    )
    if args.max_val_images > 0 and len(names) > args.max_val_images:
        indices = np.linspace(0, len(names) - 1, args.max_val_images, dtype=int)
        names = [names[index] for index in indices]
    loader = DataLoader(
        PairedPatchDataset(
            data_root,
            names,
            markers=markers,
            augment=False,
            context_halo=context_halo,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    qualities = [int(value.strip()) for value in args.jpeg_qualities.split(",")]
    def empty_totals() -> dict[str, float]:
        return {
            "ssim": 0.0,
            "psnr": 0.0,
            "squared_error": 0.0,
            "pixels": 0,
            "count": 0,
        }

    totals = {marker: defaultdict(empty_totals) for marker in markers}
    roi_totals = {
        marker: defaultdict(lambda: defaultdict(empty_totals)) for marker in markers
    }
    roi_sources: dict[str, dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "pixels": 0}
    )

    with torch.no_grad():
        for source, target, batch_names in loader:
            condition = batch_condition_tensor(
                batch_names, condition_by_roi, device
            )
            prediction = center_crop_like(
                predict_with_tta(
                    model, source.to(device), args.tta, condition
                ),
                target,
            ).cpu().numpy() * 255.0
            source_center = center_crop_like(source, target).numpy()[:, 0] * 255.0
            target = target.numpy() * 255.0
            for batch_index in range(prediction.shape[0]):
                roi = parse_patch_name(batch_names[batch_index]).roi
                roi_sources[roi]["sum"] += float(source_center[batch_index].sum())
                roi_sources[roi]["pixels"] += int(source_center[batch_index].size)
                for marker_index, marker in enumerate(markers):
                    raw = prediction[batch_index, marker_index]
                    truth = target[batch_index, marker_index]
                    update_metrics(totals[marker], "float", raw, truth)
                    update_metrics(roi_totals[marker][roi], "float", raw, truth)
                    for quality in qualities:
                        encoded = jpeg_roundtrip(raw, quality)
                        update_metrics(
                            totals[marker],
                            f"jpeg_q{quality}",
                            encoded,
                            truth,
                        )
                        update_metrics(
                            roi_totals[marker][roi],
                            f"jpeg_q{quality}",
                            encoded,
                            truth,
                        )

    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_rois": sorted(val_rois),
        "validation_images": len(names),
        "tta": args.tta,
        "context_halo": context_halo,
        "conditioning": conditioning,
        "validation_conditions": conditioning_values_report(condition_by_roi),
        "aggregation_note": (
            "Legacy psnr/competition_score aliases use mean per-image PSNR; "
            "global_mse fields pool squared error before computing PSNR."
        ),
        "roi_source_means": {
            roi: values["sum"] / values["pixels"]
            for roi, values in sorted(roi_sources.items())
        },
        "markers": {
            marker: {
                variant: summarize(values)
                for variant, values in variants.items()
            }
            for marker, variants in totals.items()
        },
        "roi_metrics": {
            marker: {
                roi: {
                    variant: summarize(values)
                    for variant, values in variants.items()
                }
                for roi, variants in sorted(marker_rois.items())
            }
            for marker, marker_rois in roi_totals.items()
        },
    }
    output = json.dumps(results, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
