from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    MARKERS,
    fit_roi_mean_calibration,
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
    predict_with_tta,
)
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning
from search_postprocessing import evaluate, evaluate_by_roi, roi_scale_multipliers


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    marker: str,
) -> tuple[torch.nn.Module, int, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    markers = tuple(checkpoint["markers"])
    if marker not in markers:
        raise ValueError(f"{marker} is not present in {checkpoint_path}: {markers}")
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    return model, markers.index(marker), checkpoint


@torch.no_grad()
def predict_pair(
    model_a: torch.nn.Module,
    index_a: int,
    model_b: torch.nn.Module,
    index_b: int,
    loader: DataLoader,
    device: torch.device,
    tta: str,
    condition_by_roi_a: dict[str, float] | None = None,
    condition_by_roi_b: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sources = []
    predictions_a = []
    predictions_b = []
    targets = []
    names: list[str] = []
    for source, target, batch_names in loader:
        source_device = source.to(device)
        condition_a = batch_condition_tensor(
            batch_names, condition_by_roi_a, device
        )
        condition_b = batch_condition_tensor(
            batch_names, condition_by_roi_b, device
        )
        prediction_a = center_crop_like(
            predict_with_tta(model_a, source_device, tta, condition_a), target
        )[:, index_a]
        prediction_b = center_crop_like(
            predict_with_tta(model_b, source_device, tta, condition_b), target
        )[:, index_b]
        source_center = center_crop_like(source, target)
        sources.append(np.rint(source_center.numpy()[:, 0] * 255.0).astype(np.uint8))
        predictions_a.append(prediction_a.cpu().numpy().astype(np.float32) * 255.0)
        predictions_b.append(prediction_b.cpu().numpy().astype(np.float32) * 255.0)
        targets.append(np.rint(target.numpy()[:, 0] * 255.0).astype(np.uint8))
        names.extend(batch_names)
    return (
        np.concatenate(sources),
        np.concatenate(predictions_a),
        np.concatenate(predictions_b),
        np.concatenate(targets),
        names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate two-model blends with fixed ROI calibration variants."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--marker", choices=MARKERS, default="CD68")
    parser.add_argument("--val-rois", default="ROI020,ROI021,ROI022,ROI023,ROI024")
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--context-halo", type=int, default=64)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--batch-size", type=int, default=8)
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
    model_a, index_a, checkpoint_a = load_model(
        args.checkpoint_a, device, args.marker
    )
    model_b, index_b, checkpoint_b = load_model(
        args.checkpoint_b, device, args.marker
    )

    data_root = resolve_data_root(args.dataset)
    val_rois = {item.strip().upper() for item in args.val_rois.split(",") if item.strip()}
    all_names = image_names(data_root / "train" / "DAPI")
    train_names = [name for name in all_names if parse_patch_name(name).roi not in val_rois]
    val_names = [name for name in all_names if parse_patch_name(name).roi in val_rois]
    conditioning_a = checkpoint_conditioning(checkpoint_a)
    conditioning_b = checkpoint_conditioning(checkpoint_b)
    condition_by_roi_a = apply_input_conditioning(
        data_root / "train" / "DAPI",
        val_names,
        conditioning_a,
    )
    condition_by_roi_b = apply_input_conditioning(
        data_root / "train" / "DAPI",
        val_names,
        conditioning_b,
    )
    slope, intercept = fit_roi_mean_calibration(data_root, args.marker, train_names)
    loader = DataLoader(
        PairedPatchDataset(
            data_root,
            val_names,
            markers=(args.marker,),
            augment=False,
            context_halo=args.context_halo,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    started = time.perf_counter()
    sources, prediction_a, prediction_b, targets, predicted_names = predict_pair(
        model_a,
        index_a,
        model_b,
        index_b,
        loader,
        device,
        args.tta,
        condition_by_roi_a,
        condition_by_roi_b,
    )
    weights = [float(item.strip()) for item in args.weights.split(",")]
    results = []
    roi_metrics = {}
    for weight in weights:
        blended = prediction_a * (1.0 - weight) + prediction_b * weight
        scales, _ = roi_scale_multipliers(
            predicted_names, sources, blended, slope, intercept
        )
        variants = {
            "raw": blended,
            "linear_half": blended * (1.0 + 0.5 * (scales - 1.0)),
            "log_half": blended * np.sqrt(scales),
            "log_0.75": blended * np.power(scales, 0.75),
        }
        for calibration, candidate in variants.items():
            name = f"weight_b_{weight:g}_{calibration}"
            metrics = evaluate(candidate, targets, args.jpeg_quality)
            results.append({"name": name, "weight_b": weight, "calibration": calibration, **metrics})
            if calibration in {"raw", "log_half", "log_0.75"}:
                roi_metrics[name] = evaluate_by_roi(
                    candidate,
                    targets,
                    predicted_names,
                    args.jpeg_quality,
                )
            print(json.dumps(results[-1]), flush=True)

    ranking_key = (
        "competition_score_mean_image_psnr"
        if args.ranking_psnr_aggregation == "mean-image"
        else "competition_score_global_mse_psnr"
    )
    results.sort(key=lambda item: item[ranking_key], reverse=True)
    report = {
        "checkpoint_a": str(args.checkpoint_a.resolve()),
        "checkpoint_b": str(args.checkpoint_b.resolve()),
        "marker": args.marker,
        "validation_rois": sorted(val_rois),
        "validation_images": len(val_names),
        "context_halo": args.context_halo,
        "tta": args.tta,
        "jpeg_quality": args.jpeg_quality,
        "ranking_psnr_aggregation": args.ranking_psnr_aggregation,
        "ranking_key": ranking_key,
        "conditioning_a": conditioning_a,
        "conditioning_b": conditioning_b,
        "roi_fit": {"slope": slope, "intercept": intercept},
        "ranking": results,
        "roi_metrics": roi_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
