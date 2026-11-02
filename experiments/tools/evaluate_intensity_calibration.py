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


def add_metric(
    totals: dict[str, dict[str, float]],
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    totals[name]["ssim"] += float(
        structural_similarity(target, prediction, data_range=255)
    )
    totals[name]["psnr"] += float(
        peak_signal_noise_ratio(target, prediction, data_range=255)
    )
    totals[name]["count"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate input-only patch and ROI intensity calibration."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--val-rois", default="ROI020,ROI021,ROI022,ROI023,ROI024")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
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
    markers = tuple(checkpoint["markers"])
    if len(markers) != 1:
        raise ValueError("Intensity calibration currently expects a single-marker checkpoint.")
    marker = markers[0]
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
    all_names = image_names(data_root / "train" / "DAPI")
    train_names = [name for name in all_names if parse_patch_name(name).roi not in val_rois]
    val_names = [name for name in all_names if parse_patch_name(name).roi in val_rois]
    condition_by_roi = apply_input_conditioning(
        data_root / "train" / "DAPI",
        val_names,
        conditioning,
    )

    patch_source_means: list[float] = []
    patch_target_means: list[float] = []
    roi_train: dict[str, dict[str, float]] = defaultdict(
        lambda: {"source_sum": 0.0, "target_sum": 0.0, "pixels": 0.0}
    )
    for name in train_names:
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / marker / name)
        patch_source_means.append(float(source.mean()))
        patch_target_means.append(float(target.mean()))
        roi = parse_patch_name(name).roi
        roi_train[roi]["source_sum"] += float(source.sum())
        roi_train[roi]["target_sum"] += float(target.sum())
        roi_train[roi]["pixels"] += source.size

    patch_fit = linear_fit(patch_source_means, patch_target_means)
    roi_source_means = [value["source_sum"] / value["pixels"] for value in roi_train.values()]
    roi_target_means = [value["target_sum"] / value["pixels"] for value in roi_train.values()]
    roi_fit = linear_fit(roi_source_means, roi_target_means)

    dataset = PairedPatchDataset(
        data_root,
        val_names,
        markers=markers,
        augment=False,
        context_halo=context_halo,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    roi_val: dict[str, dict[str, float]] = defaultdict(
        lambda: {"source_sum": 0.0, "prediction_sum": 0.0, "pixels": 0.0}
    )
    with torch.no_grad():
        for source, target, names in loader:
            condition = batch_condition_tensor(names, condition_by_roi, device)
            prediction = center_crop_like(
                predict_with_tta(model, source.to(device), args.tta, condition),
                target,
            ).cpu().numpy()[:, 0] * 255.0
            source_np = center_crop_like(source, target).numpy()[:, 0] * 255.0
            for index, name in enumerate(names):
                roi = parse_patch_name(name).roi
                roi_val[roi]["source_sum"] += float(source_np[index].sum())
                roi_val[roi]["prediction_sum"] += float(prediction[index].sum())
                roi_val[roi]["pixels"] += source_np[index].size

    roi_adjustments = {}
    for roi, values in roi_val.items():
        source_mean = values["source_sum"] / values["pixels"]
        prediction_mean = values["prediction_sum"] / values["pixels"]
        expected_mean = roi_fit[0] * source_mean + roi_fit[1]
        roi_adjustments[roi] = {
            "shift": expected_mean - prediction_mean,
            "scale": expected_mean / max(prediction_mean, 1e-6),
        }

    totals: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"ssim": 0.0, "psnr": 0.0, "count": 0}
    )
    with torch.no_grad():
        for source, target, names in loader:
            condition = batch_condition_tensor(names, condition_by_roi, device)
            prediction = center_crop_like(
                predict_with_tta(model, source.to(device), args.tta, condition),
                target,
            ).cpu().numpy()[:, 0] * 255.0
            source_np = center_crop_like(source, target).numpy()[:, 0] * 255.0
            target_np = target.numpy()[:, 0] * 255.0
            for index, name in enumerate(names):
                raw = prediction[index]
                truth = target_np[index]
                expected_patch_mean = patch_fit[0] * source_np[index].mean() + patch_fit[1]
                patch_shift = raw + (expected_patch_mean - raw.mean())
                patch_scale = raw * (expected_patch_mean / max(float(raw.mean()), 1e-6))
                roi = parse_patch_name(name).roi
                candidates = {
                    "raw": raw,
                    "patch_shift": patch_shift,
                    "patch_scale": patch_scale,
                    "roi_shift": raw + roi_adjustments[roi]["shift"],
                    "roi_scale": raw * roi_adjustments[roi]["scale"],
                }
                for variant, candidate in candidates.items():
                    clipped = np.clip(candidate, 0, 255).astype(np.float32)
                    add_metric(totals, variant, clipped, truth)
                    add_metric(
                        totals,
                        f"{variant}_jpeg_q{args.jpeg_quality}",
                        jpeg_roundtrip(clipped, args.jpeg_quality),
                        truth,
                    )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "marker": marker,
        "validation_rois": sorted(val_rois),
        "validation_images": len(val_names),
        "tta": args.tta,
        "context_halo": context_halo,
        "patch_mean_fit": {"slope": patch_fit[0], "intercept": patch_fit[1]},
        "roi_mean_fit": {"slope": roi_fit[0], "intercept": roi_fit[1]},
        "metrics": {
            variant: {
                "ssim": values["ssim"] / values["count"],
                "psnr": values["psnr"] / values["count"],
                "competition_score": competition_score(
                    values["ssim"] / values["count"],
                    values["psnr"] / values["count"],
                ),
            }
            for variant, values in totals.items()
        },
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
