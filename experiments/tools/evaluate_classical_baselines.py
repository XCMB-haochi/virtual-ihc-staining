from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import (
    MARKERS,
    choose_evenly,
    image_names,
    load_gray,
    resolve_data_root,
    split_names_by_roi,
)


def features(image: np.ndarray) -> np.ndarray:
    blur_1 = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    blur_3 = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0)
    patch_mean = np.full_like(image, image.mean())
    ones = np.ones_like(image)
    return np.stack((image, blur_1, blur_3, patch_mean, ones), axis=-1)


def fit_marker(
    data_root: Path,
    marker: str,
    train_names: list[str],
    max_fit_images: int,
    pixels_per_image: int,
    seed: int,
) -> dict[str, object]:
    selected = choose_evenly(train_names, max_fit_images)
    rng = np.random.default_rng(seed)
    xtx = np.zeros((5, 5), dtype=np.float64)
    xty = np.zeros(5, dtype=np.float64)
    x_hist = np.zeros(256, dtype=np.int64)
    y_hist = np.zeros(256, dtype=np.int64)
    conditional_target_sum = np.zeros(256, dtype=np.float64)
    y_sum = 0.0
    y_count = 0

    for name in selected:
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / marker / name)
        flat_features = features(source).reshape(-1, 5)
        flat_target = target.reshape(-1)
        count = min(pixels_per_image, flat_target.size)
        indices = rng.choice(flat_target.size, size=count, replace=False)
        sampled_x = flat_features[indices].astype(np.float64)
        sampled_y = flat_target[indices].astype(np.float64)
        xtx += sampled_x.T @ sampled_x
        xty += sampled_x.T @ sampled_y
        source_u8 = source.astype(np.uint8).ravel()
        target_flat = target.ravel()
        x_hist += np.bincount(source_u8, minlength=256)
        y_hist += np.bincount(target.astype(np.uint8).ravel(), minlength=256)
        conditional_target_sum += np.bincount(
            source_u8, weights=target_flat, minlength=256
        )
        y_sum += float(target.sum())
        y_count += target.size

    scale = max(float(np.trace(xtx)) / xtx.shape[0], 1.0)
    coefficients = np.linalg.solve(xtx + np.eye(5) * scale * 1e-8, xty)

    source_cdf = np.cumsum(x_hist) / max(x_hist.sum(), 1)
    target_cdf = np.cumsum(y_hist) / max(y_hist.sum(), 1)
    quantile_lut = np.searchsorted(target_cdf, source_cdf, side="left").clip(0, 255)
    observed = x_hist > 0
    conditional_lut = np.zeros(256, dtype=np.float64)
    conditional_lut[observed] = conditional_target_sum[observed] / x_hist[observed]
    missing = np.flatnonzero(~observed)
    if missing.size:
        present = np.flatnonzero(observed)
        conditional_lut[missing] = np.interp(missing, present, conditional_lut[present])

    return {
        "mean": y_sum / max(y_count, 1),
        "coefficients": coefficients,
        "quantile_lut": quantile_lut.astype(np.float32),
        "conditional_lut": conditional_lut.astype(np.float32),
        "fit_images": len(selected),
    }


def predictions(source: np.ndarray, model: dict[str, object]) -> dict[str, np.ndarray]:
    coefficients = np.asarray(model["coefficients"])
    linear = features(source) @ coefficients
    lut = np.asarray(model["quantile_lut"])
    conditional_lut = np.asarray(model["conditional_lut"])
    return {
        "zero": np.zeros_like(source),
        "constant_mean": np.full_like(source, float(model["mean"])),
        "identity": source,
        "quantile_map": lut[source.astype(np.uint8)],
        "conditional_mean_lut": conditional_lut[source.astype(np.uint8)],
        "multiscale_linear": np.clip(linear, 0, 255),
    }


def evaluate_marker(
    data_root: Path,
    marker: str,
    val_names: list[str],
    model: dict[str, object],
    max_val_images: int,
) -> dict[str, object]:
    selected = choose_evenly(val_names, max_val_images)
    totals: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"ssim": 0.0, "psnr": 0.0}
    )

    for name in selected:
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / marker / name)
        for baseline, prediction in predictions(source, model).items():
            prediction = prediction.astype(np.float32)
            totals[baseline]["ssim"] += float(
                structural_similarity(target, prediction, data_range=255)
            )
            totals[baseline]["psnr"] += float(
                peak_signal_noise_ratio(target, prediction, data_range=255)
            )

    return {
        "validation_images": len(selected),
        "metrics": {
            name: {
                metric: value / len(selected) for metric, value in values.items()
            }
            for name, values in totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cheap grayscale baselines with an ROI-held-out split."
    )
    parser.add_argument("dataset", type=Path, help="Dataset directory or its parent.")
    parser.add_argument(
        "--val-rois",
        default="ROI020,ROI021,ROI022,ROI023,ROI024",
        help="Comma-separated ROI identifiers reserved for validation.",
    )
    parser.add_argument("--max-fit-images", type=int, default=1000)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--pixels-per-image", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_root = resolve_data_root(args.dataset)
    names = image_names(data_root / "train" / "DAPI")
    val_rois = {item.strip().upper() for item in args.val_rois.split(",") if item.strip()}
    train_names, val_names = split_names_by_roi(names, val_rois)

    result: dict[str, object] = {
        "data_root": str(data_root),
        "validation_rois": sorted(val_rois),
        "training_images": len(train_names),
        "validation_images": len(val_names),
        "note": "Official Normalize(PSNR) is undefined; raw SSIM and PSNR are reported.",
        "markers": {},
    }
    for index, marker in enumerate(MARKERS):
        model = fit_marker(
            data_root,
            marker,
            train_names,
            args.max_fit_images,
            args.pixels_per_image,
            args.seed + index,
        )
        marker_result = evaluate_marker(
            data_root, marker, val_names, model, args.max_val_images
        )
        marker_result["fit_images"] = model["fit_images"]
        marker_result["linear_coefficients"] = np.asarray(
            model["coefficients"]
        ).round(6).tolist()
        result["markers"][marker] = marker_result

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
