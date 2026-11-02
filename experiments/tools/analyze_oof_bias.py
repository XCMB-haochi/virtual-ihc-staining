from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_output_encoding import decode_jpeg
from ihc_data import parse_patch_name


FEATURE_NAMES = (
    "dapi_mean",
    "dapi_std",
    "dapi_q05",
    "dapi_q50",
    "dapi_q95",
    "dapi_low_fraction",
    "dapi_gradient_mean",
    "prediction_mean",
    "prediction_std",
    "prediction_q05",
    "prediction_q95",
    "prediction_dapi_mean_delta",
    "prediction_dapi_std_ratio",
    "row_norm",
    "col_norm",
)


def image_features(dapi: np.ndarray, prediction: np.ndarray, names: list[str]) -> np.ndarray:
    dapi = dapi.astype(np.float32)
    prediction = prediction.astype(np.float32)
    vertical = np.abs(np.diff(dapi, axis=1)).mean(axis=(1, 2))
    horizontal = np.abs(np.diff(dapi, axis=2)).mean(axis=(1, 2))
    dapi_mean = dapi.mean(axis=(1, 2))
    dapi_std = dapi.std(axis=(1, 2))
    prediction_mean = prediction.mean(axis=(1, 2))
    prediction_std = prediction.std(axis=(1, 2))
    rows = np.asarray([parse_patch_name(name).row for name in names], dtype=np.float64)
    cols = np.asarray([parse_patch_name(name).col for name in names], dtype=np.float64)
    return np.stack(
        (
            dapi_mean,
            dapi_std,
            np.quantile(dapi, 0.05, axis=(1, 2)),
            np.quantile(dapi, 0.50, axis=(1, 2)),
            np.quantile(dapi, 0.95, axis=(1, 2)),
            np.mean(dapi <= 5.0, axis=(1, 2)),
            0.5 * (vertical + horizontal),
            prediction_mean,
            prediction_std,
            np.quantile(prediction, 0.05, axis=(1, 2)),
            np.quantile(prediction, 0.95, axis=(1, 2)),
            prediction_mean - dapi_mean,
            prediction_std / np.maximum(dapi_std, 1e-3),
            rows / 31.0,
            cols / 31.0,
        ),
        axis=1,
    ).astype(np.float64)


def score_arrays(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from ihc_torch import competition_score

    ssim_sum = psnr_sum = score_sum = 0.0
    for pred, truth in zip(prediction, target):
        decoded = decode_jpeg(pred, 95)
        ssim = float(structural_similarity(truth, decoded, data_range=255, win_size=7))
        psnr = float(peak_signal_noise_ratio(truth, decoded, data_range=255))
        ssim_sum += ssim
        psnr_sum += psnr
        score_sum += competition_score(ssim, psnr)
    n = len(prediction)
    return {"images": n, "ssim": ssim_sum / n, "psnr": psnr_sum / n, "competition_score_official": score_sum / n}


def load_cache(root: Path, max_images: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[str]]]:
    dapi_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    name_parts: list[list[str]] = []
    for fold_dir in sorted(root.glob("fold*/")):
        manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        names = list(manifest["names"])
        indices = np.arange(len(names), dtype=np.int64)
        if max_images > 0 and len(indices) > max_images:
            indices = np.linspace(0, len(indices) - 1, max_images, dtype=np.int64)
        dapi = np.load(fold_dir / "dapi.npy", mmap_mode="r")
        target = np.load(fold_dir / "target.npy", mmap_mode="r")
        first = np.load(fold_dir / "prediction_thumbnailbase32.npy", mmap_mode="r")
        second = np.load(fold_dir / "prediction_thumbnailbase48continue.npy", mmap_mode="r")
        dapi_parts.append(np.asarray(dapi[indices], dtype=np.uint8))
        target_parts.append(np.asarray(target[indices], dtype=np.uint8))
        prediction_parts.append(
            0.7 * np.asarray(first[indices], dtype=np.float32)
            + 0.3 * np.asarray(second[indices], dtype=np.float32)
        )
        name_parts.append([names[int(index)] for index in indices])
    if len(dapi_parts) != 5:
        raise RuntimeError(f"Expected five cache folds, found {len(dapi_parts)}")
    return dapi_parts, target_parts, prediction_parts, name_parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fit public-feature prediction-bias diagnostic.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--max-images-per-fold", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dapi, target, prediction, names = load_cache(args.cache_root, args.max_images_per_fold)
    features = [image_features(x, p, n) for x, p, n in zip(dapi, prediction, names)]
    target_mean = [x.mean(axis=(1, 2)).astype(np.float64) for x in target]
    prediction_mean = [x.mean(axis=(1, 2)).astype(np.float64) for x in prediction]
    target_std = [x.std(axis=(1, 2)).astype(np.float64) for x in target]
    prediction_std = [x.std(axis=(1, 2)).astype(np.float64) for x in prediction]
    residual_mean = [y - p for y, p in zip(target_mean, prediction_mean)]
    log_std_ratio = [np.log(np.maximum(y, 1e-3) / np.maximum(p, 1e-3)) for y, p in zip(target_std, prediction_std)]

    correlations = {}
    x_all = np.concatenate(features)
    y_mean_all = np.concatenate(residual_mean)
    y_ratio_all = np.concatenate(log_std_ratio)
    for index, name in enumerate(FEATURE_NAMES):
        correlations[name] = {
            "residual_mean_pearson": float(pearsonr(x_all[:, index], y_mean_all).statistic),
            "log_std_ratio_pearson": float(pearsonr(x_all[:, index], y_ratio_all).statistic),
        }

    baseline_scores: list[dict[str, float]] = []
    mean_scores: list[dict[str, float]] = []
    affine_scores: list[dict[str, float]] = []
    oracle_mean_scores: list[dict[str, float]] = []
    oracle_affine_scores: list[dict[str, float]] = []
    crossfit_details = []
    for holdout in range(5):
        train_indices = [i for i in range(5) if i != holdout]
        model_mean = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model_mean.fit(np.concatenate([features[i] for i in train_indices]), np.concatenate([residual_mean[i] for i in train_indices]))
        model_ratio = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model_ratio.fit(np.concatenate([features[i] for i in train_indices]), np.concatenate([log_std_ratio[i] for i in train_indices]))
        x_test = features[holdout]
        delta = np.clip(model_mean.predict(x_test), -8.0, 8.0)
        scale = np.clip(np.exp(model_ratio.predict(x_test)), 0.75, 1.35)
        p = prediction[holdout]
        y = target[holdout]
        p_mean = p.mean(axis=(1, 2), keepdims=True)
        corrected_mean = p + delta[:, None, None]
        corrected_affine = (p - p_mean) * scale[:, None, None] + (p_mean + delta[:, None, None])
        y_mean = y.mean(axis=(1, 2), keepdims=True)
        y_std = y.std(axis=(1, 2), keepdims=True)
        p_std = np.maximum(p.std(axis=(1, 2), keepdims=True), 1e-3)
        oracle_mean = p + (y_mean - p_mean)
        oracle_affine = (p - p_mean) * (y_std / p_std) + y_mean
        baseline_scores.append(score_arrays(p, y))
        mean_scores.append(score_arrays(corrected_mean, y))
        affine_scores.append(score_arrays(corrected_affine, y))
        oracle_mean_scores.append(score_arrays(oracle_mean, y))
        oracle_affine_scores.append(score_arrays(oracle_affine, y))
        crossfit_details.append(
            {
                "fold": f"fold{holdout}",
                "images": int(len(y)),
                "predicted_delta_mean": float(delta.mean()),
                "predicted_scale_mean": float(scale.mean()),
                "target_residual_mean": float(residual_mean[holdout].mean()),
                "target_log_std_ratio_mean": float(log_std_ratio[holdout].mean()),
            }
        )

    def average(rows: list[dict[str, float]]) -> dict[str, float]:
        total = sum(row["images"] for row in rows)
        return {
            key: float(sum(row[key] * row["images"] for row in rows) / total)
            for key in rows[0]
            if key != "images"
        } | {"images": total}

    report = {
        "version": 1,
        "method": "five-fold-crossfit-ridge-patch-mean-and-contrast-correction",
        "cache": str(args.cache_root.resolve()),
        "images": int(sum(len(x) for x in target)),
        "fixed_model": "0.7*thumbnailbase32 + 0.3*thumbnailbase48continue",
        "feature_names": list(FEATURE_NAMES),
        "ridge": {"alpha": 100.0, "mean_clip": [-8.0, 8.0], "scale_clip": [0.75, 1.35]},
        "correlations": correlations,
        "folds": crossfit_details,
        "overall": {
            "baseline": average(baseline_scores),
            "crossfit_mean_only": average(mean_scores),
            "crossfit_mean_contrast": average(affine_scores),
            "target_mean_oracle": average(oracle_mean_scores),
            "target_mean_contrast_oracle": average(oracle_affine_scores),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
