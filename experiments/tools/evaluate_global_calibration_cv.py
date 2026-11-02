from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import parse_patch_name
from ihc_torch import competition_score
from refiner_protocol import (
    load_base_protocol,
    load_refiner_protocol,
    resolve_project_path,
    sha256_file,
    validate_cache_manifest,
)
from search_postprocessing import evaluate


@dataclass(frozen=True)
class RegressionMoments:
    count: int
    prediction_mean: float
    target_mean: float
    prediction_ss: float
    prediction_target_ss: float

    def merge(self, other: "RegressionMoments") -> "RegressionMoments":
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        count = self.count + other.count
        prediction_delta = other.prediction_mean - self.prediction_mean
        target_delta = other.target_mean - self.target_mean
        cross_weight = self.count * other.count / count
        return RegressionMoments(
            count=count,
            prediction_mean=(
                self.count * self.prediction_mean + other.count * other.prediction_mean
            )
            / count,
            target_mean=(
                self.count * self.target_mean + other.count * other.target_mean
            )
            / count,
            prediction_ss=(
                self.prediction_ss
                + other.prediction_ss
                + prediction_delta * prediction_delta * cross_weight
            ),
            prediction_target_ss=(
                self.prediction_target_ss
                + other.prediction_target_ss
                + prediction_delta * target_delta * cross_weight
            ),
        )


EMPTY_MOMENTS = RegressionMoments(0, 0.0, 0.0, 0.0, 0.0)


def moments_from_arrays(
    prediction: np.ndarray, target: np.ndarray
) -> RegressionMoments:
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction and target must have the same non-empty shape.")
    prediction64 = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction_mean = float(np.mean(prediction64))
    target_mean = float(np.mean(target64))
    centered_prediction = prediction64 - prediction_mean
    centered_target = target64 - target_mean
    return RegressionMoments(
        count=int(prediction64.size),
        prediction_mean=prediction_mean,
        target_mean=target_mean,
        prediction_ss=float(np.dot(centered_prediction, centered_prediction)),
        prediction_target_ss=float(np.dot(centered_prediction, centered_target)),
    )


def merge_moments(moments: Iterable[RegressionMoments]) -> RegressionMoments:
    merged = EMPTY_MOMENTS
    for item in moments:
        merged = merged.merge(item)
    if merged.count == 0:
        raise ValueError("At least one non-empty moment set is required.")
    return merged


def fit_affine(moments: RegressionMoments) -> tuple[float, float]:
    if moments.count <= 1 or moments.prediction_ss <= 0:
        raise ValueError("Affine calibration requires non-constant predictions.")
    slope = moments.prediction_target_ss / moments.prediction_ss
    intercept = moments.target_mean - slope * moments.prediction_mean
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Affine calibration produced non-finite parameters.")
    return float(slope), float(intercept)


def cross_fitted_parameters(
    fold_moments: Mapping[str, RegressionMoments]
) -> dict[str, tuple[float, float]]:
    if len(fold_moments) < 2:
        raise ValueError("Cross-fitting requires at least two folds.")
    return {
        holdout: fit_affine(
            merge_moments(
                moments for fold, moments in fold_moments.items() if fold != holdout
            )
        )
        for holdout in fold_moments
    }


def apply_affine(
    prediction: np.ndarray,
    slope: float,
    intercept: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    if not clip_min < clip_max:
        raise ValueError("Calibration clipping bounds must be increasing.")
    calibrated = prediction.astype(np.float32, copy=True)
    calibrated *= np.float32(slope)
    calibrated += np.float32(intercept)
    np.clip(calibrated, clip_min, clip_max, out=calibrated)
    return calibrated


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


def validate_protocol(protocol: object) -> dict[str, object]:
    if not isinstance(protocol, dict) or protocol.get("version") != 1:
        raise ValueError("Global calibration protocol version must be 1.")
    for field in ("name", "marker", "cache_protocol"):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")
    fit = protocol.get("fit")
    if not isinstance(fit, dict) or fit.get("method") != "global-affine-ols":
        raise ValueError("fit.method must be global-affine-ols.")
    clip_min = _finite_number(fit.get("clip_min"), "fit.clip_min")
    clip_max = _finite_number(fit.get("clip_max"), "fit.clip_max")
    if not clip_min < clip_max:
        raise ValueError("fit clipping bounds must be increasing.")
    chunk_images = fit.get("chunk_images")
    if type(chunk_images) is not int or chunk_images <= 0:
        raise ValueError("fit.chunk_images must be a positive integer.")
    inference = protocol.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference must be an object.")
    quality = inference.get("jpeg_quality")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("inference.jpeg_quality must be in [1, 100].")
    gate = protocol.get("promotion_gate")
    if not isinstance(gate, dict):
        raise ValueError("promotion_gate must be an object.")
    for field in (
        "minimum_overall_q95_gain",
        "minimum_float_gain",
        "minimum_roi_q95_gain",
    ):
        _finite_number(gate.get(field), f"promotion_gate.{field}")
    minimum_folds = gate.get("minimum_positive_folds")
    if type(minimum_folds) is not int or not 1 <= minimum_folds <= 5:
        raise ValueError("promotion_gate.minimum_positive_folds must be in [1, 5].")
    maximum_negative = gate.get("maximum_negative_rois")
    if type(maximum_negative) is not int or maximum_negative < 0:
        raise ValueError("promotion_gate.maximum_negative_rois must be non-negative.")
    return protocol


def load_protocol(path: Path) -> dict[str, object]:
    return validate_protocol(json.loads(path.read_text(encoding="utf-8")))


def weighted_baseline(
    fold_dir: Path,
    recipe_order: list[str],
    weights: Mapping[str, float],
) -> np.ndarray:
    first = np.load(fold_dir / f"prediction_{recipe_order[0]}.npy", mmap_mode="r")
    baseline = np.asarray(first, dtype=np.float32) * np.float32(
        weights[recipe_order[0]]
    )
    for recipe in recipe_order[1:]:
        prediction = np.load(fold_dir / f"prediction_{recipe}.npy", mmap_mode="r")
        baseline += np.asarray(prediction, dtype=np.float32) * np.float32(
            weights[recipe]
        )
    return baseline


def fold_regression_moments(
    fold_dir: Path,
    recipe_order: list[str],
    weights: Mapping[str, float],
    chunk_images: int,
) -> RegressionMoments:
    predictions = {
        recipe: np.load(fold_dir / f"prediction_{recipe}.npy", mmap_mode="r")
        for recipe in recipe_order
    }
    target = np.load(fold_dir / "target.npy", mmap_mode="r")
    moments = EMPTY_MOMENTS
    for start in range(0, len(target), chunk_images):
        stop = min(start + chunk_images, len(target))
        baseline = sum(
            np.asarray(predictions[recipe][start:stop], dtype=np.float32)
            * np.float32(weights[recipe])
            for recipe in recipe_order
        )
        moments = moments.merge(moments_from_arrays(baseline, target[start:stop]))
    return moments


def official_delta(
    candidate: Mapping[str, float], baseline: Mapping[str, float]
) -> float:
    return float(candidate["competition_score_official"]) - float(
        baseline["competition_score_official"]
    )


def aggregate_metrics(
    weighted_metrics: Iterable[tuple[int, Mapping[str, float]]]
) -> dict[str, float]:
    items = list(weighted_metrics)
    total_images = sum(images for images, _ in items)
    if total_images <= 0:
        raise ValueError("Metric aggregation requires at least one image.")

    def weighted_mean(field: str) -> float:
        return float(
            sum(images * float(metrics[field]) for images, metrics in items)
            / total_images
        )

    ssim = weighted_mean("ssim")
    psnr = weighted_mean("psnr_mean_image")
    official_score = weighted_mean("competition_score_official")
    normalized_psnr = weighted_mean("normalized_psnr_mean")
    normalized_global_mse = (
        sum(
            images
            * (
                0.0
                if math.isinf(float(metrics["psnr_global_mse"]))
                else 10.0 ** (-float(metrics["psnr_global_mse"]) / 10.0)
            )
            for images, metrics in items
        )
        / total_images
    )
    psnr_global_mse = (
        float("inf")
        if normalized_global_mse == 0
        else float(-10.0 * math.log10(normalized_global_mse))
    )
    return {
        "ssim": ssim,
        "psnr": psnr,
        "competition_score": official_score,
        "competition_score_official": official_score,
        "normalized_psnr_mean": normalized_psnr,
        "psnr_mean_image": psnr,
        "competition_score_mean_image_psnr": competition_score(ssim, psnr),
        "psnr_global_mse": psnr_global_mse,
        "competition_score_global_mse_psnr": competition_score(ssim, psnr_global_mse),
    }


def aggregate_variant_metrics(
    entries: Iterable[Mapping[str, object]], quality_key: str
) -> dict[str, object]:
    items = list(entries)
    return {
        "float": aggregate_metrics(
            (int(item["images"]), item["float"]) for item in items
        ),
        quality_key: aggregate_metrics(
            (int(item["images"]), item[quality_key]) for item in items
        ),
    }


def apply_gate(
    protocol: dict[str, object],
    overall: dict[str, object],
    fold_metrics: Mapping[str, object],
    roi_metrics: Mapping[str, object],
) -> dict[str, object]:
    quality_key = f"jpeg_q{protocol['inference']['jpeg_quality']}"
    overall_q95_gain = official_delta(
        overall["calibrated"][quality_key], overall["baseline"][quality_key]
    )
    overall_float_gain = official_delta(
        overall["calibrated"]["float"], overall["baseline"]["float"]
    )
    fold_deltas = {
        fold: official_delta(
            metrics["calibrated"][quality_key],
            metrics["baseline"][quality_key],
        )
        for fold, metrics in fold_metrics.items()
    }
    roi_deltas = {
        roi: official_delta(
            metrics["calibrated"][quality_key],
            metrics["baseline"][quality_key],
        )
        for roi, metrics in roi_metrics.items()
    }
    positive_folds = sorted(fold for fold, delta in fold_deltas.items() if delta > 0)
    negative_rois = sorted(roi for roi, delta in roi_deltas.items() if delta < 0)
    gate = protocol["promotion_gate"]
    checks = {
        "overall_q95": overall_q95_gain >= float(gate["minimum_overall_q95_gain"]),
        "overall_float": overall_float_gain > float(gate["minimum_float_gain"]),
        "folds": len(positive_folds) >= int(gate["minimum_positive_folds"]),
        "roi_floor": min(roi_deltas.values()) >= float(gate["minimum_roi_q95_gain"]),
        "roi_count": len(negative_rois) <= int(gate["maximum_negative_rois"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overall_q95_gain": overall_q95_gain,
        "overall_float_gain": overall_float_gain,
        "fold_q95_deltas": fold_deltas,
        "positive_folds": positive_folds,
        "roi_q95_deltas": roi_deltas,
        "negative_rois": negative_rois,
        "minimum_roi_q95_gain": min(roi_deltas.values()),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit one analytic global affine calibration over cached OOF "
            "predictions."
        )
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    cache_protocol_path = resolve_project_path(
        protocol["cache_protocol"], "cache_protocol"
    )
    cache_protocol = load_refiner_protocol(cache_protocol_path)
    base_protocol = load_base_protocol(cache_protocol)
    if protocol["marker"] != cache_protocol["marker"]:
        raise ValueError("Calibration and cache markers differ.")
    cache_root = args.cache_root.resolve()
    cache_manifest = validate_cache_manifest(
        cache_root,
        cache_protocol_path,
        cache_protocol,
        base_protocol,
    )
    fold_names = [str(fold["name"]) for fold in base_protocol["folds"]]
    recipe_order = list(cache_protocol["recipe_order"])
    weights = dict(cache_protocol["baseline_weights"])
    fit = protocol["fit"]
    fold_moments = {
        fold: fold_regression_moments(
            cache_root / fold,
            recipe_order,
            weights,
            int(fit["chunk_images"]),
        )
        for fold in fold_names
    }
    parameters = cross_fitted_parameters(fold_moments)
    deployment_slope, deployment_intercept = fit_affine(
        merge_moments(fold_moments.values())
    )

    all_names: list[str] = []
    fold_metrics: dict[str, object] = {}
    roi_metrics: dict[str, object] = {}
    quality = int(protocol["inference"]["jpeg_quality"])
    quality_key = f"jpeg_q{quality}"
    for fold in fold_names:
        fold_dir = cache_root / fold
        manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        names = list(manifest["names"])
        baseline = weighted_baseline(fold_dir, recipe_order, weights)
        target = np.asarray(
            np.load(fold_dir / "target.npy", mmap_mode="r"), dtype=np.uint8
        )
        slope, intercept = parameters[fold]
        calibrated = apply_affine(
            baseline,
            slope,
            intercept,
            float(fit["clip_min"]),
            float(fit["clip_max"]),
        )
        fold_roi_metrics: list[dict[str, object]] = []
        for roi in sorted(set(manifest["val_rois"])):
            indices = np.asarray(
                [
                    index
                    for index, name in enumerate(names)
                    if parse_patch_name(name).roi == roi
                ],
                dtype=np.int64,
            )
            roi_entry = {
                "images": int(len(indices)),
                "baseline": {
                    "float": evaluate(baseline[indices], target[indices], 0),
                    quality_key: evaluate(baseline[indices], target[indices], quality),
                },
                "calibrated": {
                    "float": evaluate(calibrated[indices], target[indices], 0),
                    quality_key: evaluate(
                        calibrated[indices], target[indices], quality
                    ),
                },
            }
            roi_metrics[roi] = roi_entry
            fold_roi_metrics.append(roi_entry)
        fold_metrics[fold] = {
            "images": len(names),
            "fit_training_folds": [name for name in fold_names if name != fold],
            "parameters": {"slope": slope, "intercept": intercept},
            "baseline": aggregate_variant_metrics(
                (
                    entry["baseline"] | {"images": entry["images"]}
                    for entry in fold_roi_metrics
                ),
                quality_key,
            ),
            "calibrated": aggregate_variant_metrics(
                (
                    entry["calibrated"] | {"images": entry["images"]}
                    for entry in fold_roi_metrics
                ),
                quality_key,
            ),
        }
        all_names.extend(names)
        print(json.dumps({"completed_fold": fold, "images": len(names)}), flush=True)

    if len(all_names) != len(set(all_names)):
        raise RuntimeError("OOF cache contains duplicate image names.")
    overall = {
        "baseline": aggregate_variant_metrics(
            (
                metrics["baseline"] | {"images": metrics["images"]}
                for metrics in roi_metrics.values()
            ),
            quality_key,
        ),
        "calibrated": aggregate_variant_metrics(
            (
                metrics["calibrated"] | {"images": metrics["images"]}
                for metrics in roi_metrics.values()
            ),
            quality_key,
        ),
    }
    rois = sorted({parse_patch_name(name).roi for name in all_names})
    if set(roi_metrics) != set(rois):
        raise RuntimeError("OOF cache ROI coverage is incomplete.")
    gate = apply_gate(protocol, overall, fold_metrics, roi_metrics)
    report = {
        "version": 1,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_name": protocol["name"],
        "marker": protocol["marker"],
        "cache_protocol": str(cache_protocol_path),
        "cache_manifest": str((cache_root / "manifest.json").resolve()),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "cache_protocol_name": cache_manifest["protocol_name"],
        "base_protocol_name": base_protocol["name"],
        "base_candidate": cache_protocol["base_candidate"],
        "validation_images": len(all_names),
        "validation_rois": rois,
        "fit": {
            "method": fit["method"],
            "units": "uint8 intensity levels before JPEG encoding",
            "clip": [fit["clip_min"], fit["clip_max"]],
            "cross_fitted_parameters": {
                fold: {"slope": slope, "intercept": intercept}
                for fold, (slope, intercept) in parameters.items()
            },
            "deployment_parameters_from_all_oof": {
                "slope": deployment_slope,
                "intercept": deployment_intercept,
            },
        },
        "official_metric": {
            "score": "mean_i(100*(0.7*SSIM_i+0.3*clip(PSNR_i/50,0,1)))",
            "aggregation": "per-image then mean",
            "jpeg_quality": quality,
        },
        "overall": overall,
        "fold_metrics": fold_metrics,
        "roi_metrics": roi_metrics,
        "promotion_gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
