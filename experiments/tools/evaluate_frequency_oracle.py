from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

from search_postprocessing import jpeg_roundtrip


METHODS = (
    "baseline",
    "crossfit_global_unsharp",
    "per_image_mse_oracle",
    "dense_local_variance_oracle",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Frequency-oracle protocol version must be 1.")
    for field in ("name", "cache_root", "cache_manifest_sha256"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"{field} must be a non-empty string.")
    fold_hashes = payload.get("fold_manifest_sha256")
    expected_folds = {f"fold{index}" for index in range(5)}
    if not isinstance(fold_hashes, dict) or set(fold_hashes) != expected_folds:
        raise ValueError("fold_manifest_sha256 must define fold0 through fold4.")
    recipe_order = payload.get("recipe_order")
    weights = payload.get("baseline_weights")
    if not isinstance(recipe_order, list) or len(recipe_order) < 1:
        raise ValueError("recipe_order must be a non-empty list.")
    if not isinstance(weights, dict) or set(weights) != set(recipe_order):
        raise ValueError("baseline_weights must match recipe_order.")
    if abs(sum(finite_number(weights[name], f"weight.{name}") for name in recipe_order) - 1.0) > 1e-9:
        raise ValueError("baseline_weights must sum to one.")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object.")
    window = analysis.get("window_size")
    if type(window) is not int or window < 3 or window % 2 != 1:
        raise ValueError("analysis.window_size must be an odd integer >= 3.")
    chunk = analysis.get("chunk_images")
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("analysis.chunk_images must be positive.")
    quality = analysis.get("jpeg_quality")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("analysis.jpeg_quality must be in [1, 100].")
    alpha_min = finite_number(analysis.get("alpha_min"), "analysis.alpha_min")
    alpha_max = finite_number(analysis.get("alpha_max"), "analysis.alpha_max")
    if not 0.0 <= alpha_min < alpha_max:
        raise ValueError("analysis alpha bounds are invalid.")
    if finite_number(analysis.get("dense_variance_threshold"), "analysis.dense_variance_threshold") <= 0:
        raise ValueError("dense_variance_threshold must be positive.")
    if finite_number(analysis.get("variance_epsilon"), "analysis.variance_epsilon") <= 0:
        raise ValueError("variance_epsilon must be positive.")
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("decision must be an object.")
    for field in (
        "minimum_crossfit_q95_gain",
        "minimum_crossfit_roi_gain",
        "minimum_training_oracle_q95_gain",
    ):
        finite_number(decision.get(field), f"decision.{field}")
    for field, upper in (
        ("minimum_crossfit_positive_folds", 5),
        ("minimum_crossfit_positive_rois", 25),
        ("minimum_training_oracle_positive_folds", 5),
    ):
        value = decision.get(field)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"decision.{field} is invalid.")
    return payload


def local_highpass(prediction: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    prediction32 = np.asarray(prediction, dtype=np.float32)
    mean = uniform_filter(
        prediction32,
        size=(1, window_size, window_size),
        mode="reflect",
    )
    return mean, prediction32 - mean


def unsharp_moments(
    prediction: np.ndarray,
    target: np.ndarray,
    highpass: np.ndarray,
) -> tuple[float, float]:
    residual = np.asarray(target, dtype=np.float32) - np.asarray(
        prediction, dtype=np.float32
    )
    highpass32 = np.asarray(highpass, dtype=np.float32)
    return (
        float(np.sum(residual.astype(np.float64) * highpass32.astype(np.float64))),
        float(np.sum(np.square(highpass32, dtype=np.float64))),
    )


def clipped_alpha(numerator: float, denominator: float, minimum: float, maximum: float) -> float:
    if denominator <= 0.0:
        return minimum
    return float(np.clip(numerator / denominator, minimum, maximum))


def per_image_alphas(
    prediction: np.ndarray,
    target: np.ndarray,
    highpass: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    highpass64 = np.asarray(highpass, dtype=np.float64)
    numerators = np.sum(residual * highpass64, axis=(1, 2))
    denominators = np.sum(highpass64 * highpass64, axis=(1, 2))
    raw = np.divide(
        numerators,
        denominators,
        out=np.full_like(numerators, minimum),
        where=denominators > 0,
    )
    return np.clip(raw, minimum, maximum).astype(np.float32)


def dense_local_variance_oracle(
    prediction: np.ndarray,
    target: np.ndarray,
    window_size: int,
    variance_threshold: float,
    epsilon: float,
    maximum_alpha: float,
) -> np.ndarray:
    prediction32 = np.asarray(prediction, dtype=np.float32)
    target32 = np.asarray(target, dtype=np.float32)
    size = (1, window_size, window_size)
    prediction_mean = uniform_filter(prediction32, size=size, mode="reflect")
    target_mean = uniform_filter(target32, size=size, mode="reflect")
    prediction_var = np.maximum(
        uniform_filter(prediction32 * prediction32, size=size, mode="reflect")
        - prediction_mean * prediction_mean,
        0.0,
    )
    target_var = np.maximum(
        uniform_filter(target32 * target32, size=size, mode="reflect")
        - target_mean * target_mean,
        0.0,
    )
    scale = np.ones_like(prediction32)
    active = (target_var > variance_threshold) & (target_var > prediction_var)
    ratio = np.sqrt((target_var + epsilon) / (prediction_var + epsilon))
    np.clip(ratio, 1.0, 1.0 + maximum_alpha, out=ratio)
    scale[active] = ratio[active]
    candidate = prediction_mean + scale * (prediction32 - prediction_mean)
    return np.clip(candidate, 0.0, 255.0)


@dataclass
class MetricAccumulator:
    count: int = 0
    ssim_sum: float = 0.0
    psnr_sum: float = 0.0
    score_sum: float = 0.0
    normalized_psnr_sum: float = 0.0
    squared_error_sum: float = 0.0
    pixels: int = 0

    def add(self, prediction: np.ndarray, target: np.ndarray) -> float:
        truth = np.asarray(target, dtype=np.float32)
        candidate = np.asarray(prediction, dtype=np.float32)
        ssim = float(structural_similarity(truth, candidate, data_range=255, win_size=7))
        image_error = truth.astype(np.float64) - candidate.astype(np.float64)
        image_mse = float(np.mean(np.square(image_error)))
        psnr = (
            float("inf")
            if image_mse == 0.0
            else float(10.0 * math.log10(255.0 * 255.0 / image_mse))
        )
        normalized = float(np.clip(psnr / 50.0, 0.0, 1.0))
        score = 100.0 * (0.7 * ssim + 0.3 * normalized)
        self.count += 1
        self.ssim_sum += ssim
        self.psnr_sum += psnr
        self.score_sum += score
        self.normalized_psnr_sum += normalized
        self.squared_error_sum += float(np.square(image_error).sum())
        self.pixels += int(image_error.size)
        return score

    def result(self) -> dict[str, float]:
        if self.count <= 0 or self.pixels <= 0:
            raise ValueError("Metric accumulator is empty.")
        mse = self.squared_error_sum / self.pixels
        global_psnr = float("inf") if mse == 0 else float(10.0 * math.log10(255.0 * 255.0 / mse))
        ssim = self.ssim_sum / self.count
        mean_psnr = self.psnr_sum / self.count
        return {
            "ssim": ssim,
            "psnr": mean_psnr,
            "competition_score_official": self.score_sum / self.count,
            "normalized_psnr_mean": self.normalized_psnr_sum / self.count,
            "psnr_global_mse": global_psnr,
            "competition_score_global_mse_psnr": 100.0
            * (0.7 * ssim + 0.3 * min(global_psnr, 50.0) / 50.0),
        }


def parse_roi(name: str) -> str:
    return name.split("_", 1)[0].upper()


def validate_cache(protocol: Mapping[str, object], project_root: Path) -> tuple[Path, dict[str, object]]:
    cache_root = Path(str(protocol["cache_root"]))
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    root_manifest_path = cache_root / "manifest.json"
    if sha256_file(root_manifest_path) != protocol["cache_manifest_sha256"]:
        raise ValueError("Frequency-oracle cache manifest hash mismatch.")
    root = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    if root.get("images") != 6296 or len(root.get("folds", [])) != 5:
        raise ValueError("Frequency-oracle cache coverage is incomplete.")
    if root.get("recipe_order") != protocol["recipe_order"]:
        raise ValueError("Frequency-oracle cache recipe order mismatch.")
    if root.get("baseline_weights") != protocol["baseline_weights"]:
        raise ValueError("Frequency-oracle cache weights mismatch.")
    for fold in root["folds"]:
        fold_name = str(fold["name"])
        fold_dir = cache_root / fold_name
        manifest_path = fold_dir / "manifest.json"
        if sha256_file(manifest_path) != protocol["fold_manifest_sha256"][fold_name]:
            raise ValueError(f"Frequency-oracle {fold_name} manifest hash mismatch.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest.get("names")
        expected_shape = (int(fold["images"]), 256, 256)
        if not isinstance(names, list) or len(names) != expected_shape[0]:
            raise ValueError(f"Frequency-oracle {fold_name} name coverage mismatch.")
        arrays = {
            "target.npy": np.uint8,
            "dapi.npy": np.uint8,
            **{
                f"prediction_{recipe}.npy": np.float32
                for recipe in protocol["recipe_order"]
            },
        }
        for filename, dtype in arrays.items():
            array = np.load(fold_dir / filename, mmap_mode="r")
            if array.shape != expected_shape or array.dtype != dtype:
                raise ValueError(
                    f"Frequency-oracle {fold_name}/{filename} has invalid shape/dtype."
                )
    return cache_root, root


def weighted_prediction(
    fold_dir: Path,
    recipes: list[str],
    weights: Mapping[str, float],
    start: int,
    stop: int,
) -> np.ndarray:
    result = None
    for recipe in recipes:
        source = np.load(
            fold_dir / f"prediction_{recipe}.npy", mmap_mode="r"
        )[start:stop]
        contribution = np.asarray(source, dtype=np.float32) * np.float32(
            weights[recipe]
        )
        result = contribution if result is None else result + contribution
    if result is None:
        raise RuntimeError("Frequency-oracle baseline has no members.")
    return result


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure target-free and target-oracle high-frequency correction bounds."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-images-per-fold", type=int, default=0)
    args = parser.parse_args()
    if args.max_images_per_fold < 0:
        raise ValueError("--max-images-per-fold must be non-negative.")

    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    cache_root, cache_manifest = validate_cache(protocol, project_root)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": True,
                    "cache_root": str(cache_root),
                    "images": cache_manifest["images"],
                    "folds": len(cache_manifest["folds"]),
                }
            ),
            flush=True,
        )
        return
    analysis = protocol["analysis"]
    recipes = list(protocol["recipe_order"])
    weights = dict(protocol["baseline_weights"])
    window = int(analysis["window_size"])
    chunk_images = int(analysis["chunk_images"])
    alpha_min = float(analysis["alpha_min"])
    alpha_max = float(analysis["alpha_max"])

    fold_moments: dict[str, tuple[float, float]] = {}
    for fold in cache_manifest["folds"]:
        fold_name = str(fold["name"])
        fold_dir = cache_root / fold_name
        target = np.load(fold_dir / "target.npy", mmap_mode="r")
        fold_images = (
            len(target)
            if args.max_images_per_fold == 0
            else min(len(target), args.max_images_per_fold)
        )
        numerator = 0.0
        denominator = 0.0
        for start in range(0, fold_images, chunk_images):
            stop = min(start + chunk_images, fold_images)
            prediction = weighted_prediction(
                fold_dir, recipes, weights, start, stop
            )
            _, highpass = local_highpass(prediction, window)
            chunk_num, chunk_den = unsharp_moments(
                prediction,
                target[start:stop],
                highpass,
            )
            numerator += chunk_num
            denominator += chunk_den
        fold_moments[fold_name] = (numerator, denominator)

    crossfit_alpha = {}
    for holdout in fold_moments:
        numerator = sum(
            values[0] for fold, values in fold_moments.items() if fold != holdout
        )
        denominator = sum(
            values[1] for fold, values in fold_moments.items() if fold != holdout
        )
        crossfit_alpha[holdout] = clipped_alpha(
            numerator, denominator, alpha_min, alpha_max
        )
    deployment_alpha = clipped_alpha(
        sum(values[0] for values in fold_moments.values()),
        sum(values[1] for values in fold_moments.values()),
        alpha_min,
        alpha_max,
    )

    accumulators: dict[str, dict[str, dict[str, MetricAccumulator]]] = {
        method: {
            "float": defaultdict(MetricAccumulator),
            f"jpeg_q{analysis['jpeg_quality']}": defaultdict(MetricAccumulator),
        }
        for method in METHODS
    }
    alpha_values: list[float] = []
    density_rows: list[tuple[float, float, float, float]] = []
    quality = int(analysis["jpeg_quality"])
    for fold in cache_manifest["folds"]:
        fold_name = str(fold["name"])
        fold_dir = cache_root / fold_name
        manifest = json.loads(
            (fold_dir / "manifest.json").read_text(encoding="utf-8")
        )
        names = list(manifest["names"])
        target = np.load(fold_dir / "target.npy", mmap_mode="r")
        dapi = np.load(fold_dir / "dapi.npy", mmap_mode="r")
        fold_images = (
            len(target)
            if args.max_images_per_fold == 0
            else min(len(target), args.max_images_per_fold)
        )
        for start in range(0, fold_images, chunk_images):
            stop = min(start + chunk_images, fold_images)
            prediction = weighted_prediction(
                fold_dir, recipes, weights, start, stop
            )
            target_chunk = np.asarray(target[start:stop], dtype=np.uint8)
            dapi_chunk = np.asarray(dapi[start:stop], dtype=np.uint8)
            _, highpass = local_highpass(prediction, window)
            image_alpha = per_image_alphas(
                prediction,
                target_chunk,
                highpass,
                alpha_min,
                alpha_max,
            )
            alpha_values.extend(float(value) for value in image_alpha)
            methods = {
                "baseline": prediction,
                "crossfit_global_unsharp": np.clip(
                    prediction + np.float32(crossfit_alpha[fold_name]) * highpass,
                    0.0,
                    255.0,
                ),
                "per_image_mse_oracle": np.clip(
                    prediction + image_alpha[:, None, None] * highpass,
                    0.0,
                    255.0,
                ),
                "dense_local_variance_oracle": dense_local_variance_oracle(
                    prediction,
                    target_chunk,
                    window,
                    float(analysis["dense_variance_threshold"]),
                    float(analysis["variance_epsilon"]),
                    alpha_max,
                ),
            }
            for index, name in enumerate(names[start:stop]):
                roi = parse_roi(name)
                groups = ("overall", f"fold:{fold_name}", f"roi:{roi}")
                scores: dict[str, float] = {}
                for method, candidate_batch in methods.items():
                    candidate = np.clip(candidate_batch[index], 0.0, 255.0)
                    truth = target_chunk[index]
                    float_accumulator = MetricAccumulator()
                    float_score = float_accumulator.add(candidate, truth)
                    jpeg_candidate = jpeg_roundtrip(candidate, quality)
                    jpeg_accumulator = MetricAccumulator()
                    jpeg_score = jpeg_accumulator.add(jpeg_candidate, truth)
                    for group in groups:
                        target_float = accumulators[method]["float"][group]
                        target_float.count += float_accumulator.count
                        target_float.ssim_sum += float_accumulator.ssim_sum
                        target_float.psnr_sum += float_accumulator.psnr_sum
                        target_float.score_sum += float_accumulator.score_sum
                        target_float.normalized_psnr_sum += float_accumulator.normalized_psnr_sum
                        target_float.squared_error_sum += float_accumulator.squared_error_sum
                        target_float.pixels += float_accumulator.pixels
                        target_jpeg = accumulators[method][f"jpeg_q{quality}"][group]
                        target_jpeg.count += jpeg_accumulator.count
                        target_jpeg.ssim_sum += jpeg_accumulator.ssim_sum
                        target_jpeg.psnr_sum += jpeg_accumulator.psnr_sum
                        target_jpeg.score_sum += jpeg_accumulator.score_sum
                        target_jpeg.normalized_psnr_sum += jpeg_accumulator.normalized_psnr_sum
                        target_jpeg.squared_error_sum += jpeg_accumulator.squared_error_sum
                        target_jpeg.pixels += jpeg_accumulator.pixels
                    scores[method] = jpeg_score
                density_rows.append(
                    (
                        float(np.mean(dapi_chunk[index])),
                        float(np.mean(target_chunk[index])),
                        scores["per_image_mse_oracle"] - scores["baseline"],
                        scores["dense_local_variance_oracle"] - scores["baseline"],
                    )
                )
        print(json.dumps({"completed_fold": fold_name}), flush=True)

    metrics = {
        method: {
            variant: {
                group: accumulator.result()
                for group, accumulator in groups.items()
            }
            for variant, groups in variants.items()
        }
        for method, variants in accumulators.items()
    }
    quality_key = f"jpeg_q{quality}"
    baseline_score = metrics["baseline"][quality_key]["overall"][
        "competition_score_official"
    ]
    deltas = {
        method: {
            "overall_q95": metrics[method][quality_key]["overall"][
                "competition_score_official"
            ]
            - baseline_score,
            "overall_float": metrics[method]["float"]["overall"][
                "competition_score_official"
            ]
            - metrics["baseline"]["float"]["overall"][
                "competition_score_official"
            ],
            "fold_q95": {
                fold["name"]: metrics[method][quality_key][f"fold:{fold['name']}"][
                    "competition_score_official"
                ]
                - metrics["baseline"][quality_key][f"fold:{fold['name']}"][
                    "competition_score_official"
                ]
                for fold in cache_manifest["folds"]
            },
            "roi_q95": {
                group.split(":", 1)[1]: values["competition_score_official"]
                - metrics["baseline"][quality_key][group][
                    "competition_score_official"
                ]
                for group, values in metrics[method][quality_key].items()
                if group.startswith("roi:")
            },
        }
        for method in METHODS
        if method != "baseline"
    }
    decision = protocol["decision"]
    crossfit = deltas["crossfit_global_unsharp"]
    crossfit_roi = crossfit["roi_q95"]
    crossfit_checks = {
        "overall": crossfit["overall_q95"]
        >= float(decision["minimum_crossfit_q95_gain"]),
        "folds": sum(value > 0 for value in crossfit["fold_q95"].values())
        >= int(decision["minimum_crossfit_positive_folds"]),
        "rois": sum(value > 0 for value in crossfit_roi.values())
        >= int(decision["minimum_crossfit_positive_rois"]),
        "roi_floor": min(crossfit_roi.values())
        >= float(decision["minimum_crossfit_roi_gain"]),
    }
    oracle_methods = ("per_image_mse_oracle", "dense_local_variance_oracle")
    training_oracle_checks = {
        method: {
            "overall": deltas[method]["overall_q95"]
            >= float(decision["minimum_training_oracle_q95_gain"]),
            "folds": sum(value > 0 for value in deltas[method]["fold_q95"].values())
            >= int(decision["minimum_training_oracle_positive_folds"]),
        }
        for method in oracle_methods
    }
    formal = args.max_images_per_fold == 0
    density_array = np.asarray(density_rows, dtype=np.float64)
    order = np.argsort(density_array[:, 1])
    density_deciles = []
    for indices in np.array_split(order, 10):
        density_deciles.append(
            {
                "images": int(len(indices)),
                "dapi_mean": float(np.mean(density_array[indices, 0])),
                "target_mean": float(np.mean(density_array[indices, 1])),
                "per_image_oracle_q95_gain": float(np.mean(density_array[indices, 2])),
                "dense_local_oracle_q95_gain": float(np.mean(density_array[indices, 3])),
            }
        )
    result = {
        "version": 1,
        "protocol": str(protocol_path),
        "protocol_name": protocol["name"],
        "cache_root": str(cache_root),
        "cache_manifest_sha256": protocol["cache_manifest_sha256"],
        "images": len(density_rows),
        "formal": formal,
        "methods": list(METHODS),
        "crossfit_alpha_by_fold": crossfit_alpha,
        "deployment_alpha": deployment_alpha,
        "per_image_alpha_quantiles": {
            str(quantile): float(np.quantile(alpha_values, quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "metrics": metrics,
        "deltas": deltas,
        "density_deciles": density_deciles,
        "decision": {
            "thresholds": decision,
            "crossfit_checks": crossfit_checks,
            "crossfit_actionable": formal and all(crossfit_checks.values()),
            "training_oracle_checks": training_oracle_checks,
            "frequency_training_actionable": formal
            and any(all(checks.values()) for checks in training_oracle_checks.values()),
            "test_inference_authorized": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
