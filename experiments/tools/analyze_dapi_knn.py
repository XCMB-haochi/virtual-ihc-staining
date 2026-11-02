from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_torch import competition_score, shift_array_edge


def jpeg_roundtrip(image: np.ndarray, quality: int = 95) -> np.ndarray:
    buffer = io.BytesIO()
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").convert("RGB").save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def downsample_features(images: np.ndarray) -> np.ndarray:
    if images.ndim != 3 or images.shape[1:] != (256, 256):
        raise ValueError(f"Expected N x 256 x 256 images, got {images.shape}")
    # Average-pool to 16x16, retaining coarse morphology and intensity layout.
    pooled = images.reshape(len(images), 16, 16, 16, 16).mean(axis=(2, 4))
    means = images.mean(axis=(1, 2), keepdims=True)
    stds = images.std(axis=(1, 2), keepdims=True).clip(min=1.0)
    normalized = (pooled - means) / stds
    horizontal = np.abs(np.diff(images.astype(np.float32), axis=2)).mean(axis=(1, 2))
    vertical = np.abs(np.diff(images.astype(np.float32), axis=1)).mean(axis=(1, 2))
    summary = np.stack(
        (
            means[:, 0, 0],
            stds[:, 0, 0],
            horizontal,
            vertical,
            np.quantile(images, 0.05, axis=(1, 2)),
            np.quantile(images, 0.50, axis=(1, 2)),
            np.quantile(images, 0.95, axis=(1, 2)),
        ),
        axis=1,
    )
    return np.concatenate((pooled.reshape(len(images), -1), normalized.reshape(len(images), -1), summary), axis=1).astype(np.float32)


def score_batch(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    ssim_sum = psnr_sum = score_sum = 0.0
    for pred, truth in zip(prediction, target):
        decoded = jpeg_roundtrip(pred)
        ssim = float(structural_similarity(truth, decoded, data_range=255, win_size=7))
        psnr = float(peak_signal_noise_ratio(truth, decoded, data_range=255))
        ssim_sum += ssim
        psnr_sum += psnr
        score_sum += competition_score(ssim, psnr)
    count = len(prediction)
    return {
        "images": count,
        "ssim": ssim_sum / count,
        "psnr": psnr_sum / count,
        "competition_score_official": score_sum / count,
    }


def load_cache(root: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[list[str]]]:
    dapi_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    names: list[list[str]] = []
    for fold_dir in sorted(root.glob("fold*/")):
        manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        fold_names = list(manifest["names"])
        dapi_parts.append(np.asarray(np.load(fold_dir / "dapi.npy", mmap_mode="r"), dtype=np.uint8))
        target_parts.append(np.asarray(np.load(fold_dir / "target.npy", mmap_mode="r"), dtype=np.uint8))
        first = np.load(fold_dir / "prediction_thumbnailbase32.npy", mmap_mode="r")
        second = np.load(fold_dir / "prediction_thumbnailbase48continue.npy", mmap_mode="r")
        baseline_parts.append(0.7 * np.asarray(first, dtype=np.float32) + 0.3 * np.asarray(second, dtype=np.float32))
        names.append(fold_names)
    if len(dapi_parts) != 5:
        raise RuntimeError(f"Expected five cache folds, found {len(dapi_parts)}")
    return dapi_parts, target_parts, baseline_parts, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted DAPI nearest-neighbor target diagnostic.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--blend-weight", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.neighbors < 1 or not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("neighbors must be positive and blend weight must be in [0,1]")

    dapi, target, baseline, names = load_cache(args.cache_root)
    features = [downsample_features(values) for values in dapi]
    fold_reports = []
    baseline_rows = []
    knn_rows = []
    blend_rows = []
    for holdout in range(5):
        train_indices = [i for i in range(5) if i != holdout]
        train_x = np.concatenate([features[i] for i in train_indices])
        test_x = features[holdout]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
        neighbors = NearestNeighbors(n_neighbors=args.neighbors, metric="euclidean", algorithm="auto")
        neighbors.fit(train_x)
        distances, indices = neighbors.kneighbors(test_x)
        train_targets = np.concatenate([target[i] for i in train_indices])
        retrieved = train_targets[indices].mean(axis=1).astype(np.float32)
        y = target[holdout]
        p = baseline[holdout]
        # The fixed shift is applied to every candidate equally in this diagnostic.
        p_shift = shift_array_edge(p, (0, -1))
        retrieved_shift = shift_array_edge(retrieved, (0, -1))
        blend = (1.0 - args.blend_weight) * p_shift + args.blend_weight * retrieved_shift
        base_score = score_batch(p_shift, y)
        knn_score = score_batch(retrieved_shift, y)
        blend_score = score_batch(blend, y)
        baseline_rows.append(base_score)
        knn_rows.append(knn_score)
        blend_rows.append(blend_score)
        fold_reports.append(
            {
                "fold": f"fold{holdout}",
                "images": int(len(y)),
                "nearest_distance_mean": float(distances.mean()),
                "nearest_distance_median": float(np.median(distances[:, 0])),
                "baseline": base_score,
                "knn": knn_score,
                "fixed_blend": blend_score,
            }
        )

    def average(rows: list[dict[str, float]]) -> dict[str, float]:
        total = sum(row["images"] for row in rows)
        return {
            key: float(sum(row[key] * row["images"] for row in rows) / total)
            for key in rows[0]
            if key != "images"
        } | {"images": total}

    overall = {
        "baseline": average(baseline_rows),
        "knn": average(knn_rows),
        "fixed_blend": average(blend_rows),
    }
    overall["knn_delta_vs_baseline"] = overall["knn"]["competition_score_official"] - overall["baseline"]["competition_score_official"]
    overall["blend_delta_vs_baseline"] = overall["fixed_blend"]["competition_score_official"] - overall["baseline"]["competition_score_official"]
    report = {
        "version": 1,
        "method": "five-fold-crossfit-dapi-16x16-nearest-target",
        "cache": str(args.cache_root.resolve()),
        "neighbors": args.neighbors,
        "blend_weight": args.blend_weight,
        "feature_definition": "raw and per-image-standardized 16x16 average-pooled DAPI plus seven summaries",
        "data_scope": "each holdout ROI retrieves only from the other four OOF folds; no test inputs or labels",
        "folds": fold_reports,
        "overall": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
