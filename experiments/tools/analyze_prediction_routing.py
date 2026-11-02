from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def jpeg_roundtrip(image: np.ndarray, quality: int = 95) -> np.ndarray:
    from io import BytesIO
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode="L").convert(
        "RGB"
    ).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def features(dapi: np.ndarray, prediction_a: np.ndarray, prediction_b: np.ndarray) -> np.ndarray:
    dapi_float = dapi.astype(np.float32)
    grad = np.hypot(sobel(dapi_float, axis=1), sobel(dapi_float, axis=2))
    p_a = prediction_a.astype(np.float32)
    p_b = prediction_b.astype(np.float32)
    disagreement = np.abs(p_a - p_b)
    return np.stack(
        (
            dapi_float.mean(axis=(1, 2)),
            dapi_float.std(axis=(1, 2)),
            np.quantile(dapi_float, 0.05, axis=(1, 2)),
            np.quantile(dapi_float, 0.50, axis=(1, 2)),
            np.quantile(dapi_float, 0.95, axis=(1, 2)),
            np.mean(dapi_float <= 5.0, axis=(1, 2)),
            grad.mean(axis=(1, 2)),
            p_a.mean(axis=(1, 2)),
            p_b.mean(axis=(1, 2)),
            disagreement.mean(axis=(1, 2)),
            disagreement.std(axis=(1, 2)),
        ),
        axis=1,
    ).astype(np.float64)


def score_one(prediction: np.ndarray, target: np.ndarray) -> float:
    candidate = jpeg_roundtrip(prediction)
    ssim = structural_similarity(target, candidate, data_range=255, win_size=7)
    psnr = peak_signal_noise_ratio(target, candidate, data_range=255)
    return float(100.0 * (0.7 * ssim + 0.3 * np.clip(psnr / 50.0, 0.0, 1.0)))


def score_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([score_one(p, t) for p, t in zip(prediction, target)], dtype=np.float64)


def load_cache(root: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    dapi, target, first, second = [], [], [], []
    for fold in sorted(root.glob("fold*")):
        dapi.append(np.load(fold / "dapi.npy", mmap_mode="r"))
        target.append(np.load(fold / "target.npy", mmap_mode="r"))
        first.append(
            np.load(fold / "prediction_thumbnailbase32.npy", mmap_mode="r")
        )
        second.append(
            np.load(
                fold / "prediction_thumbnailbase48continue.npy", mmap_mode="r"
            )
        )
    if len(dapi) != 5:
        raise RuntimeError(f"Expected five cache folds, found {len(dapi)}")
    return dapi, target, first, second


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted OOF prediction-routing diagnostic.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dapi, target, first, second = load_cache(args.cache_root)
    weight_grid = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    baseline_index = int(np.flatnonzero(np.isclose(weight_grid, 0.7))[0])
    baseline_scores: list[np.ndarray] = []
    oracle_scores: list[np.ndarray] = []
    oracle_weights: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for fold_index in range(5):
        a, b, y = first[fold_index], second[fold_index], target[fold_index]
        grid_scores = np.stack(
            [score_rows(weight * a + (1.0 - weight) * b, y) for weight in weight_grid],
            axis=1,
        )
        score_base = grid_scores[:, baseline_index]
        best_indices = grid_scores.argmax(axis=1)
        score_oracle = grid_scores[np.arange(len(y)), best_indices]
        weight_oracle = weight_grid[best_indices].astype(np.float64)
        baseline_scores.append(score_base)
        oracle_scores.append(score_oracle)
        oracle_weights.append(weight_oracle)
        rows.append(
            {
                "fold": f"fold{fold_index}",
                "images": int(len(y)),
                "base_mean": float(score_base.mean()),
                "oracle_grid_mean": float(score_oracle.mean()),
                "oracle_weight_a_mean": float(weight_oracle.mean()),
            }
        )

    # Each held-out fold receives a gate fitted only on the other four folds.
    # The regression target is the best 0.1-spaced convex weight per training
    # image; inference stays continuous but clipped to the same convex range.
    crossfit_scores: list[np.ndarray] = []
    for holdout in range(5):
        train_indices = [i for i in range(5) if i != holdout]
        x_train = np.concatenate(
            [features(dapi[i], first[i], second[i]) for i in train_indices]
        )
        y_train = np.concatenate([oracle_weights[i] for i in train_indices])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x_train, y_train)
        x_test = features(dapi[holdout], first[holdout], second[holdout])
        weight_a = np.clip(model.predict(x_test), 0.0, 1.0)
        routed = weight_a[:, None, None] * first[holdout] + (1.0 - weight_a[:, None, None]) * second[holdout]
        routed_scores = score_rows(routed, target[holdout])
        crossfit_scores.append(routed_scores)
        rows[holdout]["routing_mean"] = float(routed_scores.mean())
        rows[holdout]["routing_weight_a_mean"] = float(weight_a.mean())

    baseline_all = np.concatenate(baseline_scores)
    oracle_all = np.concatenate(oracle_scores)
    routing_all = np.concatenate(crossfit_scores)
    report = {
        "version": 1,
        "method": "five-fold-crossfit-ridge-dapi-and-disagreement-routing",
        "images": int(len(baseline_all)),
        "overall": {
            "fixed_70_30": float(baseline_all.mean()),
            "oracle_0p1_convex_grid": float(oracle_all.mean()),
            "crossfit_ridge_routing": float(routing_all.mean()),
            "oracle_gain": float(oracle_all.mean() - baseline_all.mean()),
            "crossfit_gain": float(routing_all.mean() - baseline_all.mean()),
        },
        "weight_grid": weight_grid.tolist(),
        "folds": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
