from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import load_gray


def activity(image: np.ndarray) -> tuple[float, float, float, float, float]:
    values = image.astype(np.float32)
    horizontal = np.abs(values[:, 1:] - values[:, :-1]).mean()
    vertical = np.abs(values[1:, :] - values[:-1, :]).mean()
    mean = float(values.mean())
    std = float(values.std())
    p95 = float(np.quantile(values, 0.95))
    gradient = float((horizontal + vertical) / 2.0)
    foreground = float(np.mean(values > mean))
    combined = std + gradient + 0.25 * foreground + 0.25 * p95
    return combined, mean, std, gradient, foreground


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit activity and OOF error strata without test labels.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_root = args.cache_root
    dapi_activity: list[float] = []
    target_activity: list[float] = []
    baseline_error: list[float] = []
    names: list[str] = []
    for fold_dir in sorted(cache_root.glob("fold*/")):
        manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        fold_names = list(manifest["names"])
        dapi = np.load(fold_dir / "dapi.npy", mmap_mode="r")
        target = np.load(fold_dir / "target.npy", mmap_mode="r")
        first = np.load(fold_dir / "prediction_thumbnailbase32.npy", mmap_mode="r")
        second = np.load(fold_dir / "prediction_thumbnailbase48continue.npy", mmap_mode="r")
        prediction = 0.7 * np.asarray(first, dtype=np.float32) + 0.3 * np.asarray(second, dtype=np.float32)
        for index, name in enumerate(fold_names):
            dapi_value, *_ = activity(np.asarray(dapi[index]))
            target_value, *_ = activity(np.asarray(target[index]))
            dapi_activity.append(dapi_value)
            target_activity.append(target_value)
            baseline_error.append(float(np.mean((prediction[index] - target[index]) ** 2)))
            names.append(name)

    dapi_values = np.asarray(dapi_activity, dtype=np.float64)
    target_values = np.asarray(target_activity, dtype=np.float64)
    errors = np.asarray(baseline_error, dtype=np.float64)
    quantiles = np.quantile(target_values, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for index in range(4):
        mask = (target_values >= quantiles[index]) & (target_values <= quantiles[index + 1])
        bins.append(
            {
                "bin": index,
                "target_activity_range": [float(quantiles[index]), float(quantiles[index + 1])],
                "images": int(mask.sum()),
                "dapi_activity_mean": float(dapi_values[mask].mean()),
                "baseline_mse_mean": float(errors[mask].mean()),
            }
        )

    report = {
        "version": 1,
        "method": "training-oof-target-activity-vs-dapi-activity-audit",
        "images": len(names),
        "activity_definition": "std + mean_abs_neighbor_difference + 0.25*foreground_fraction_above_mean + 0.25*p95",
        "correlation": {
            "dapi_target_activity_pearson": float(pearsonr(dapi_values, target_values).statistic),
            "target_activity_error_mse_pearson": float(pearsonr(target_values, errors).statistic),
            "dapi_activity_error_mse_pearson": float(pearsonr(dapi_values, errors).statistic),
        },
        "train_target_activity_quantiles": quantiles.tolist(),
        "train_target_activity_bins": bins,
        "train_dapi_activity": {
            "mean": float(dapi_values.mean()),
            "std": float(dapi_values.std()),
            "min": float(dapi_values.min()),
            "max": float(dapi_values.max()),
        },
        "data_scope": "OOF targets and predictions only; no test inputs, labels, or model fitting",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
