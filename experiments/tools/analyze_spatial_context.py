from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ihc_data import image_names, load_gray, parse_patch_name, resolve_data_root
from ihc_spatial import (
    LOCAL_FEATURE_NAMES,
    build_spatial_context_features,
    local_intensity_features,
)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    mse = float(np.mean(residual**2))
    variance = float(np.mean((target - target.mean()) ** 2))
    return {
        "rmse_uint8": float(np.sqrt(mse) * 255.0),
        "mae_uint8": float(np.mean(np.abs(residual)) * 255.0),
        "r2": 1.0 - mse / variance if variance > 0 else float("nan"),
    }


def make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether DAPI context beyond one tile predicts held-out ROI CD68 intensity."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/round26_cv5.json")
    parser.add_argument("--marker", default="CD68")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_root = resolve_data_root(args.dataset)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    folds = protocol["folds"]
    names = image_names(data_root / "train" / "DAPI")
    patches = [parse_patch_name(name) for name in names]

    local_rows: list[np.ndarray] = []
    targets: list[float] = []
    for index, name in enumerate(names, start=1):
        source = load_gray(data_root / "train" / "DAPI" / name)
        target = load_gray(data_root / "train" / args.marker / name)
        local_rows.append(local_intensity_features(source))
        targets.append(float(target.mean()) / 255.0)
        if index % 500 == 0 or index == len(names):
            print(f"features {index}/{len(names)}", flush=True)

    local = np.stack(local_rows)
    context = build_spatial_context_features(patches, local, radius=args.radius)
    neighborhood_start = context.names.index("radius1_mean")
    geometry = np.concatenate((local, context.values[:, :neighborhood_start]), axis=1)
    neighborhood = np.concatenate((local, context.values[:, neighborhood_start:]), axis=1)
    augmented = np.concatenate((local, context.values), axis=1)
    target_array = np.asarray(targets, dtype=np.float32)
    roi_array = np.asarray([patch.roi for patch in patches])

    predictions = {
        "local": np.empty_like(target_array),
        "local_geometry": np.empty_like(target_array),
        "local_neighborhood": np.empty_like(target_array),
        "local_spatial": np.empty_like(target_array),
    }
    fold_reports: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        validation = np.isin(roi_array, fold["val_rois"])
        training = ~validation
        fold_report: dict[str, object] = {
            "name": fold["name"],
            "val_rois": fold["val_rois"],
            "train_images": int(training.sum()),
            "validation_images": int(validation.sum()),
            "models": {},
        }
        for model_name, features in (
            ("local", local),
            ("local_geometry", geometry),
            ("local_neighborhood", neighborhood),
            ("local_spatial", augmented),
        ):
            model = make_regressor(args.seed + fold_index)
            model.fit(features[training], target_array[training])
            prediction = model.predict(features[validation]).astype(np.float32)
            predictions[model_name][validation] = prediction
            fold_report["models"][model_name] = regression_metrics(
                target_array[validation], prediction
            )
        local_rmse = fold_report["models"]["local"]["rmse_uint8"]
        spatial_rmse = fold_report["models"]["local_spatial"]["rmse_uint8"]
        fold_report["relative_rmse_reduction"] = 1.0 - spatial_rmse / local_rmse
        fold_reports.append(fold_report)
        print(json.dumps(fold_report, ensure_ascii=False), flush=True)

    overall = {
        model_name: regression_metrics(target_array, prediction)
        for model_name, prediction in predictions.items()
    }
    overall["relative_rmse_reduction"] = (
        1.0
        - overall["local_spatial"]["rmse_uint8"]
        / overall["local"]["rmse_uint8"]
    )
    positive_folds = sum(
        report["relative_rmse_reduction"] > 0 for report in fold_reports
    )
    decision = {
        "minimum_overall_relative_rmse_reduction": 0.01,
        "minimum_positive_folds": 4,
        "positive_folds": positive_folds,
        "promote_spatial_context": (
            overall["relative_rmse_reduction"] >= 0.01 and positive_folds >= 4
        ),
        "scope": (
            "Proxy decision for a neural spatial-context experiment; tile-mean "
            "regression is not a competition-score estimate."
        ),
    }
    report = {
        "dataset": str(data_root),
        "protocol": str(args.protocol.resolve()),
        "marker": args.marker,
        "images": len(names),
        "rois": sorted(set(roi_array.tolist())),
        "radius_tiles": args.radius,
        "local_features": list(LOCAL_FEATURE_NAMES),
        "spatial_features": list(context.names),
        "feature_groups": {
            "geometry": list(context.names[:neighborhood_start]),
            "neighborhood": list(context.names[neighborhood_start:]),
        },
        "regressor": {
            "type": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "seed": args.seed,
        },
        "folds": fold_reports,
        "overall": overall,
        "decision": decision,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
