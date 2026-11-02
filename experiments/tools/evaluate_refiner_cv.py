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

from evaluate_cv_ensemble import select_indices, variant_metrics
from ihc_data import image_names, parse_patch_name, resolve_data_root, roi_intensity_means
from ihc_refiner import (
    CachedOOFDataset,
    build_refiner_from_checkpoint,
    predict_refiner_with_tta,
)
from refiner_protocol import (
    load_base_protocol,
    load_refiner_protocol,
    sha256_file,
    validate_cache_manifest,
)
from search_postprocessing import evaluate


@torch.no_grad()
def predict_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    baselines: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    for features, target, batch_names in loader:
        features = features.to(device, non_blocking=True)
        prediction = predict_refiner_with_tta(model, features, tta)
        baselines.append(
            features[:, model.baseline_channel].cpu().numpy().astype(np.float32)
            * 255.0
        )
        candidates.append(prediction[:, 0].cpu().numpy().astype(np.float32) * 255.0)
        targets.append(np.rint(target[:, 0].numpy() * 255.0).astype(np.uint8))
        names.extend(batch_names)
    return (
        np.concatenate(baselines),
        np.concatenate(candidates),
        np.concatenate(targets),
        names,
    )


def score_pair(metrics: dict[str, float]) -> tuple[float, float]:
    return (
        float(metrics["competition_score_mean_image_psnr"]),
        float(metrics["competition_score_global_mse_psnr"]),
    )


def subtract_scores(
    candidate: dict[str, float], baseline: dict[str, float]
) -> list[float]:
    candidate_scores = score_pair(candidate)
    baseline_scores = score_pair(baseline)
    return [
        candidate_scores[index] - baseline_scores[index] for index in range(2)
    ]


def apply_promotion_gate(
    protocol: dict[str, object],
    overall: dict[str, object],
    test_like: dict[str, object],
    fold_metrics: dict[str, object],
    roi_metrics: dict[str, object],
    test_like_rois: set[str],
) -> dict[str, object]:
    quality_key = f"jpeg_q{protocol['inference']['jpeg_quality']}"
    overall_q = subtract_scores(
        overall["refined"][quality_key], overall["baseline"][quality_key]
    )
    test_like_q = subtract_scores(
        test_like["refined"][quality_key], test_like["baseline"][quality_key]
    )
    overall_float = subtract_scores(
        overall["refined"]["float"], overall["baseline"]["float"]
    )
    test_like_float = subtract_scores(
        test_like["refined"]["float"], test_like["baseline"]["float"]
    )
    fold_deltas = {
        name: subtract_scores(
            metrics["refined"][quality_key], metrics["baseline"][quality_key]
        )
        for name, metrics in fold_metrics.items()
    }
    dual_positive_folds = sum(all(value > 0 for value in delta) for delta in fold_deltas.values())
    roi_deltas = {
        roi: subtract_scores(
            roi_metrics["refined"][quality_key][roi],
            roi_metrics["baseline"][quality_key][roi],
        )
        for roi in sorted(test_like_rois)
    }
    dual_negative_rois = [
        roi for roi, delta in roi_deltas.items() if all(value < 0 for value in delta)
    ]
    minimum_roi_gain = min(value for delta in roi_deltas.values() for value in delta)
    gate = protocol["promotion_gate"]
    checks = {
        "overall_q95": min(overall_q)
        >= float(gate["minimum_overall_q95_gain"]),
        "test_like_q95": min(test_like_q)
        >= float(gate["minimum_test_like_q95_gain"]),
        "float": min(overall_float + test_like_float)
        > float(gate["minimum_float_gain"]),
        "folds": dual_positive_folds
        >= int(gate["minimum_dual_positive_folds"]),
        "roi_floor": minimum_roi_gain
        >= float(gate["minimum_test_like_roi_q95_gain"]),
        "roi_count": len(dual_negative_rois)
        <= int(gate["maximum_dual_negative_test_like_rois"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overall_q95_delta": overall_q,
        "test_like_q95_delta": test_like_q,
        "overall_float_delta": overall_float,
        "test_like_float_delta": test_like_float,
        "fold_q95_deltas": fold_deltas,
        "dual_positive_folds": dual_positive_folds,
        "test_like_roi_q95_deltas": roi_deltas,
        "dual_negative_test_like_rois": dual_negative_rois,
        "minimum_test_like_roi_q95_gain": minimum_roi_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate five cross-fitted refiners over complete OOF predictions."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_refiner_protocol(protocol_path)
    base_protocol = load_base_protocol(protocol)
    data_root = resolve_data_root(args.dataset)
    cache_root = args.cache_root.resolve()
    validate_cache_manifest(
        cache_root,
        protocol_path,
        protocol,
        base_protocol,
    )
    run_root = args.run_root.resolve()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    inference = protocol["inference"]
    recipe_order = list(protocol["recipe_order"])
    weights = dict(protocol["baseline_weights"])
    baseline_parts: list[np.ndarray] = []
    refined_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    all_names: list[str] = []
    fold_metrics: dict[str, object] = {}
    checkpoint_provenance: dict[str, object] = {}

    fold_names = [str(fold["name"]) for fold in base_protocol["folds"]]
    for fold in base_protocol["folds"]:
        fold_name = str(fold["name"])
        dataset = CachedOOFDataset(
            cache_root,
            [fold_name],
            recipe_order,
            weights,
            augment=False,
            preload=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        checkpoint_path = (
            run_root / f"{protocol['name']}_{fold_name}" / "best_global_mse.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing refiner checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_training_folds = [name for name in fold_names if name != fold_name]
        if (
            checkpoint.get("protocol_sha256") != sha256_file(protocol_path)
            or checkpoint.get("holdout_fold") != fold_name
            or checkpoint.get("training_folds") != expected_training_folds
            or set(checkpoint.get("val_rois", [])) != set(fold["val_rois"])
        ):
            raise ValueError(f"Refiner checkpoint leakage metadata failed for {fold_name}.")
        model = build_refiner_from_checkpoint(checkpoint, device).eval()
        baseline, refined, target, names = predict_fold(
            model,
            loader,
            device,
            str(inference["tta"]),
        )
        expected_names = [
            name
            for name in image_names(data_root / "train" / "DAPI")
            if parse_patch_name(name).roi in set(fold["val_rois"])
        ]
        if names != expected_names:
            raise RuntimeError(f"Cached prediction order mismatch for {fold_name}.")
        quality = int(inference["jpeg_quality"])
        fold_metrics[fold_name] = {
            "baseline": {
                "float": evaluate(baseline, target, 0),
                f"jpeg_q{quality}": evaluate(baseline, target, quality),
            },
            "refined": {
                "float": evaluate(refined, target, 0),
                f"jpeg_q{quality}": evaluate(refined, target, quality),
            },
        }
        checkpoint_provenance[fold_name] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "training_folds": checkpoint["training_folds"],
            "val_rois": checkpoint["val_rois"],
        }
        baseline_parts.append(baseline)
        refined_parts.append(refined)
        target_parts.append(target)
        all_names.extend(names)
        del model, dataset, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"completed_fold": fold_name}), flush=True)

    expected_all_names = image_names(data_root / "train" / "DAPI")
    if len(all_names) != len(set(all_names)) or set(all_names) != set(expected_all_names):
        raise RuntimeError("Refiner OOF predictions do not cover each image once.")
    baseline = np.concatenate(baseline_parts)
    refined = np.concatenate(refined_parts)
    target = np.concatenate(target_parts)
    quality = int(inference["jpeg_quality"])
    variants = {"baseline": baseline, "refined": refined}
    overall: dict[str, object] = {}
    roi_metrics: dict[str, object] = {}
    for name, prediction in variants.items():
        metrics = variant_metrics(prediction, target, all_names, quality)
        overall[name] = {
            "float": metrics["float"],
            f"jpeg_q{quality}": metrics[f"jpeg_q{quality}"],
        }
        roi_metrics[name] = metrics["roi"]

    train_roi_means, _ = roi_intensity_means(
        data_root / "train" / "DAPI", expected_all_names
    )
    test_names = image_names(data_root / "test" / "DAPI")
    test_roi_means, _ = roi_intensity_means(data_root / "test" / "DAPI", test_names)
    test_min = min(test_roi_means.values())
    test_max = max(test_roi_means.values())
    test_like_rois = {
        roi for roi, value in train_roi_means.items() if test_min <= value <= test_max
    }
    indices = select_indices(all_names, test_like_rois)
    test_like: dict[str, object] = {}
    for name, prediction in variants.items():
        test_like[name] = {
            "images": int(len(indices)),
            "float": evaluate(prediction[indices], target[indices], 0),
            f"jpeg_q{quality}": evaluate(
                prediction[indices], target[indices], quality
            ),
        }
    gate = apply_promotion_gate(
        protocol,
        overall,
        test_like,
        fold_metrics,
        roi_metrics,
        test_like_rois,
    )
    report = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_name": protocol["name"],
        "marker": protocol["marker"],
        "validation_images": len(all_names),
        "validation_rois": sorted({parse_patch_name(name).roi for name in all_names}),
        "test_dapi_roi_means": test_roi_means,
        "test_like_dapi_mean_range": [test_min, test_max],
        "test_like_train_rois": sorted(test_like_rois),
        "checkpoint_provenance": checkpoint_provenance,
        "overall": overall,
        "test_like": test_like,
        "fold_metrics": fold_metrics,
        "roi_metrics": roi_metrics,
        "promotion_gate": gate,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(output + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
