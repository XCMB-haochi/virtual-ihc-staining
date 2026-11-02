from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cv_protocol import (
    candidate_output_source_offset,
    inference_checkpoint_filename,
    load_protocol,
    recipe_inference_context_halo,
    sourced_run_directory,
)
from ihc_data import (
    image_names,
    parse_patch_name,
    resolve_data_root,
    roi_intensity_means,
)
from ihc_torch import (
    PairedPatchDataset,
    batch_condition_tensor,
    build_model_from_checkpoint,
    center_crop_like,
    checkpoint_conditioning,
    crop_context_halo,
    predict_with_tta,
    shift_array_edge,
)
from ihc_spatial import apply_complete_roi_conditioning
from search_postprocessing import evaluate, evaluate_by_roi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_prediction(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    unknown = sorted(set(weights) - set(predictions))
    if unknown:
        raise ValueError(f"Candidate weights reference missing predictions: {unknown}.")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Candidate weights must sum to one.")
    result = None
    for name, weight in weights.items():
        contribution = predictions[name] * float(weight)
        result = contribution if result is None else result + contribution
    if result is None:
        raise ValueError("Candidate weights cannot be empty.")
    return result.astype(np.float32, copy=False)


def shift_prediction_edge(
    prediction: np.ndarray,
    source_offset: tuple[int, int],
) -> np.ndarray:
    """Translate output by sampling a fixed source offset with edge padding."""

    return shift_array_edge(prediction, source_offset)


def validate_checkpoint_fold(
    checkpoint: dict[str, object], expected_rois: set[str], marker: str, path: Path
) -> None:
    markers = tuple(checkpoint.get("markers", ()))
    if marker not in markers:
        raise ValueError(f"{path} does not contain marker {marker}: {markers}.")
    arguments = checkpoint.get("args")
    if not isinstance(arguments, dict):
        raise ValueError(f"{path} has no checkpoint argument metadata.")
    actual_rois = {
        item.strip().upper()
        for item in str(arguments.get("val_rois", "")).split(",")
        if item.strip()
    }
    if actual_rois != expected_rois:
        raise ValueError(
            f"Checkpoint fold mismatch for {path}: expected={sorted(expected_rois)}, "
            f"actual={sorted(actual_rois)}."
        )


@torch.no_grad()
def predict_fold(
    checkpoint_paths: dict[str, Path],
    loader: DataLoader,
    val_names: list[str],
    expected_rois: set[str],
    data_root: Path,
    marker: str,
    device: torch.device,
    tta: str,
    source_context_halo: int,
    recipe_context_halos: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, object]]:
    if set(checkpoint_paths) != set(recipe_context_halos):
        raise ValueError("Checkpoint and recipe inference-halo plans do not match.")
    models: dict[str, torch.nn.Module] = {}
    marker_indices: dict[str, int] = {}
    conditions: dict[str, dict[str, float] | None] = {}
    provenance: dict[str, object] = {}
    for recipe_name, path in checkpoint_paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        validate_checkpoint_fold(checkpoint, expected_rois, marker, path)
        markers = tuple(checkpoint["markers"])
        models[recipe_name] = build_model_from_checkpoint(checkpoint, device).eval()
        marker_indices[recipe_name] = markers.index(marker)
        conditioning = checkpoint_conditioning(checkpoint)
        conditions[recipe_name] = apply_complete_roi_conditioning(
            data_root / "train" / "DAPI",
            val_names,
            conditioning,
        )
        provenance[recipe_name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "epoch": int(checkpoint["epoch"]),
            "selection_psnr_aggregation": checkpoint.get("selection_psnr_aggregation"),
            "best_score": checkpoint.get("best_score"),
            "conditioning": conditioning,
            "inference_context_halo": recipe_context_halos[recipe_name],
        }

    collected = {name: [] for name in models}
    targets: list[np.ndarray] = []
    names: list[str] = []
    for source, target, batch_names in loader:
        source_device = source.to(device)
        for recipe_name, model in models.items():
            recipe_source = crop_context_halo(
                source_device,
                source_context_halo,
                recipe_context_halos[recipe_name],
            )
            condition = batch_condition_tensor(
                batch_names, conditions[recipe_name], device
            )
            prediction = center_crop_like(
                predict_with_tta(model, recipe_source, tta, condition), target
            )[:, marker_indices[recipe_name]]
            collected[recipe_name].append(
                prediction.cpu().numpy().astype(np.float32) * 255.0
            )
        targets.append(np.rint(target.numpy()[:, 0] * 255.0).astype(np.uint8))
        names.extend(batch_names)

    predictions = {name: np.concatenate(items) for name, items in collected.items()}
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, np.concatenate(targets), names, provenance


def variant_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    names: list[str],
    jpeg_quality: int,
) -> dict[str, object]:
    return {
        "float": evaluate(prediction, target, 0),
        f"jpeg_q{jpeg_quality}": evaluate(prediction, target, jpeg_quality),
        "roi": {
            "float": evaluate_by_roi(prediction, target, names, 0),
            f"jpeg_q{jpeg_quality}": evaluate_by_roi(
                prediction, target, names, jpeg_quality
            ),
        },
    }


def select_indices(names: list[str], rois: set[str]) -> np.ndarray:
    return np.asarray(
        [index for index, name in enumerate(names) if parse_patch_name(name).roi in rois],
        dtype=np.int64,
    )


def protocol_evaluation_names(
    all_names: list[str],
    protocol: dict[str, object],
) -> tuple[list[str], set[str]]:
    """Resolve the exact OOF universe while retaining complete-CV safeguards."""
    actual_rois = {parse_patch_name(name).roi for name in all_names}
    expected_rois = set(protocol["expected_rois"])
    roi_scope = str(protocol.get("roi_scope", "complete"))
    if roi_scope == "complete":
        if actual_rois != expected_rois:
            raise ValueError("Dataset ROI set does not match the CV protocol.")
    elif roi_scope == "subset":
        missing = sorted(expected_rois - actual_rois)
        if missing:
            raise ValueError(
                "Dataset is missing ROIs required by the subset CV protocol: "
                f"{missing}."
            )
    else:
        raise ValueError(f"Unsupported protocol roi_scope: {roi_scope!r}.")
    evaluation_names = [
        name for name in all_names if parse_patch_name(name).roi in expected_rois
    ]
    if not evaluation_names:
        raise ValueError("The CV protocol selects no dataset images.")
    return evaluation_names, actual_rois


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed fold-specific models over complete OOF predictions."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    data_root = resolve_data_root(args.dataset)
    all_names = image_names(data_root / "train" / "DAPI")
    evaluation_names, actual_rois = protocol_evaluation_names(all_names, protocol)
    expected_rois = set(protocol["expected_rois"])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    inference = protocol["inference"]
    checkpoint_filename = inference_checkpoint_filename(protocol)
    test_like_diagnostic = bool(inference.get("test_like_diagnostic", True))
    source_context_halo = int(inference["context_halo"])
    recipe_context_halos = {
        recipe_name: recipe_inference_context_halo(protocol, recipe_name)
        for recipe_name in protocol["recipes"]
    }
    output_root = args.output_root.resolve()
    candidate_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    target_parts: list[np.ndarray] = []
    oof_names: list[str] = []
    fold_metrics: dict[str, object] = {}
    checkpoint_provenance: dict[str, object] = {}

    for fold in protocol["folds"]:
        val_rois = set(fold["val_rois"])
        val_names = [
            name
            for name in evaluation_names
            if parse_patch_name(name).roi in val_rois
        ]
        loader = DataLoader(
            PairedPatchDataset(
                data_root,
                val_names,
                markers=(protocol["marker"],),
                augment=False,
                context_halo=source_context_halo,
            ),
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        checkpoint_paths = {
            recipe_name: sourced_run_directory(
                output_root,
                protocol,
                str(fold["name"]),
                recipe_name,
            )
            / checkpoint_filename
            for recipe_name in protocol["recipes"]
        }
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing fold checkpoints: {missing}")
        predictions, target, predicted_names, provenance = predict_fold(
            checkpoint_paths,
            loader,
            val_names,
            val_rois,
            data_root,
            str(protocol["marker"]),
            device,
            str(inference["tta"]),
            source_context_halo,
            recipe_context_halos,
        )
        if predicted_names != val_names:
            raise RuntimeError(f"Prediction order changed for {fold['name']}.")
        checkpoint_provenance[str(fold["name"])] = provenance
        fold_metrics[str(fold["name"])] = {}
        for candidate_name, weights in protocol["candidates"].items():
            candidate = weighted_prediction(predictions, weights)
            candidate = shift_prediction_edge(
                candidate,
                candidate_output_source_offset(protocol, candidate_name),
            )
            candidate_parts[candidate_name].append(candidate)
            fold_metrics[str(fold["name"])][candidate_name] = {
                "float": evaluate(candidate, target, 0),
                f"jpeg_q{inference['jpeg_quality']}": evaluate(
                    candidate, target, int(inference["jpeg_quality"])
                ),
            }
        target_parts.append(target)
        oof_names.extend(predicted_names)
        print(
            json.dumps(
                {"completed_fold": fold["name"], "validation_images": len(val_names)}
            ),
            flush=True,
        )

    if (
        len(oof_names) != len(set(oof_names))
        or set(oof_names) != set(evaluation_names)
    ):
        raise RuntimeError("OOF predictions do not cover each training image exactly once.")
    target = np.concatenate(target_parts)
    predictions = {
        candidate_name: np.concatenate(parts)
        for candidate_name, parts in candidate_parts.items()
    }

    test_roi_means = None
    test_like_range = None
    test_like_rois: set[str] = set()
    test_like_indices = None
    if test_like_diagnostic:
        train_roi_means, _ = roi_intensity_means(
            data_root / "train" / "DAPI", all_names
        )
        test_names = image_names(data_root / "test" / "DAPI")
        test_roi_means, _ = roi_intensity_means(data_root / "test" / "DAPI", test_names)
        test_min = min(test_roi_means.values())
        test_max = max(test_roi_means.values())
        test_like_range = [test_min, test_max]
        test_like_rois = {
            roi
            for roi, value in train_roi_means.items()
            if roi in expected_rois and test_min <= value <= test_max
        }
        test_like_indices = select_indices(oof_names, test_like_rois)
        if not len(test_like_indices):
            raise RuntimeError("The input-derived test-like OOF subset is empty.")

    overall = {}
    test_like = {}
    roi = {}
    for candidate_name, prediction in predictions.items():
        metrics = variant_metrics(
            prediction,
            target,
            oof_names,
            int(inference["jpeg_quality"]),
        )
        overall[candidate_name] = {
            "float": metrics["float"],
            f"jpeg_q{inference['jpeg_quality']}": metrics[f"jpeg_q{inference['jpeg_quality']}"],
        }
        roi[candidate_name] = metrics["roi"]
        if test_like_indices is not None:
            subset_prediction = prediction[test_like_indices]
            subset_target = target[test_like_indices]
            test_like[candidate_name] = {
                "images": int(len(test_like_indices)),
                "float": evaluate(subset_prediction, subset_target, 0),
                f"jpeg_q{inference['jpeg_quality']}": evaluate(
                    subset_prediction,
                    subset_target,
                    int(inference["jpeg_quality"]),
                ),
            }

    report = {
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["name"],
        "marker": protocol["marker"],
        "validation_images": len(oof_names),
        "validation_rois": sorted(expected_rois),
        "dataset_rois": sorted(actual_rois),
        "roi_scope": protocol.get("roi_scope", "complete"),
        "inference": inference,
        "checkpoint_filename": checkpoint_filename,
        "official_metric": {
            "score": "mean_i(100*(0.7*SSIM_i+0.3*clip(PSNR_i/50,0,1)))",
            "psnr_cap_db": 50,
            "aggregation": "per-image then mean",
            "ssim_implementation": "skimage.metrics.structural_similarity",
            "ssim_data_range": 255,
            "ssim_window": 7,
            "image_format": "RGB uint8; grayscale-equivalent channels in this task",
        },
        "baseline_candidate": protocol["baseline_candidate"],
        "primary_candidate": protocol["primary_candidate"],
        "test_dapi_roi_means": test_roi_means,
        "test_like_dapi_mean_range": test_like_range,
        "test_like_train_rois": sorted(test_like_rois),
        "checkpoint_provenance": checkpoint_provenance,
        "overall": overall,
        "test_like": test_like,
        "fold_metrics": fold_metrics,
        "roi_metrics": roi,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
