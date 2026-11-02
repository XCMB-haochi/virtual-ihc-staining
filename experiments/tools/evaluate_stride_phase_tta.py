from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    sourced_run_directory,
)
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning
from ihc_torch import (
    PairedPatchDataset,
    batch_condition_tensor,
    build_model_from_checkpoint,
    center_crop_like,
    checkpoint_conditioning,
    crop_context_halo,
    predict_with_stride_phase_tta,
    predict_with_tta,
    shift_array_edge,
)
from search_postprocessing import evaluate, evaluate_by_roi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standard D4 with aligned four-phase D4 self-ensemble."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images-per-fold", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    if protocol["name"] != "round102_cv2_stride_phase_tta":
        raise ValueError("This evaluator is bound to the Round102 protocol.")
    phase_spec = protocol.get("phase_tta")
    expected_phase_spec = {
        "model_context_halo": 128,
        "outer_margin": 1,
        "phase_offsets": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "alignment": "crop-to-fixed-center-before-average",
    }
    if phase_spec != expected_phase_spec:
        raise ValueError("Round102 phase specification changed.")
    if args.max_images_per_fold < 0:
        raise ValueError("--max-images-per-fold must be non-negative.")

    data_root = resolve_data_root(args.dataset)
    all_names = image_names(data_root / "train" / "DAPI")
    expected_rois = set(protocol["expected_rois"])
    evaluation_names = [
        name for name in all_names if parse_patch_name(name).roi in expected_rois
    ]
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    inference = protocol["inference"]
    outer_halo = int(inference["context_halo"])
    model_halo = int(phase_spec["model_context_halo"])
    jpeg_quality = int(inference["jpeg_quality"])
    checkpoint_filename = inference_checkpoint_filename(protocol)
    output_root = args.output_root.resolve()

    candidates = ("standard_d4_context128", "stride_phase4_d4_context128")
    collected: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    target_parts: list[np.ndarray] = []
    oof_names: list[str] = []
    fold_metrics: dict[str, object] = {}
    provenance: dict[str, object] = {}

    for fold in protocol["folds"]:
        fold_name = str(fold["name"])
        val_rois = set(fold["val_rois"])
        val_names = [
            name
            for name in evaluation_names
            if parse_patch_name(name).roi in val_rois
        ]
        if args.max_images_per_fold:
            val_names = val_names[: args.max_images_per_fold]
        checkpoint_path = (
            sourced_run_directory(
                output_root,
                protocol,
                fold_name,
                "standardauxiliary15",
            )
            / checkpoint_filename
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        markers = tuple(checkpoint.get("markers", ()))
        if protocol["marker"] not in markers:
            raise ValueError(f"Checkpoint is missing {protocol['marker']}.")
        arguments = checkpoint.get("args")
        if not isinstance(arguments, dict):
            raise ValueError("Checkpoint has no argument metadata.")
        actual_val_rois = {
            item.strip().upper()
            for item in str(arguments.get("val_rois", "")).split(",")
            if item.strip()
        }
        if actual_val_rois != val_rois:
            raise ValueError(f"Checkpoint fold mismatch for {fold_name}.")
        marker_index = markers.index(protocol["marker"])
        conditioning = checkpoint_conditioning(checkpoint)
        condition_by_name = apply_input_conditioning(
            data_root / "train" / "DAPI", val_names, conditioning
        )
        model = build_model_from_checkpoint(checkpoint, device).eval()
        loader = DataLoader(
            PairedPatchDataset(
                data_root,
                val_names,
                markers=(protocol["marker"],),
                augment=False,
                context_halo=outer_halo,
            ),
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        fold_predictions: dict[str, list[np.ndarray]] = {
            name: [] for name in candidates
        }
        fold_targets: list[np.ndarray] = []
        predicted_names: list[str] = []
        with torch.no_grad():
            for outer_source, target, batch_names in loader:
                outer_source = outer_source.to(device, non_blocking=True)
                condition = batch_condition_tensor(
                    batch_names, condition_by_name, device
                )
                standard_source = crop_context_halo(
                    outer_source, outer_halo, model_halo
                )
                standard = center_crop_like(
                    predict_with_tta(
                        model,
                        standard_source,
                        str(inference["tta"]),
                        condition,
                    ),
                    target,
                )[:, marker_index]
                phase = predict_with_stride_phase_tta(
                    model,
                    outer_source,
                    target.shape[-2:],
                    model_halo,
                    str(inference["tta"]),
                    condition,
                )[:, marker_index]
                fold_predictions[candidates[0]].append(
                    standard.cpu().numpy().astype(np.float32) * 255.0
                )
                fold_predictions[candidates[1]].append(
                    phase.cpu().numpy().astype(np.float32) * 255.0
                )
                fold_targets.append(
                    np.rint(target.numpy()[:, 0] * 255.0).astype(np.uint8)
                )
                predicted_names.extend(batch_names)

        if predicted_names != val_names:
            raise RuntimeError(f"Prediction order changed for {fold_name}.")
        target_array = np.concatenate(fold_targets)
        fold_metrics[fold_name] = {}
        for candidate in candidates:
            prediction = np.concatenate(fold_predictions[candidate])
            prediction = shift_array_edge(
                prediction,
                candidate_output_source_offset(protocol, candidate),
            )
            collected[candidate].append(prediction)
            fold_metrics[fold_name][candidate] = {
                "float": evaluate(prediction, target_array, 0),
                f"jpeg_q{jpeg_quality}": evaluate(
                    prediction, target_array, jpeg_quality
                ),
            }
        target_parts.append(target_array)
        oof_names.extend(predicted_names)
        provenance[fold_name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "conditioning": conditioning,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {"completed_fold": fold_name, "validation_images": len(val_names)}
            ),
            flush=True,
        )

    if not args.max_images_per_fold and (
        len(oof_names) != len(set(oof_names))
        or set(oof_names) != set(evaluation_names)
    ):
        raise RuntimeError("Round102 OOF coverage is incomplete.")
    target = np.concatenate(target_parts)
    overall: dict[str, object] = {}
    roi_metrics: dict[str, object] = {}
    for candidate in candidates:
        prediction = np.concatenate(collected[candidate])
        metrics = variant_metrics(prediction, target, oof_names, jpeg_quality)
        overall[candidate] = {
            "float": metrics["float"],
            f"jpeg_q{jpeg_quality}": metrics[f"jpeg_q{jpeg_quality}"],
        }
        roi_metrics[candidate] = metrics["roi"]

    report = {
        "version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["name"],
        "marker": protocol["marker"],
        "validation_images": len(oof_names),
        "validation_rois": sorted(
            {parse_patch_name(name).roi for name in oof_names}
        ),
        "phase_tta": phase_spec,
        "checkpoint_provenance": provenance,
        "overall": overall,
        "fold_metrics": fold_metrics,
        "roi_metrics": roi_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
