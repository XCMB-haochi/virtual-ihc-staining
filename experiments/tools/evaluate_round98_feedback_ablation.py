from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import evaluate_cv_ensemble
from cv_protocol import (
    candidate_output_source_offset,
    inference_checkpoint_filename,
    load_protocol,
    recipe_inference_context_halo,
    sourced_run_directory,
)
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_torch import PairedPatchDataset, shift_array_edge
from search_postprocessing import evaluate, evaluate_by_roi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Round98 with its learned feedback projection disabled."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    if protocol["name"] != "round98_cv2_cross_marker_feedback":
        raise ValueError("This diagnostic is bound to the Round98 protocol.")
    recipe_name = "crossmarkerfeedback15"
    checkpoint_filename = inference_checkpoint_filename(protocol)
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

    original_builder = evaluate_cv_ensemble.build_model_from_checkpoint

    def build_without_feedback(checkpoint: dict[str, object], target_device: torch.device):
        model = original_builder(checkpoint, target_device)
        feedback_head = getattr(model, "feedback_head", None)
        if feedback_head is None or not isinstance(feedback_head[-1], torch.nn.Conv2d):
            raise TypeError("Round98 checkpoint does not expose feedback_head.")
        with torch.no_grad():
            feedback_head[-1].weight.zero_()
            feedback_head[-1].bias.zero_()
        return model

    evaluate_cv_ensemble.build_model_from_checkpoint = build_without_feedback
    candidate_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    names: list[str] = []
    fold_metrics: dict[str, object] = {}
    checkpoint_records: dict[str, object] = {}
    try:
        for fold in protocol["folds"]:
            fold_name = str(fold["name"])
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
                    context_halo=int(protocol["inference"]["context_halo"]),
                ),
                batch_size=int(protocol["inference"]["batch_size"]),
                shuffle=False,
                num_workers=0,
            )
            checkpoint = (
                sourced_run_directory(
                    args.output_root.resolve(), protocol, fold_name, recipe_name
                )
                / checkpoint_filename
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            predictions, target, predicted_names, provenance = evaluate_cv_ensemble.predict_fold(
                {recipe_name: checkpoint},
                loader,
                val_names,
                val_rois,
                data_root,
                str(protocol["marker"]),
                device,
                str(protocol["inference"]["tta"]),
                int(protocol["inference"]["context_halo"]),
                {recipe_name: recipe_inference_context_halo(protocol, recipe_name)},
            )
            if predicted_names != val_names:
                raise RuntimeError(f"Prediction order changed for {fold_name}.")
            prediction = shift_array_edge(
                predictions[recipe_name],
                candidate_output_source_offset(protocol, "crossmarkerfeedback15_context128"),
            )
            candidate_parts.append(prediction)
            target_parts.append(target)
            names.extend(predicted_names)
            checkpoint_records[fold_name] = {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                "provenance": provenance[recipe_name],
            }
            fold_metrics[fold_name] = {
                "float": evaluate(prediction, target, 0),
                "jpeg_q95": evaluate(prediction, target, 95),
            }
    finally:
        evaluate_cv_ensemble.build_model_from_checkpoint = original_builder

    target = np.concatenate(target_parts)
    prediction = np.concatenate(candidate_parts)
    report = {
        "version": 1,
        "protocol_name": protocol["name"],
        "mode": "zero-learned-feedback-projection",
        "validation_images": len(names),
        "validation_rois": sorted(expected_rois),
        "overall": {
            "float": evaluate(prediction, target, 0),
            "jpeg_q95": evaluate(prediction, target, 95),
        },
        "roi_metrics": {
            "float": evaluate_by_roi(prediction, target, names, 0),
            "jpeg_q95": evaluate_by_roi(prediction, target, names, 95),
        },
        "fold_metrics": fold_metrics,
        "checkpoint_provenance": checkpoint_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
