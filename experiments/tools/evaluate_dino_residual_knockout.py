from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_cv_ensemble as cv
from cv_protocol import load_protocol, sourced_run_directory
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_torch import PairedPatchDataset, shift_array_edge
from ihc_unext2 import UNeXt2VirtualStainer
from search_postprocessing import evaluate, evaluate_by_roi


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed ROI-stratified DINO output-branch knockout; diagnostic only.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(ROOT / "configs/round113_cv2_dino_residual.json")
    data_root = resolve_data_root(ROOT / "dataset")
    all_names = image_names(data_root / "train/DAPI")
    selected = []
    for roi in sorted(protocol["expected_rois"]):
        pool = sorted(name for name in all_names if parse_patch_name(name).roi == roi)
        if len(pool) < 8:
            raise ValueError(f"ROI {roi} has fewer than eight images.")
        selected.extend(pool[int(i)] for i in np.linspace(0, len(pool) - 1, 8, dtype=int))
    builder = cv.build_model_from_checkpoint

    def without_residual(checkpoint, device):
        model = builder(checkpoint, device)
        if not hasattr(model, "dinov2_residual"):
            raise TypeError("Expected a DINO residual checkpoint.")
        model.forward = MethodType(UNeXt2VirtualStainer.forward, model)
        return model

    predictions = {"enabled": [], "disabled": []}
    targets, names, provenance = [], [], {}
    for fold in protocol["folds"]:
        fold_name = fold["name"]
        val_names = [name for name in selected if parse_patch_name(name).roi in fold["val_rois"]]
        checkpoint = sourced_run_directory(ROOT / "runs", protocol, fold_name, "dinov2residual") / protocol["inference"]["checkpoint_filename"]
        loader = DataLoader(PairedPatchDataset(data_root, val_names, markers=("CD68",),
            augment=False, context_halo=128), batch_size=4, shuffle=False, num_workers=0)
        for mode in predictions:
            cv.build_model_from_checkpoint = builder if mode == "enabled" else without_residual
            try:
                output, target, output_names, source = cv.predict_fold(
                    {"dinov2residual": checkpoint}, loader, val_names, set(fold["val_rois"]),
                    data_root, "CD68", torch.device("cuda:0"), "d4", 128, {"dinov2residual": 128})
            finally:
                cv.build_model_from_checkpoint = builder
            if output_names != val_names:
                raise RuntimeError("Diagnostic prediction order changed.")
            predictions[mode].append(shift_array_edge(output["dinov2residual"], (0, -1)))
            if mode == "enabled":
                targets.append(target)
                names.extend(output_names)
                provenance[fold_name] = source
        print(f"Completed diagnostic {fold_name}", flush=True)
    target = np.concatenate(targets)
    arrays = {mode: np.concatenate(parts) for mode, parts in predictions.items()}
    report = {
        "formal": False,
        "conditioning_scope": "complete DAPI ROIs; predictions sampled only",
        "purpose": "Determine reliance on the learned DINO output branch; no candidate selection or tuning.",
        "selection": "Eight evenly spaced sorted filenames per each of the ten registered ROIs, fixed before evaluation.",
        "names": names,
        "validation_images": len(names),
        "overall": {mode: {"float": evaluate(array, target, 0), "jpeg_q95": evaluate(array, target, 95)}
                    for mode, array in arrays.items()},
        "roi_q95": {mode: evaluate_by_roi(array, target, names, 95) for mode, array in arrays.items()},
        "branch_rms_uint8": float(np.sqrt(np.mean((arrays["enabled"] - arrays["disabled"]) ** 2))),
        "checkpoint_provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("validation_images", "overall", "branch_rms_uint8")}), flush=True)


if __name__ == "__main__":
    main()
