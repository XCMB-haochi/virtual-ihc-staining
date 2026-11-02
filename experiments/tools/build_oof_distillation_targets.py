from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import (
    inference_checkpoint_filename,
    load_protocol,
    recipe_inference_context_halo,
    sourced_run_directory,
)
from evaluate_cv_ensemble import predict_fold, sha256_file
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_torch import PairedPatchDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe OOF teacher targets for single-model distillation."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    candidate = protocol["candidates"].get(args.candidate)
    if not isinstance(candidate, dict) or len(candidate) != 1:
        raise ValueError("Distillation teacher candidate must contain exactly one recipe.")
    recipe_name, weight = next(iter(candidate.items()))
    if abs(float(weight) - 1.0) > 1e-12:
        raise ValueError("Distillation teacher candidate weight must be one.")
    if len(protocol["folds"]) != 5 or protocol.get("roi_scope") not in {"subset", "complete"}:
        raise ValueError("Distillation cache requires a five-fold grouped protocol.")

    data_root = resolve_data_root(args.dataset)
    all_names = image_names(data_root / "train" / "DAPI")
    expected_rois = set(protocol["expected_rois"])
    if {parse_patch_name(name).roi for name in all_names} != expected_rois:
        raise ValueError("Training ROI universe does not match the teacher protocol.")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite distillation cache: {output}")
    building = output.with_name(f".{output.name}.building")
    if building.exists():
        raise FileExistsError(f"Incomplete cache requires inspection: {building}")
    targets = building / "targets"
    targets.mkdir(parents=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    inference = protocol["inference"]
    source_halo = int(inference["context_halo"])
    recipe_halo = recipe_inference_context_halo(protocol, recipe_name)
    checkpoint_name = inference_checkpoint_filename(protocol)
    output_root = args.output_root.resolve()
    cached_names: list[str] = []
    folds: list[dict[str, object]] = []

    for fold in protocol["folds"]:
        fold_name = str(fold["name"])
        val_rois = set(fold["val_rois"])
        val_names = [
            name for name in all_names if parse_patch_name(name).roi in val_rois
        ]
        checkpoint = (
            sourced_run_directory(
                output_root, protocol, fold_name, str(recipe_name)
            )
            / checkpoint_name
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        loader = DataLoader(
            PairedPatchDataset(
                data_root,
                val_names,
                markers=(str(protocol["marker"]),),
                augment=False,
                context_halo=source_halo,
            ),
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        predictions, _, names, provenance = predict_fold(
            {str(recipe_name): checkpoint},
            loader,
            val_names,
            val_rois,
            data_root,
            str(protocol["marker"]),
            device,
            str(inference["tta"]),
            source_halo,
            {str(recipe_name): recipe_halo},
        )
        if names != val_names:
            raise RuntimeError(f"Teacher prediction order changed for {fold_name}.")
        values = np.clip(
            np.rint(predictions[str(recipe_name)]), 0, 255
        ).astype(np.uint8)
        for name, value in zip(names, values):
            Image.fromarray(value, mode="L").save(
                targets / name, format="PNG", optimize=False
            )
        cached_names.extend(names)
        folds.append(
            {
                "name": fold_name,
                "val_rois": sorted(val_rois),
                "images": len(names),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "provenance": provenance[str(recipe_name)],
            }
        )
        print(json.dumps({"fold": fold_name, "images": len(names)}), flush=True)

    if len(cached_names) != len(set(cached_names)) or set(cached_names) != set(all_names):
        raise RuntimeError("OOF teacher targets do not cover every training image once.")
    manifest = {
        "version": 1,
        "method": "fold-excluded-single-recipe-d4-teacher",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_name": protocol["name"],
        "candidate": args.candidate,
        "recipe": recipe_name,
        "marker": protocol["marker"],
        "tta": inference["tta"],
        "output_source_offset_applied": False,
        "storage": "uint8 PNG bytes with original .jpg patch names",
        "images": len(cached_names),
        "folds": folds,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (building / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    building.replace(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
