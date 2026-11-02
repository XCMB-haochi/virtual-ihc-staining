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

from cv_protocol import (
    inference_checkpoint_filename,
    recipe_inference_context_halo,
    sourced_run_directory,
)
from evaluate_cv_ensemble import predict_fold, weighted_prediction
from ihc_data import image_names, load_gray, parse_patch_name, resolve_data_root
from ihc_torch import PairedPatchDataset
from refiner_protocol import (
    load_base_protocol,
    load_refiner_protocol,
    sha256_file,
)
from search_postprocessing import evaluate


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_existing_fold(
    fold_dir: Path,
    fold_name: str,
    recipe_order: list[str],
    weights: dict[str, float],
    expected_rois: set[str],
    checkpoint_paths: dict[str, Path],
) -> dict[str, object]:
    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Refusing to reuse incomplete cache directory: {fold_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fold") != fold_name:
        raise ValueError(f"Cache fold identity mismatch in {manifest_path}.")
    if manifest.get("recipe_order") != recipe_order:
        raise ValueError(f"Cache recipe order mismatch in {manifest_path}.")
    if manifest.get("baseline_weights") != weights:
        raise ValueError(f"Cache baseline weights mismatch in {manifest_path}.")
    if set(manifest.get("val_rois", [])) != expected_rois:
        raise ValueError(f"Cache validation ROI mismatch in {manifest_path}.")
    provenance = manifest.get("checkpoint_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Cache checkpoint provenance is missing in {manifest_path}.")
    for name, checkpoint_path in checkpoint_paths.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing base checkpoint: {checkpoint_path}")
        cached = provenance.get(name)
        if not isinstance(cached, dict) or cached.get("sha256") != sha256_file(
            checkpoint_path
        ):
            raise ValueError(
                f"Cache checkpoint hash mismatch for {fold_name}/{name}."
            )
    names = manifest.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Cache fold {fold_name} has no names.")
    expected_shape = (len(names), 256, 256)
    arrays = {
        "dapi": (fold_dir / "dapi.npy", np.uint8),
        "target": (fold_dir / "target.npy", np.uint8),
        **{
            f"prediction:{name}": (
                fold_dir / f"prediction_{name}.npy",
                np.float32,
            )
            for name in recipe_order
        },
    }
    for label, (path, dtype) in arrays.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached {label}: {path}")
        array = np.load(path, mmap_mode="r")
        if array.shape != expected_shape or array.dtype != dtype:
            raise ValueError(
                f"Cached {label} has {array.shape}/{array.dtype}; expected "
                f"{expected_shape}/{np.dtype(dtype)}."
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache leakage-safe fold-specific OOF predictions for a refiner."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_refiner_protocol(protocol_path)
    base_protocol = load_base_protocol(protocol)
    data_root = resolve_data_root(args.dataset)
    all_names = image_names(data_root / "train" / "DAPI")
    actual_rois = {parse_patch_name(name).roi for name in all_names}
    if actual_rois != set(base_protocol["expected_rois"]):
        raise ValueError("Dataset ROI set does not match the base CV protocol.")
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    root_manifest_path = cache_root / "manifest.json"
    recipe_order = list(protocol["recipe_order"])
    weights = dict(protocol["baseline_weights"])
    output_root = args.output_root.resolve()
    checkpoint_filename = inference_checkpoint_filename(base_protocol)
    if root_manifest_path.is_file():
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        if (
            root_manifest.get("refiner_protocol_sha256")
            != sha256_file(protocol_path)
            or root_manifest.get("recipe_order") != recipe_order
            or root_manifest.get("baseline_weights") != weights
        ):
            raise ValueError("Existing OOF cache does not match the refiner protocol.")
        for fold in base_protocol["folds"]:
            fold_name = str(fold["name"])
            checkpoint_paths = {
                name: sourced_run_directory(
                    output_root,
                    base_protocol,
                    fold_name,
                    name,
                )
                / checkpoint_filename
                for name in recipe_order
            }
            validate_existing_fold(
                cache_root / fold_name,
                fold_name,
                recipe_order,
                weights,
                set(fold["val_rois"]),
                checkpoint_paths,
            )
        print(json.dumps(root_manifest, ensure_ascii=False, indent=2))
        return
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    inference = base_protocol["inference"]
    source_context_halo = int(inference["context_halo"])
    recipe_context_halos = {
        name: recipe_inference_context_halo(base_protocol, name)
        for name in recipe_order
    }
    fold_manifests: list[dict[str, object]] = []

    for fold in base_protocol["folds"]:
        fold_name = str(fold["name"])
        fold_dir = cache_root / fold_name
        val_rois = set(fold["val_rois"])
        checkpoint_paths = {
            name: sourced_run_directory(
                output_root,
                base_protocol,
                fold_name,
                name,
            )
            / checkpoint_filename
            for name in recipe_order
        }
        if fold_dir.exists():
            manifest = validate_existing_fold(
                fold_dir,
                fold_name,
                recipe_order,
                weights,
                val_rois,
                checkpoint_paths,
            )
            fold_manifests.append(manifest)
            print(json.dumps({"reused_fold": fold_name}), flush=True)
            continue
        building_dir = cache_root / f".{fold_name}.building"
        if building_dir.exists():
            raise FileExistsError(
                f"Incomplete temporary cache requires inspection: {building_dir}"
            )
        building_dir.mkdir()
        val_names = [
            name for name in all_names if parse_patch_name(name).roi in val_rois
        ]
        loader = DataLoader(
            PairedPatchDataset(
                data_root,
                val_names,
                markers=(str(protocol["marker"]),),
                augment=False,
                context_halo=source_context_halo,
            ),
            batch_size=int(inference["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing base checkpoints: {missing}")
        predictions, target, names, provenance = predict_fold(
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
        if names != val_names:
            raise RuntimeError(f"Prediction order changed for {fold_name}.")
        dapi = np.stack(
            [
                np.rint(load_gray(data_root / "train" / "DAPI" / name)).astype(
                    np.uint8
                )
                for name in names
            ]
        )
        np.save(building_dir / "dapi.npy", dapi, allow_pickle=False)
        np.save(building_dir / "target.npy", target, allow_pickle=False)
        for name in recipe_order:
            np.save(
                building_dir / f"prediction_{name}.npy",
                predictions[name].astype(np.float32, copy=False),
                allow_pickle=False,
            )
        baseline = weighted_prediction(predictions, weights)
        fold_metrics = {
            "float": evaluate(baseline, target, 0),
            f"jpeg_q{inference['jpeg_quality']}": evaluate(
                baseline, target, int(inference["jpeg_quality"])
            ),
        }
        manifest = {
            "version": 1,
            "fold": fold_name,
            "val_rois": sorted(val_rois),
            "images": len(names),
            "names": names,
            "recipe_order": recipe_order,
            "baseline_weights": weights,
            "checkpoint_provenance": provenance,
            "baseline_metrics": fold_metrics,
        }
        write_json(building_dir / "manifest.json", manifest)
        building_dir.replace(fold_dir)
        fold_manifests.append(manifest)
        del dapi, target, predictions, baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps({"completed_fold": fold_name, "images": len(names)}),
            flush=True,
        )

    cached_names = [
        name for manifest in fold_manifests for name in manifest["names"]
    ]
    if len(cached_names) != len(set(cached_names)) or set(cached_names) != set(all_names):
        raise RuntimeError("OOF cache does not cover every training image exactly once.")
    root_manifest = {
        "version": 1,
        "refiner_protocol": str(protocol_path),
        "refiner_protocol_sha256": sha256_file(protocol_path),
        "protocol_name": protocol["name"],
        "base_protocol_name": base_protocol["name"],
        "base_candidate": protocol["base_candidate"],
        "recipe_order": recipe_order,
        "baseline_weights": weights,
        "images": len(cached_names),
        "folds": [
            {
                "name": manifest["fold"],
                "val_rois": manifest["val_rois"],
                "images": manifest["images"],
            }
            for manifest in fold_manifests
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(root_manifest_path, root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
