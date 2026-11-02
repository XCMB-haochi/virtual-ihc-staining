"""Fixed-checkpoint OOF audit; never fits parameters or evaluates test labels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ihc_data import image_names, load_gray, parse_patch_name, resolve_data_root

MARKERS = ("CD68", "CD45RO", "HLA-DR", "Vimentin")
PROFILES = {
    "uniform7": dict(win_size=7, gaussian_weights=False, use_sample_covariance=True),
    "gaussian11": dict(win_size=11, gaussian_weights=True, sigma=1.5,
                       use_sample_covariance=False),
}


def image_metrics(prediction, target):
    """Match submitted RGB JPEG encoding, then score identical decoded pixels."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected matching grayscale images")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite image")
    buffer = io.BytesIO()
    gray = np.clip(np.rint(prediction), 0, 255).astype(np.uint8)
    Image.fromarray(gray).convert("RGB").save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as image:
        decoded = np.asarray(image.convert("L"), dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((target - decoded) ** 2))
    psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
    result = {"mse": mse, "psnr": psnr, "psnr_capped": min(50., max(0., psnr))}
    for profile, options in PROFILES.items():
        ssim = float(structural_similarity(target, decoded, data_range=255, **options))
        result[profile + "_ssim"] = ssim
        result[profile + "_score"] = 70 * ssim + .6 * result["psnr_capped"]
    return result


def summarize(rows):
    if not rows:
        raise ValueError("Empty metric collection")
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result["images"] = len(rows)
    result["psnr_global_mse"] = float(10 * np.log10(255 ** 2 / max(result["mse"], 1e-10)))
    return result


def fold_plan(manifest, names):
    assignment = {}
    for fold in manifest["folds"]:
        for roi in fold["val_rois"]:
            if roi in assignment:
                raise ValueError(f"Repeated validation ROI: {roi}")
            assignment[roi] = fold["name"]
    if set(assignment) != {parse_patch_name(name).roi for name in names}:
        raise ValueError("OOF universe differs from dataset")
    if len(names) != manifest["images"]:
        raise ValueError("OOF image count mismatch")
    for fold in manifest["folds"]:
        count = sum(assignment[parse_patch_name(n).roi] == fold["name"] for n in names)
        if count != fold["images"]:
            raise ValueError("Fold image count mismatch")
    return assignment


def cached_predictions(data_root, cache, names, assignment):
    if set(image_names(cache / "targets")) != set(names):
        raise ValueError("Cache image universe mismatch")
    for name in names:
        yield name, assignment[parse_patch_name(name).roi], "CD68", load_gray(cache / "targets" / name), load_gray(data_root / "train" / "CD68" / name)


def model_predictions(data_root, manifest, names, device_name):
    import torch
    from torch.utils.data import DataLoader
    from evaluate_cv_ensemble import validate_checkpoint_fold
    from ihc_spatial import apply_complete_roi_conditioning
    from ihc_torch import (PairedPatchDataset, batch_condition_tensor,
                           build_model_from_checkpoint, center_crop_like,
                           checkpoint_conditioning, predict_with_tta)

    torch.manual_seed(2026)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    for fold in manifest["folds"]:
        path = Path(fold["checkpoint"])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != fold["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for marker in MARKERS:
            validate_checkpoint_fold(checkpoint, set(fold["val_rois"]), marker, path)
        if checkpoint["args"].get("train_all", False):
            raise ValueError("Full-data model cannot supply OOF predictions")
        markers = tuple(checkpoint["markers"])
        if set(markers) != set(MARKERS) or len(markers) != 4:
            raise ValueError("Unexpected model output channels")
        model = build_model_from_checkpoint(checkpoint, device).eval()
        val_names = [n for n in names if parse_patch_name(n).roi in fold["val_rois"]]
        conditions = apply_complete_roi_conditioning(data_root / "train" / "DAPI", val_names, checkpoint_conditioning(checkpoint))
        loader = DataLoader(PairedPatchDataset(data_root, val_names, markers=markers, augment=False, context_halo=128), batch_size=2, shuffle=False, num_workers=2)
        print(f"Starting {fold['name']}: epoch={checkpoint['epoch']}, images={len(val_names)}", flush=True)
        with torch.inference_mode():
            for source, target, batch_names in loader:
                condition = batch_condition_tensor(batch_names, conditions, device)
                prediction = center_crop_like(predict_with_tta(model, source.to(device), "d4", condition), target).cpu().numpy() * 255.
                truth = np.rint(target.numpy() * 255.).astype(np.uint8)
                for index, name in enumerate(batch_names):
                    for channel, marker in enumerate(markers):
                        yield name, fold["name"], marker, prediction[index, channel], truth[index, channel]
        del model, checkpoint, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("cache", "models"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subset-rois", help="Explicit training-ROI subset for a preregistered screen")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    data_root = resolve_data_root(args.dataset)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest["method"] != "fold-excluded-single-recipe-d4-teacher" or manifest["output_source_offset_applied"]:
        raise ValueError("Unexpected teacher cache recipe")
    names = image_names(data_root / "train" / "DAPI")
    if args.subset_rois:
        selected = set(args.subset_rois.split(','))
        if not selected or not selected <= {parse_patch_name(n).roi for n in names}:
            raise ValueError('Unknown subset ROI')
        names = [n for n in names if parse_patch_name(n).roi in selected]
    assignment = fold_plan(manifest, names)
    items = cached_predictions(data_root, args.cache, names, assignment) if args.mode == "cache" else model_predictions(data_root, manifest, names, args.device)
    groups = defaultdict(list)
    seen = set()
    # Bounded batches avoid retaining all 4-channel predictions in host memory.
    batch = []
    def consume(pool):
        values = list(pool.map(lambda item: image_metrics(item[3], item[4]), batch))
        for item, metrics in zip(batch, values):
            name, fold, marker = item[:3]
            key = (marker, name)
            if key in seen:
                raise ValueError(f"Duplicate OOF prediction: {key}")
            seen.add(key)
            roi = parse_patch_name(name).roi
            for scope in ("overall", fold, roi):
                groups[(marker, scope)].append(metrics)
        batch.clear()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in items:
            batch.append(item)
            if len(batch) == 16:
                consume(pool)
        if batch:
            consume(pool)
    markers = ("CD68",) if args.mode == "cache" else MARKERS
    if seen != {(m, n) for m in markers for n in names}:
        raise ValueError("Incomplete OOF coverage")
    report = {
        "mode": args.mode, "elapsed_seconds": time.monotonic() - started,
        "subset_rois": args.subset_rois,
        "metric_profiles": PROFILES,
        "assumptions": "Gaussian sigma=1.5, population covariance, valid 11x11 window; organizer confirmed Gaussian11 only. Score retains preliminary 70*SSIM+0.6*clipped_PSNR pending confirmation.",
        "inference": {"tta": "d4", "halo": 128, "offset": [0, 0], "jpeg_quality": 95},
        "limitations": "Fixed CD68-selected fold checkpoints; not an unbiased comparison of separately optimized marker models. Cache mode uses rounded uint8 teacher predictions, models mode uses original float outputs before JPEG.",
        "fold_sources": [{k: f[k] for k in ("name", "val_rois", "checkpoint", "checkpoint_sha256")} for f in manifest["folds"]],
        "markers": {m: {scope: summarize(rows) for (marker, scope), rows in groups.items() if marker == m} for m in markers},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({m: report["markers"][m]["overall"] for m in markers}, indent=2), flush=True)


if __name__ == "__main__":
    main()
