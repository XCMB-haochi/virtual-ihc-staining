"""Bounded CPU-only audit of existing pipeline contracts; no training or selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_spatial import apply_input_conditioning, spatial_thumbnail_conditioning_specification
from ihc_torch import (
    PairedPatchDataset, batch_condition_tensor, build_model_from_checkpoint,
    center_crop_like, predict_with_tta, shift_tensor_edge,
)
from search_postprocessing import evaluate
from train_unet import validate


class Identity(torch.nn.Module):
    def forward(self, inputs, condition=None):
        return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    started = time.perf_counter()
    data_root = resolve_data_root(ROOT / "dataset")
    all_names = image_names(data_root / "train/DAPI")
    protocol = json.loads((ROOT / "configs/round113_cv2_dino_residual.json").read_text())
    old = json.loads((ROOT / "reports/round113_dino_residual_knockout.json").read_text())
    selected = old["names"]
    spec = spatial_thumbnail_conditioning_specification()
    all_conditions = apply_input_conditioning(data_root / "train/DAPI", all_names, spec)
    validation_names = [n for n in all_names if parse_patch_name(n).roi in protocol["expected_rois"]]
    validation_conditions = apply_input_conditioning(data_root / "train/DAPI", validation_names, spec)
    sparse_conditions = apply_input_conditioning(data_root / "train/DAPI", selected, spec)
    report = {
        "formal": False, "device": "cpu", "training": False,
        "conditioning": {
            "formal_validation_images": len(validation_names),
            "formal_vs_all_roi_max_abs": max(float(np.abs(all_conditions[n] - validation_conditions[n]).max()) for n in validation_names),
            "sparse_images": len(selected),
            "sparse_changed": sum(not np.array_equal(all_conditions[n], sparse_conditions[n]) for n in selected),
            "sparse_condition_mean_abs": float(np.mean([np.abs(all_conditions[n] - sparse_conditions[n]).mean() for n in selected])),
        },
    }
    print(json.dumps(report["conditioning"]), flush=True)

    prediction = torch.stack((torch.zeros(1, 32, 32), torch.full((1, 32, 32), .5)))
    target = torch.zeros_like(prediction)
    with np.errstate(divide="ignore"):
        fixed = validate(Identity(), [(prediction, target, ["a", "b"])], torch.device("cpu"), ("CD68",), 0)
        official = evaluate(prediction[:, 0].numpy() * 255, target[:, 0].numpy() * 255, 0)
    report["synthetic_metric"] = {
        "before_fix": official["competition_score_mean_image_psnr"],
        "after_fix": fixed["mean_competition_score"],
        "official": official["competition_score_official"],
    }

    # Fixed first and last images of fold0 in the previous 80-image sample.
    fold = protocol["folds"][0]
    pool = [n for n in selected if parse_patch_name(n).roi in fold["val_rois"]]
    query = [pool[0], pool[-1]]
    path = ROOT / "runs/round55_cv2_unext2_official_auxiliary_fold0_officialauxiliary15_stage2/best_global_mse.pt"
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ck["args"]["val_rois"].split(",")) == set(fold["val_rois"])
    model = build_model_from_checkpoint(ck, torch.device("cpu")).eval()
    dataset = PairedPatchDataset(data_root, query, markers=("CD68",), context_halo=128)
    batches = [dataset[i] for i in range(len(query))]
    source = torch.stack([b[0] for b in batches])
    truth = torch.stack([b[1] for b in batches])
    predictions = {}
    with torch.no_grad():
        for name, conditions in (("full_roi", all_conditions), ("sparse_roi", sparse_conditions)):
            value = center_crop_like(predict_with_tta(model, source, "d4", batch_condition_tensor(query, conditions, source.device)), truth)[:, :1]
            predictions[name] = shift_tensor_edge(value, (0, -1))[:, 0].numpy() * 255
    report["two_image_probe"] = {
        "purpose": "Verify input-contract effect only; not an accuracy estimate or model selection.",
        "names": query, "checkpoint_sha256": checksum,
        "prediction_rms_uint8": float(np.sqrt(np.mean((predictions["full_roi"] - predictions["sparse_roi"]) ** 2))),
        "q95": {name: evaluate(value, np.rint(truth[:, 0].numpy() * 255).astype(np.uint8), 95)["competition_score_official"] for name, value in predictions.items()},
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/ihc_spatial.py", "tools/train_unet.py", "tools/reassess_pipeline.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
