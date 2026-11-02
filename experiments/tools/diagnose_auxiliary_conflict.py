from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from cv_protocol import load_protocol, run_directory
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_spatial import apply_complete_roi_conditioning
from ihc_torch import (
    CompetitionLoss, PairedPatchDataset, batch_condition_tensor,
    build_model_from_checkpoint, checkpoint_conditioning,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure main/auxiliary gradient conflict on fixed training-only batches.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Use cpu for a bounded audit without occupying a GPU.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    protocol = load_protocol(ROOT / "configs/round55_cv2_unext2_official_auxiliary.json")
    data_root = resolve_data_root(ROOT / "dataset")
    all_names = image_names(data_root / "train/DAPI")
    criterion = CompetitionLoss(ssim_weight=0.7, l1_weight=0.2, mse_weight=0.1, ssim_mode="uniform7").to(device)
    records = []
    started = time.perf_counter()
    for fold in protocol["folds"]:
        pool = [n for n in all_names if parse_patch_name(n).roi not in fold["val_rois"]]
        selected = sorted(np.random.default_rng(2044).choice(pool, size=64, replace=False).tolist())
        loader = DataLoader(PairedPatchDataset(data_root, selected,
            markers=("CD68", "CD45RO", "HLA-DR", "Vimentin"), augment=False,
            context_halo=64, target_context_halo=64), batch_size=16, shuffle=False, num_workers=0)
        for stage in (1, 2):
            path = run_directory(ROOT / "runs", protocol["name"], fold["name"], "officialauxiliary15", stage) / "best_global_mse.pt"
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert set(checkpoint["args"]["val_rois"].split(",")) == set(fold["val_rois"])
            model = build_model_from_checkpoint(checkpoint, device).eval()
            conditions = apply_complete_roi_conditioning(data_root / "train/DAPI", selected, checkpoint_conditioning(checkpoint))
            named = list(model.named_parameters())
            for index, (source, target, names) in enumerate(loader):
                source, target = source.to(device), target.to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                    prediction = model(source, batch_condition_tensor(names, conditions, source.device))
                    primary = .85 * criterion(prediction[:, :1], target[:, :1])
                    auxiliary = .15 * criterion(prediction[:, 1:], target[:, 1:])
                # Scale both losses equally so small auxiliary gradients survive AMP.
                gradient_scale = 1024.0 if device.type == "cuda" else 1.0
                gp = torch.autograd.grad(primary * gradient_scale, [p for _, p in named], retain_graph=True)
                ga = torch.autograd.grad(auxiliary * gradient_scale, [p for _, p in named])
                gp = tuple(g.float() / gradient_scale for g in gp)
                ga = tuple(g.float() / gradient_scale for g in ga)
                if not all(bool(torch.isfinite(g).all()) for g in (*gp, *ga)):
                    raise RuntimeError("Non-finite scaled diagnostic gradient.")
                metrics = {}
                for scope in ("all", "encoder", "decoder"):
                    pairs = [(a, b) for (name, _), a, b in zip(named, gp, ga)
                             if scope == "all" or name.startswith("encoder.") == (scope == "encoder")]
                    dot = sum((a.float() * b.float()).sum() for a, b in pairs).item()
                    p2 = sum(a.float().square().sum() for a, _ in pairs).item()
                    a2 = sum(b.float().square().sum() for _, b in pairs).item()
                    metrics[scope] = {"dot": dot, "primary_squared_norm": p2,
                        "auxiliary_squared_norm": a2, "cosine": dot / max((p2 * a2) ** .5, 1e-30),
                        "opposing_primary_fraction": max(0., -dot) / max(p2, 1e-30)}
                records.append({"fold": fold["name"], "stage": stage, "batch": index,
                    "checkpoint_sha256": checksum, "names": list(names), "metrics": metrics})
                del prediction, primary, auxiliary, gp, ga
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for scope in ("all", "encoder", "decoder"):
        values = [r["metrics"][scope] for r in records]
        summary[scope] = {"batches": len(values), "negative_batches": sum(v["dot"] < 0 for v in values),
            "median_cosine": float(np.median([v["cosine"] for v in values])),
            "median_opposing_primary_fraction": float(np.median([v["opposing_primary_fraction"] for v in values]))}
    report = {"formal": False, "gradient_scale": 1024.0 if device.type == "cuda" else 1.0,
        "device": str(device), "conditioning_scope": "complete training DAPI ROIs",
        "selection": "64 seed2044 training-only images per fold; four batch16 groups at each saved stage endpoint",
        "summary": summary, "records": records, "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
