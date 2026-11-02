from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_refiner import (
    CachedOOFDataset,
    CrossFitRefiner,
    predict_refiner_with_tta,
    refiner_feature_specification,
)
from ihc_torch import CompetitionLoss
from refiner_protocol import (
    load_base_protocol,
    load_refiner_protocol,
    sha256_file,
    validate_cache_manifest,
)
from search_postprocessing import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def limited_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    indices = np.linspace(0, len(dataset) - 1, num=limit, dtype=int).tolist()
    return Subset(dataset, indices)


def prepare_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def predict_dataset(
    model: CrossFitRefiner,
    loader: DataLoader,
    device: torch.device,
    tta: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    candidates: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    for features, target, batch_names in loader:
        features = features.to(device, non_blocking=True)
        prediction = predict_refiner_with_tta(model, features, tta)
        candidates.append(prediction[:, 0].cpu().numpy().astype(np.float32) * 255.0)
        baselines.append(
            features[:, model.baseline_channel].cpu().numpy().astype(np.float32)
            * 255.0
        )
        targets.append(np.rint(target[:, 0].numpy() * 255.0).astype(np.uint8))
        names.extend(batch_names)
    return (
        np.concatenate(candidates),
        np.concatenate(baselines),
        np.concatenate(targets),
        names,
    )


def checkpoint_payload(
    model: CrossFitRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    protocol_path: Path,
    cache_root: Path,
    holdout_fold: str,
    training_folds: list[str],
    val_rois: list[str],
    baseline_metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "version": 1,
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "model_spec": model.specification(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "cache_root": str(cache_root),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "holdout_fold": holdout_fold,
        "training_folds": training_folds,
        "val_rois": val_rois,
        "baseline_metrics": baseline_metrics,
        "runtime_git_commit": git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one leakage-safe cross-fitted residual refiner."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol_path = args.protocol.resolve()
    protocol = load_refiner_protocol(protocol_path)
    base_protocol = load_base_protocol(protocol)
    folds = {str(fold["name"]): fold for fold in base_protocol["folds"]}
    if args.fold not in folds:
        raise ValueError(f"Unknown holdout fold: {args.fold}.")
    holdout = folds[args.fold]
    training_folds = [name for name in folds if name != args.fold]
    if len(training_folds) != 4:
        raise RuntimeError("Cross-fit training must use exactly four source folds.")
    cache_root = args.cache_root.resolve()
    validate_cache_manifest(
        cache_root,
        protocol_path,
        protocol,
        base_protocol,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known_outputs = {"last.pt", "best_global_mse.pt", "metrics.json"}
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in known_outputs
    )
    if unexpected:
        raise FileExistsError(f"Unexpected files in output directory: {unexpected}")

    training_config = protocol["training"]
    epochs = args.epochs or int(training_config["epochs"])
    workers = (
        int(training_config["workers"]) if args.workers is None else args.workers
    )
    if epochs <= 0 or workers < 0:
        raise ValueError("epochs must be positive and workers non-negative.")
    fold_index = list(folds).index(args.fold)
    seed = int(training_config["seed"]) + fold_index
    set_seed(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    prepare_cuda_runtime(device)

    recipe_order = list(protocol["recipe_order"])
    weights = dict(protocol["baseline_weights"])
    train_dataset = limited_dataset(
        CachedOOFDataset(
            cache_root,
            training_folds,
            recipe_order,
            weights,
            augment=True,
            preload=True,
        ),
        args.max_train_images,
    )
    val_dataset = limited_dataset(
        CachedOOFDataset(
            cache_root,
            [args.fold],
            recipe_order,
            weights,
            augment=False,
            preload=True,
        ),
        args.max_val_images,
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training_config["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(protocol["inference"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    feature_spec = refiner_feature_specification(recipe_order)
    model_config = protocol["model"]
    model = CrossFitRefiner(
        input_channels=len(feature_spec["channels"]),
        baseline_channel=int(feature_spec["baseline_channel"]),
        width=int(model_config["width"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        residual_scale=float(model_config["residual_scale"]),
    ).to(device)
    loss_function = CompetitionLoss(
        ssim_weight=float(training_config["ssim_weight"]),
        l1_weight=float(training_config["l1_weight"]),
        mse_weight=float(training_config["mse_weight"]),
        ssim_mode=str(training_config["ssim_mode"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    first_features, _, _ = next(iter(val_loader))
    with torch.no_grad():
        initial = model(first_features.to(device)).cpu()
    baseline_channel = first_features[:, model.baseline_channel : model.baseline_channel + 1]
    if not torch.equal(initial, baseline_channel):
        raise RuntimeError("Zero-initialized refiner does not reproduce the baseline.")

    _, baseline, target, val_names = predict_dataset(
        model,
        val_loader,
        device,
        "none",
    )
    jpeg_quality = int(training_config["selection_jpeg_quality"])
    baseline_metrics = {
        "float": evaluate(baseline, target, 0),
        f"jpeg_q{jpeg_quality}": evaluate(baseline, target, jpeg_quality),
    }
    metrics_path = output_dir / "metrics.json"
    metrics: dict[str, object] = {
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "holdout_fold": args.fold,
        "training_folds": training_folds,
        "val_rois": list(holdout["val_rois"]),
        "training_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "feature_spec": feature_spec,
        "model_spec": model.specification(),
        "baseline": baseline_metrics,
        "epochs": [],
    }
    start_epoch = 1
    best_score = -float("inf")
    last_path = output_dir / "last.pt"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("protocol_sha256") != sha256_file(protocol_path)
            or checkpoint.get("holdout_fold") != args.fold
            or checkpoint.get("training_folds") != training_folds
        ):
            raise ValueError("Existing refiner checkpoint does not match this run.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if start_epoch > epochs:
        print(json.dumps({"completed": args.fold, "epochs": epochs}))
        return

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        samples = 0
        for features, batch_target, _ in train_loader:
            features = features.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                prediction = model(features)
                loss = loss_function(prediction, batch_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * features.shape[0]
            samples += features.shape[0]
        scheduler.step()

        candidate, _, epoch_target, epoch_names = predict_dataset(
            model,
            val_loader,
            device,
            str(training_config["selection_tta"]),
        )
        if epoch_names != val_names or not np.array_equal(epoch_target, target):
            raise RuntimeError("Validation cache order changed during training.")
        validation = {
            "float": evaluate(candidate, target, 0),
            f"jpeg_q{jpeg_quality}": evaluate(candidate, target, jpeg_quality),
        }
        score = float(
            validation[f"jpeg_q{jpeg_quality}"][
                "competition_score_global_mse_psnr"
            ]
        )
        best_score = max(best_score, score)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
            "selection_score": score,
            "seconds": time.perf_counter() - epoch_started,
        }
        metrics["epochs"].append(epoch_record)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            protocol_path,
            cache_root,
            args.fold,
            training_folds,
            list(holdout["val_rois"]),
            baseline_metrics,
        )
        atomic_torch_save(last_path, payload)
        if score >= best_score:
            atomic_torch_save(output_dir / "best_global_mse.pt", payload)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        metrics["peak_cuda_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        )
        atomic_json(metrics_path, metrics)
        print(json.dumps(epoch_record), flush=True)


if __name__ == "__main__":
    main()
