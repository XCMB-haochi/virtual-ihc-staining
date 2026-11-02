from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import MARKERS, image_names, load_gray, parse_patch_name, resolve_data_root
from ihc_unext2 import (
    CYTOLAND_VSCYTO2D_TINY_MODE,
    DapiFCMAE,
    FCMAE_MASK_RATIO,
    OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE,
    OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE,
    OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE,
    initialize_fcmae_encoder_from_public_checkpoint,
    unext2_encoder_specification,
)


PRETRAINING_MARKERS = ("DAPI", *MARKERS)


def initialize_packaged_public_encoder(encoder, path, expected_sha256, model_size):
    """Reuse the exact epoch0 baseline bytes, including its saved input stem."""
    path = Path(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != expected_sha256:
        raise ValueError('Packaged encoder hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('task') != 'dapi-fcmae-pretraining' or payload.get('epoch') != 0
            or payload.get('val_rois') != []
            or payload.get('config', {}).get('data_scope') != 'public-only'
            or payload.get('encoder_spec') != unext2_encoder_specification(model_size)):
        raise ValueError('Expected an epoch0 public-only encoder, not domain-trained weights')
    encoder.load_state_dict(payload['encoder'], strict=True)
    return {'mode': 'packaged-public-encoder', 'checkpoint': str(path.resolve()),
            'checkpoint_sha256': digest, 'source_initialization': payload['config'].get('encoder_initialization')}


class MarkerPatchDataset(Dataset):
    def __init__(
        self,
        train_directory: Path,
        samples: list[tuple[str, str]],
        augment: bool = True,
    ) -> None:
        self.train_directory = train_directory
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        marker, name = self.samples[index]
        image = load_gray(self.train_directory / marker / name)[None, ...]
        if self.augment:
            transform_index = random.randrange(8)
            image = np.rot90(image, transform_index % 4, axes=(-2, -1))
            if transform_index >= 4:
                image = image[..., ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(image) / 255.0).float()
        return tensor, f"{marker}/{name}"


def parse_pretraining_markers(value: str) -> list[str]:
    markers = [marker.strip() for marker in value.split(",") if marker.strip()]
    if not markers:
        raise ValueError("At least one FCMAE pretraining marker is required.")
    if len(set(markers)) != len(markers):
        raise ValueError("FCMAE pretraining markers must be unique.")
    unsupported = sorted(set(markers) - set(PRETRAINING_MARKERS))
    if unsupported:
        raise ValueError(f"Unsupported FCMAE pretraining markers: {unsupported}.")
    if "DAPI" not in markers:
        raise ValueError("FCMAE pretraining markers must include DAPI.")
    return markers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_rng_states_for_restore(value: object) -> list[torch.Tensor]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("CUDA RNG state must be a non-empty tensor sequence.")
    states = []
    for state in value:
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
        ):
            raise ValueError("CUDA RNG states must be one-dimensional ByteTensors.")
        states.append(state.detach().cpu())
    return states


def learning_rate_for_epoch(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch must be between 1 and epochs.")
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be non-negative and below epochs.")
    if warmup_epochs and epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    progress = (epoch - warmup_epochs - 1) / max(epochs - warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(
        minimum_learning_rate
        + (base_learning_rate - minimum_learning_rate) * cosine
    )


def gradient_accumulation_sample_scale(
    batch_index: int,
    micro_batch_size: int,
    current_batch_size: int,
    accumulation_steps: int,
    dataset_size: int,
) -> float:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    group_first_sample = group_start * micro_batch_size
    group_sample_count = min(
        accumulation_steps * micro_batch_size,
        dataset_size - group_first_sample,
    )
    if group_sample_count <= 0 or current_batch_size > group_sample_count:
        raise ValueError("FCMAE gradient accumulation batch index exceeds the dataset.")
    return current_batch_size / group_sample_count


def runtime_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def truncate_metrics(path: Path, maximum_epoch: int) -> None:
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if int(payload["epoch"]) <= maximum_epoch:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def build_epoch_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=generator,
    )


def select_pretraining_names(
    all_names: list[str],
    val_rois: set[str],
    data_scope: str,
) -> list[str]:
    if data_scope == "fold-train":
        return [
            name
            for name in all_names
            if parse_patch_name(name).roi not in val_rois
        ]
    if data_scope == "all-train":
        return list(all_names)
    raise ValueError(f"Unsupported DAPI pretraining data scope: {data_scope}.")


def select_pretraining_samples(
    train_directory: Path,
    markers: list[str],
    val_rois: set[str],
    data_scope: str,
) -> list[tuple[str, str]]:
    names_by_marker = {
        marker: image_names(train_directory / marker) for marker in markers
    }
    reference_marker = markers[0]
    reference_names = names_by_marker[reference_marker]
    for marker, names in names_by_marker.items():
        if names != reference_names:
            missing = sorted(set(reference_names) - set(names))[:5]
            extra = sorted(set(names) - set(reference_names))[:5]
            raise ValueError(
                "FCMAE marker filenames must match exactly: "
                f"{marker} versus {reference_marker}; missing={missing}, extra={extra}."
            )
    selected_names = select_pretraining_names(reference_names, val_rois, data_scope)
    return [(marker, name) for marker in markers for name in selected_names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAPI masked reconstruction pretraining for UNeXt2."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-rois", required=True)
    parser.add_argument(
        "--data-scope",
        choices=("fold-train", "all-train"),
        default="fold-train",
    )
    parser.add_argument("--pretraining-markers", default="DAPI")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=FCMAE_MASK_RATIO)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--encoder-init-checkpoint", type=Path)
    parser.add_argument(
        "--encoder-init-mode",
        choices=(
            'packaged-public-encoder',
            CYTOLAND_VSCYTO2D_TINY_MODE,
            OFFICIAL_CONVNEXTV2_FCMAE_TINY_MODE,
            OFFICIAL_CONVNEXTV2_FCMAE_BASE_MODE,
            OFFICIAL_CONVNEXTV2_SUPERVISED_22K_TINY_MODE,
        ),
    )
    parser.add_argument("--encoder-init-sha256")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("--mask-ratio must be strictly between 0 and 1.")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must be non-negative and below epochs.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")
    if not 0 <= args.minimum_learning_rate < args.learning_rate:
        raise ValueError("minimum learning rate must be below the learning rate.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    initialization_arguments = (
        args.encoder_init_checkpoint,
        args.encoder_init_mode,
        args.encoder_init_sha256,
    )
    if any(value is not None for value in initialization_arguments) and not all(
        value is not None for value in initialization_arguments
    ):
        raise ValueError(
            "Encoder initialization requires checkpoint, mode, and SHA-256."
        )

    data_root = resolve_data_root(args.dataset)
    pretraining_markers = parse_pretraining_markers(args.pretraining_markers)
    val_rois = {
        value.strip().upper() for value in args.val_rois.split(",") if value.strip()
    }
    train_samples = select_pretraining_samples(
        data_root / "train",
        pretraining_markers,
        val_rois,
        args.data_scope,
    )
    excluded_val_rois = val_rois if args.data_scope == "fold-train" else set()
    if args.max_train_images > 0 and len(train_samples) > args.max_train_images:
        indices = np.linspace(
            0,
            len(train_samples) - 1,
            num=args.max_train_images,
            dtype=int,
        )
        train_samples = [train_samples[index] for index in indices]
    if not train_samples:
        raise ValueError("Fold-local FCMAE split contains no training images.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir = args.output_dir.resolve()
    latest_path = args.output_dir / "latest.pt"
    final_path = args.output_dir / "final_encoder.pt"
    metrics_path = args.output_dir / "metrics.jsonl"
    config_path = args.output_dir / "config.json"
    complete_path = args.output_dir / "complete.json"

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"FCMAE output directory is non-empty: {args.output_dir}; use --resume."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = DapiFCMAE(model_size=args.model_size)
    encoder_initialization = None
    if args.encoder_init_checkpoint is not None:
        args.encoder_init_checkpoint = args.encoder_init_checkpoint.resolve()
        if args.encoder_init_mode == 'packaged-public-encoder':
            encoder_initialization = initialize_packaged_public_encoder(
                model.encoder, args.encoder_init_checkpoint,
                str(args.encoder_init_sha256), args.model_size)
        else:
            encoder_initialization = initialize_fcmae_encoder_from_public_checkpoint(
                model.encoder, args.encoder_init_checkpoint,
                str(args.encoder_init_sha256), str(args.encoder_init_mode))
        if encoder_initialization["mode"] != args.encoder_init_mode:
            raise ValueError("Encoder initialization mode does not match checkpoint.")
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    if args.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                cuda_rng_states_for_restore(checkpoint["cuda_rng_state"])
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        truncate_metrics(metrics_path, start_epoch - 1)
    elif args.resume and any(args.output_dir.iterdir()):
        raise FileNotFoundError(
            f"Cannot resume FCMAE without {latest_path}."
        )

    config = {
        "version": 1,
        "task": "dapi-fcmae-pretraining",
        "dataset": str(data_root),
        "output_dir": str(args.output_dir),
        "data_scope": args.data_scope,
        "pretraining_markers": pretraining_markers,
        "val_rois_excluded": sorted(excluded_val_rois),
        "train_rois": sorted(
            {parse_patch_name(name).roi for _, name in train_samples}
        ),
        "train_images": len(train_samples),
        "train_images_by_marker": {
            marker: sum(
                sample_marker == marker for sample_marker, _ in train_samples
            )
            for marker in pretraining_markers
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "model_size": args.model_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "device_resolved": str(device),
        "amp": use_amp,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_spec": unext2_encoder_specification(args.model_size),
        "runtime": runtime_metadata(),
    }
    if encoder_initialization is not None:
        config["encoder_initialization"] = encoder_initialization
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        previous.setdefault("gradient_accumulation_steps", 1)
        previous.setdefault("effective_batch_size", int(previous["batch_size"]))
        previous.setdefault("model_size", "tiny")
        previous.setdefault("data_scope", "fold-train")
        previous.setdefault("pretraining_markers", ["DAPI"])
        previous.setdefault(
            "train_images_by_marker",
            {"DAPI": int(previous["train_images"])},
        )
        comparable_keys = set(config) - {"runtime"}
        if any(previous.get(key) != config[key] for key in comparable_keys):
            raise ValueError("FCMAE resume configuration differs from the existing run.")
    else:
        write_json(config_path, config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    dataset = MarkerPatchDataset(data_root / "train", train_samples)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        learning_rate = learning_rate_for_epoch(
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.learning_rate,
            args.minimum_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loader = build_epoch_loader(
            dataset,
            args.batch_size,
            args.workers,
            args.seed,
            epoch,
            pin_memory=device.type == "cuda",
        )
        model.train()
        loss_sum = 0.0
        sample_count = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (source, _) in enumerate(loader):
            source = source.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                reconstruction, mask = model(source, mask_ratio=args.mask_ratio)
                squared_error = (reconstruction - source).square()
                loss = (squared_error * mask).sum() / (
                    mask.sum() * source.shape[1]
                ).clamp_min(1)
            scale = gradient_accumulation_sample_scale(
                batch_index,
                args.batch_size,
                int(source.shape[0]),
                args.gradient_accumulation_steps,
                len(dataset),
            )
            scaler.scale(loss * scale).backward()
            update_due = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if update_due:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            loss_sum += float(loss.detach()) * source.shape[0]
            sample_count += source.shape[0]

        metrics = {
            "epoch": epoch,
            "loss": loss_sum / sample_count,
            "learning_rate": learning_rate,
            "seconds": time.perf_counter() - started,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "optimizer_steps": optimizer_steps,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            atomic_torch_save(
                {
                    "version": 1,
                    "task": "dapi-fcmae-resume",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "encoder_initialization": encoder_initialization,
                },
                latest_path,
            )

    atomic_torch_save(
        {
            "version": 1,
            "task": "dapi-fcmae-pretraining",
            "epoch": args.epochs,
            "val_rois": sorted(excluded_val_rois),
            "encoder_spec": unext2_encoder_specification(args.model_size),
            "encoder": model.encoder.state_dict(),
            "config": config,
        },
        final_path,
    )
    write_json(
        complete_path,
        {
            "version": 1,
            "complete": True,
            "epoch": args.epochs,
            "final_encoder": str(final_path),
        },
    )
    print(f"COMPLETE DAPI FCMAE: {final_path}", flush=True)


if __name__ == "__main__":
    main()
