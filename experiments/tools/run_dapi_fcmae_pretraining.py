from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cv_protocol import fcmae_pretraining_directory, load_protocol
from run_cv_training import selected_protocol_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def completed_pretraining(output_dir: Path, expected_epoch: int) -> bool:
    complete_path = output_dir / "complete.json"
    final_path = output_dir / "final_encoder.pt"
    if not complete_path.is_file() or not final_path.is_file():
        return False
    try:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("complete")) and int(payload.get("epoch", -1)) == (
        expected_epoch
    )


def build_command(
    dataset: Path,
    output_dir: Path,
    val_rois: list[str],
    pretraining: dict[str, object],
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "train_dapi_fcmae.py"),
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--val-rois",
        ",".join(val_rois),
        "--data-scope",
        str(pretraining.get("data_scope", "fold-train")),
        "--pretraining-markers",
        ",".join(pretraining.get("pretraining_markers", ["DAPI"])),
        "--epochs",
        str(pretraining["epochs"]),
        "--batch-size",
        str(pretraining["batch_size"]),
        "--gradient-accumulation-steps",
        str(pretraining.get("gradient_accumulation_steps", 1)),
        "--model-size",
        str(pretraining.get("model_size", "tiny")),
        "--workers",
        str(pretraining["workers"]),
        "--learning-rate",
        str(pretraining["learning_rate"]),
        "--minimum-learning-rate",
        str(pretraining["minimum_learning_rate"]),
        "--warmup-epochs",
        str(pretraining["warmup_epochs"]),
        "--weight-decay",
        str(pretraining["weight_decay"]),
        "--mask-ratio",
        str(pretraining["mask_ratio"]),
        "--checkpoint-interval",
        str(pretraining["checkpoint_interval"]),
        "--seed",
        str(pretraining["seed"]),
        "--device",
        device,
    ]
    initialization = pretraining.get("encoder_initialization")
    if isinstance(initialization, dict):
        checkpoint = Path(str(initialization["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        command.extend(
            (
                "--encoder-init-checkpoint",
                str(checkpoint.resolve()),
                "--encoder-init-mode",
                str(initialization["mode"]),
                "--encoder-init-sha256",
                str(initialization["sha256"]),
            )
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fold-local DAPI FCMAE pretraining from a CV protocol."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    pretraining = protocol.get("fcmae_pretraining")
    if not isinstance(pretraining, dict):
        raise ValueError("Protocol has no fold-local FCMAE pretraining section.")
    folds = selected_protocol_folds(protocol, args.folds)
    output_root = args.output_root.resolve()
    expected_epoch = int(pretraining["epochs"])
    completed: list[str] = []
    shared = bool(pretraining.get("shared_across_folds", False))
    planned_folds = folds[:1] if shared else folds
    for fold in planned_folds:
        fold_name = str(fold["name"])
        output_dir = fcmae_pretraining_directory(
            output_root,
            protocol,
            fold_name,
        )
        command = build_command(
            args.dataset.resolve(),
            output_dir,
            [] if shared else list(fold["val_rois"]),
            pretraining,
            args.device,
        )
        job_name = "shared" if shared else fold_name
        payload = {"fold": job_name, "command": command}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        if completed_pretraining(output_dir, expected_epoch):
            completed.append(job_name)
            print(f"SKIP completed FCMAE {job_name}: {output_dir}", flush=True)
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not completed_pretraining(output_dir, expected_epoch):
            raise RuntimeError(
                f"FCMAE did not produce a complete artifact for {fold_name}."
            )
        completed.append(job_name)
    if not args.dry_run:
        print(
            f"COMPLETE FCMAE jobs: {len(completed)}/{len(planned_folds)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
