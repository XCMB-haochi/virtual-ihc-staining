from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cv_protocol import (
    evenly_spaced_epochs,
    fcmae_encoder_checkpoint,
    initialization_checkpoint,
    load_protocol,
    run_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def selected_protocol_folds(
    protocol: dict[str, object], requested: str | None
) -> list[dict[str, object]]:
    folds = list(protocol["folds"])
    if requested is None:
        return folds
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--folds must contain at least one fold name.")
    if len(names) != len(set(names)):
        raise ValueError("--folds contains duplicate fold names.")
    known = {str(fold["name"]) for fold in folds}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"--folds contains unknown folds: {unknown}.")
    requested_names = set(names)
    return [fold for fold in folds if str(fold["name"]) in requested_names]


def selected_protocol_recipes(
    protocol: dict[str, object], requested: str | None
) -> list[tuple[str, dict[str, object]]]:
    recipes = list(protocol["recipes"].items())
    if requested is None:
        return recipes
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--recipes must contain at least one recipe name.")
    if len(names) != len(set(names)):
        raise ValueError("--recipes contains duplicate recipe names.")
    known = {str(name) for name, _ in recipes}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"--recipes contains unknown recipes: {unknown}.")
    requested_names = set(names)
    return [
        (str(name), recipe)
        for name, recipe in recipes
        if str(name) in requested_names
    ]


def completed_run(output_dir: Path, epochs: int) -> bool:
    checkpoint = output_dir / "best_global_mse.pt"
    metrics_path = output_dir / "metrics.jsonl"
    if not checkpoint.is_file() or not metrics_path.is_file():
        return False
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != epochs:
        return False
    try:
        last_epoch = int(json.loads(lines[-1])["epoch"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return last_epoch == epochs


def build_train_command(
    python_executable: str,
    dataset: Path,
    output_dir: Path,
    val_rois: list[str],
    marker: str,
    recipe: dict[str, object],
    training: dict[str, object],
    stage: int,
    init_checkpoint: Path | None,
    device: str,
    train_rois: list[str] | None = None,
    fold_name: str | None = None,
) -> list[str]:
    learning_rate = training[f"stage{stage}_learning_rate"]
    micro_batch_size = int(recipe.get("batch_size", training["batch_size"]))
    accumulation_steps = int(recipe.get("gradient_accumulation_steps", 1))
    training_markers = recipe.get("training_markers", [marker])
    command = [
        python_executable,
        str(PROJECT_ROOT / "tools" / "train_unet.py"),
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--val-rois",
        ",".join(val_rois),
        "--markers",
        ",".join(str(item) for item in training_markers),
        "--epochs",
        str(recipe.get("epochs_per_stage", training["epochs_per_stage"])),
        "--batch-size",
        str(micro_batch_size),
        "--gradient-accumulation-steps",
        str(accumulation_steps),
        "--model-arch",
        str(recipe.get("model_arch", "unet")),
        "--base-channels",
        str(recipe.get("base_channels", 32)),
        "--block-type",
        str(recipe.get("block_type", "residual")),
        "--condition-mode",
        str(recipe.get("condition_mode", "none")),
        "--context-halo",
        str(recipe["context_halo"]),
        "--train-target-context-halo",
        str(recipe["train_target_context_halo"]),
        "--learning-rate",
        str(learning_rate),
        "--minimum-learning-rate",
        str(training.get("minimum_learning_rate", 0.0)),
        "--gradient-clip-norm",
        str(training.get("gradient_clip_norm", 0.0)),
        "--weight-decay",
        str(training["weight_decay"]),
        "--ssim-weight",
        str(training["ssim_weight"]),
        "--l1-weight",
        str(training["l1_weight"]),
        "--mse-weight",
        str(training["mse_weight"]),
        "--loss-profile",
        str(training.get("loss_profile", "hybrid")),
        "--ssim-mode",
        str(training["ssim_mode"]),
        "--selection-marker",
        marker,
        "--selection-psnr-aggregation",
        str(training["selection_psnr_aggregation"]),
        "--selection-jpeg-quality",
        str(training["selection_jpeg_quality"]),
        "--workers",
        str(training["workers"]),
        "--seed",
        str(training["seed"]),
        "--device",
        device,
    ]
    for field, flag in (
        ("train_target_source_offset", "--train-target-source-offset"),
        ("validation_output_source_offset", "--validation-output-source-offset"),
    ):
        offset = recipe.get(field)
        if offset is not None:
            command.extend((flag, ",".join(str(value) for value in offset)))
    if train_rois is not None:
        command.extend(("--train-rois", ",".join(train_rois)))
    if stage == 1 and recipe.get("dinov2_encoder_init", False):
        command.extend(("--encoder-init", "dinov2"))
    encoder_lr_multiplier = recipe.get("encoder_learning_rate_multiplier")
    if encoder_lr_multiplier is not None:
        command.extend(
            (
                "--encoder-learning-rate",
                str(float(learning_rate) * float(encoder_lr_multiplier)),
            )
        )
    encoder_drop_path_rate = float(recipe.get("encoder_drop_path_rate", 0.0))
    if encoder_drop_path_rate > 0.0:
        command.extend(("--encoder-drop-path-rate", str(encoder_drop_path_rate)))
    cytoland_initialization = recipe.get("cytoland_initialization")
    if stage == 1 and isinstance(cytoland_initialization, dict):
        checkpoint = Path(str(cytoland_initialization["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        command.extend(
            (
                "--cytoland-init-checkpoint",
                str(checkpoint.resolve()),
                "--cytoland-init-mode",
                str(cytoland_initialization["mode"]),
                "--cytoland-init-sha256",
                str(cytoland_initialization["sha256"]),
                "--cytoland-init-scope",
                str(recipe["cytoland_initialization_scope"]),
            )
        )
    validation_evaluations = recipe.get("validation_evaluations")
    if validation_evaluations is not None:
        validation_epochs = evenly_spaced_epochs(
            int(recipe.get("epochs_per_stage", training["epochs_per_stage"])),
            int(validation_evaluations),
        )
        command.extend(
            (
                "--validation-epochs",
                ",".join(str(epoch) for epoch in validation_epochs),
            )
        )
    marker_loss_weights = recipe.get("marker_loss_weights")
    if marker_loss_weights is not None:
        command.extend(
            (
                "--marker-loss-weights",
                ",".join(str(value) for value in marker_loss_weights),
            )
        )
    target_l1_weight_strength = float(recipe.get("target_l1_weight_strength", 0.0))
    if target_l1_weight_strength > 0.0:
        command.extend(
            ("--target-l1-weight-strength", str(target_l1_weight_strength))
        )
    auxiliary_gradient_scope = recipe.get("auxiliary_gradient_scope")
    if recipe.get("primary_gradient_projection", False):
        command.append("--primary-gradient-projection")
    if auxiliary_gradient_scope is not None:
        command.extend(
            (
                "--auxiliary-gradient-scope",
                str(auxiliary_gradient_scope),
            )
        )
    d4_consistency_weight = float(recipe.get("d4_consistency_weight", 0.0))
    if d4_consistency_weight > 0.0:
        command.extend(("--d4-consistency-weight", str(d4_consistency_weight)))
    pyramid_weight = float(
        recipe.get("pyramid_weight", training.get("pyramid_weight", 0.0))
    )
    if pyramid_weight > 0.0:
        command.extend(("--pyramid-weight", str(pyramid_weight)))
    robust_objective = str(recipe.get("robust_objective", "none"))
    if robust_objective != "none":
        command.extend(("--robust-objective", robust_objective))
        command.extend(("--group-dro-eta", str(recipe.get("group_dro_eta", 5.0))))
    feature_guidance = recipe.get("feature_guidance")
    if feature_guidance is not None:
        if not isinstance(feature_guidance, dict) or fold_name is None:
            raise ValueError(
                "Feature guidance requires a configured object and fold name."
            )
        teachers = feature_guidance["teachers"]
        teacher = teachers[fold_name]
        checkpoint = Path(str(teacher["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        command.extend(
            (
                "--feature-guidance-checkpoint",
                str(checkpoint.resolve()),
                "--feature-guidance-checkpoint-sha256",
                str(teacher["sha256"]),
                "--feature-guidance-marker",
                str(feature_guidance["marker"]),
                "--feature-guidance-weight",
                str(feature_guidance["weight"]),
                "--feature-guidance-stage",
                str(feature_guidance["stage"]),
                "--feature-guidance-crop-size",
                str(feature_guidance["crop_size"]),
                "--feature-guidance-batch-samples",
                str(feature_guidance["batch_samples"]),
            )
        )
        if "mode" in feature_guidance:
            command.extend(("--feature-guidance-mode", str(feature_guidance["mode"])))
    if training.get("save_every_epoch", True):
        command.append("--save-every-epoch")
    if stage == 2:
        if init_checkpoint is None:
            raise ValueError("Stage 2 requires a stage-1 checkpoint.")
        command.extend(("--init-checkpoint", str(init_checkpoint)))
    elif init_checkpoint is not None:
        raise ValueError("Stage 1 must not receive an initialization checkpoint.")
    if recipe.get("init_add_conditioning", False):
        if init_checkpoint is None:
            raise ValueError("Add-conditioning initialization requires a checkpoint.")
        command.append("--init-add-conditioning")
    ema_decay = float(recipe.get("ema_decay", 0.0))
    if ema_decay > 0:
        command.extend(("--ema-decay", str(ema_decay)))
    mixstyle_probability = float(recipe.get("mixstyle_probability", 0.0))
    if mixstyle_probability > 0:
        command.extend(
            (
                "--mixstyle-probability",
                str(mixstyle_probability),
                "--mixstyle-alpha",
                str(recipe.get("mixstyle_alpha", 0.1)),
            )
        )
    if recipe.get("source_photometric_augmentation", False):
        command.append("--source-photometric-augmentation")
    activity_sampler = recipe.get("activity_sampler")
    if activity_sampler is not None:
        if not isinstance(activity_sampler, dict):
            raise ValueError("Recipe activity_sampler must be an object.")
        command.extend(
            (
                "--activity-sampler",
                str(activity_sampler["mode"]),
                "--activity-sampler-start-ratio",
                str(activity_sampler["start_ratio"]),
                "--activity-sampler-hard-fraction",
                str(activity_sampler["hard_fraction"]),
                "--activity-sampler-bins",
                str(activity_sampler["num_bins"]),
            )
        )
    paired_affine = recipe.get("paired_affine_augmentation")
    if paired_affine is not None:
        command.extend(
            (
                "--paired-affine-probability",
                str(paired_affine["probability"]),
                "--paired-affine-outer-halo",
                str(paired_affine["outer_halo"]),
                "--paired-affine-degrees",
                str(paired_affine["degrees"]),
                "--paired-affine-scale-min",
                str(paired_affine["scale_min"]),
                "--paired-affine-scale-max",
                str(paired_affine["scale_max"]),
            )
        )
    return command


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all stages in a validated ROI cross-validation protocol."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--folds",
        help="Optional comma-separated fold subset; protocol order is preserved.",
    )
    parser.add_argument(
        "--recipes",
        help="Optional comma-separated recipe subset; protocol order is preserved.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    folds = selected_protocol_folds(protocol, args.folds)
    recipes = selected_protocol_recipes(protocol, args.recipes)
    output_root = args.output_root.resolve()
    status_name = str(protocol["name"])
    status_parts: list[str] = []
    if args.folds is not None:
        status_parts.append("-".join(str(fold["name"]) for fold in folds))
    if args.recipes is not None:
        status_parts.append("-".join(name for name, _ in recipes))
    if status_parts:
        status_name = f"{status_name}_{'_'.join(status_parts)}"
    status_path = output_root / f"{status_name}_status.json"
    completed: list[str] = []
    planned: list[dict[str, object]] = []
    total_runs = len(folds) * sum(
        len(recipe.get("train_stages", [1, 2]))
        for _, recipe in recipes
    )

    for fold in folds:
        for recipe_name, recipe in recipes:
            recipe_epochs = int(
                recipe.get(
                    "epochs_per_stage",
                    protocol["training"]["epochs_per_stage"],
                )
            )
            train_rois_by_fold = recipe.get("train_rois_by_fold")
            train_rois = (
                list(train_rois_by_fold[str(fold["name"])])
                if isinstance(train_rois_by_fold, dict)
                else None
            )
            for stage in recipe.get("train_stages", [1, 2]):
                output_dir = run_directory(
                    output_root,
                    protocol["name"],
                    fold["name"],
                    recipe_name,
                    stage,
                )
                init_checkpoint = initialization_checkpoint(
                    output_root,
                    protocol,
                    str(fold["name"]),
                    recipe_name,
                    stage,
                )
                command = build_train_command(
                    sys.executable,
                    args.dataset.resolve(),
                    output_dir,
                    list(fold["val_rois"]),
                    str(protocol["marker"]),
                    recipe,
                    protocol["training"],
                    stage,
                    init_checkpoint,
                    args.device,
                    train_rois,
                    str(fold["name"]),
                )
                encoder_checkpoint = None
                if stage == 1 and recipe.get("fcmae_encoder_init", False):
                    encoder_checkpoint = fcmae_encoder_checkpoint(
                        output_root,
                        protocol,
                        str(fold["name"]),
                    )
                    command.extend(
                        (
                            "--fcmae-encoder-init-checkpoint",
                            str(encoder_checkpoint),
                        )
                    )
                run_id = f"{fold['name']}/{recipe_name}/stage{stage}"
                planned.append({"run_id": run_id, "command": command})
                if args.dry_run:
                    print(json.dumps(planned[-1], ensure_ascii=False), flush=True)
                    continue
                if init_checkpoint is not None and not init_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Missing initialization checkpoint for {run_id}: "
                        f"{init_checkpoint}"
                    )
                if encoder_checkpoint is not None and not encoder_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Missing fold-local FCMAE encoder for {run_id}: "
                        f"{encoder_checkpoint}"
                    )
                if completed_run(output_dir, recipe_epochs):
                    completed.append(run_id)
                    print(f"SKIP completed {run_id}: {output_dir}", flush=True)
                    continue
                if output_dir.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite incomplete run {run_id}: {output_dir}"
                    )
                print(json.dumps(planned[-1], ensure_ascii=False), flush=True)
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                if not completed_run(output_dir, recipe_epochs):
                    raise RuntimeError(f"Run did not produce a complete result: {run_id}")
                completed.append(run_id)
                write_status(
                    status_path,
                    {
                        "protocol": protocol["name"],
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "completed": completed,
                        "total_runs": total_runs,
                    },
                )

    if not args.dry_run:
        write_status(
            status_path,
            {
                "protocol": protocol["name"],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed": completed,
                "total_runs": total_runs,
                "complete": True,
            },
        )
        print(f"COMPLETE {protocol['name']}: {len(completed)} stages", flush=True)


if __name__ == "__main__":
    main()
