from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cv_protocol import (
    evenly_spaced_epochs,
    inference_checkpoint_filename,
    load_protocol,
    sourced_run_directory,
)
from ihc_data import image_names, parse_patch_name, resolve_data_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_updates_per_stage(
    train_images: int,
    micro_batch_size: int,
    accumulation_steps: int,
    epochs: int,
) -> int:
    micro_batches = math.ceil(train_images / micro_batch_size)
    return math.ceil(micro_batches / accumulation_steps) * epochs


def aggregate_records(
    records: dict[str, dict[str, object]],
    metadata: dict[str, dict[str, object]],
    baseline_recipe: str,
) -> dict[str, object]:
    baseline_q95 = float(records[baseline_recipe]["jpeg_q95"]["competition_score"])
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    draws: dict[str, dict[int, str]] = defaultdict(dict)
    for recipe_name, record in records.items():
        recipe_metadata = metadata[recipe_name]
        roi_count = int(recipe_metadata["roi_count"])
        grouped[roi_count].append(record)
        draw = str(recipe_metadata["draw"])
        if draw != "baseline":
            draws[draw][roi_count] = recipe_name

    scale_summary: dict[str, object] = {}
    for roi_count, items in sorted(grouped.items()):
        q95_scores = [float(item["jpeg_q95"]["competition_score"]) for item in items]
        float_scores = [float(item["float"]["competition_score"]) for item in items]
        train_images = [int(item["train_images"]) for item in items]
        scale_summary[str(roi_count)] = {
            "recipes": [str(item["recipe"]) for item in items],
            "count": len(items),
            "train_images_mean": statistics.fmean(train_images),
            "jpeg_q95_mean": statistics.fmean(q95_scores),
            "jpeg_q95_pstdev": statistics.pstdev(q95_scores),
            "jpeg_q95_min": min(q95_scores),
            "jpeg_q95_max": max(q95_scores),
            "gap_to_full20": baseline_q95 - statistics.fmean(q95_scores),
            "float_mean": statistics.fmean(float_scores),
            "float_pstdev": statistics.pstdev(float_scores),
        }

    draw_curves: dict[str, object] = {}
    for draw, scale_recipes in sorted(draws.items()):
        required_scales = {5, 10, 15}
        if set(scale_recipes) != required_scales:
            raise ValueError(
                f"Draw {draw} must define scales {sorted(required_scales)}, "
                f"got {sorted(scale_recipes)}."
            )
        ordered_scores = [
            float(records[scale_recipes[scale]]["jpeg_q95"]["competition_score"])
            for scale in (5, 10, 15)
        ] + [baseline_q95]
        draw_curves[draw] = {
            "recipes": {
                str(scale): scale_recipes[scale] for scale in sorted(scale_recipes)
            },
            "jpeg_q95_scores": {
                str(scale): score
                for scale, score in zip((5, 10, 15, 20), ordered_scores)
            },
            "strictly_monotonic": all(
                lower < upper
                for lower, upper in zip(ordered_scores, ordered_scores[1:])
            ),
        }

    return {
        "baseline_recipe": baseline_recipe,
        "baseline_jpeg_q95": baseline_q95,
        "scale_summary": scale_summary,
        "draw_curves": draw_curves,
        "monotonic_draws": sorted(
            draw for draw, values in draw_curves.items() if values["strictly_monotonic"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a preregistered labelled-data scale study."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--fold")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    study = protocol.get("label_scale_study")
    if not isinstance(study, dict) or study.get("version") != 1:
        raise ValueError("Protocol does not contain a version-1 label_scale_study.")
    fold_name = args.fold or str(study["initial_fold"])
    fold_by_name = {str(fold["name"]): fold for fold in protocol["folds"]}
    if fold_name not in fold_by_name:
        raise ValueError(f"Unknown study fold: {fold_name}.")
    fold = fold_by_name[fold_name]
    val_rois = set(fold["val_rois"])

    metadata = study.get("recipes")
    if not isinstance(metadata, dict) or set(metadata) != set(protocol["recipes"]):
        raise ValueError("label_scale_study.recipes must describe every recipe.")
    baseline_recipe = str(study["baseline_recipe"])
    if baseline_recipe not in metadata:
        raise ValueError("The label-scale baseline recipe is not configured.")
    selection_evaluations = int(study["selection_evaluations_per_stage"])

    data_root = resolve_data_root(args.dataset)
    names = image_names(data_root / "train" / "DAPI")
    available_rois = {parse_patch_name(name).roi for name in names}
    validation_images = sum(parse_patch_name(name).roi in val_rois for name in names)
    checkpoint_filename = inference_checkpoint_filename(protocol)
    records: dict[str, dict[str, object]] = {}

    for recipe_name, recipe_metadata in metadata.items():
        recipe = protocol["recipes"][recipe_name]
        train_rois_by_fold = recipe.get("train_rois_by_fold")
        expected_train_rois = (
            set(train_rois_by_fold[fold_name])
            if isinstance(train_rois_by_fold, dict)
            else available_rois - val_rois
        )
        if len(expected_train_rois) != int(recipe_metadata["roi_count"]):
            raise ValueError(f"ROI-count metadata mismatch for {recipe_name}.")
        expected_train_images = sum(
            parse_patch_name(name).roi in expected_train_rois for name in names
        )
        registered_train_images = int(
            recipe_metadata["expected_train_images"][fold_name]
        )
        if expected_train_images != registered_train_images:
            raise ValueError(
                f"Training-image count mismatch for {recipe_name}: "
                f"expected={registered_train_images}, actual={expected_train_images}."
            )

        run_dir = sourced_run_directory(
            args.output_root.resolve(),
            protocol,
            fold_name,
            recipe_name,
        )
        checkpoint = run_dir / checkpoint_filename
        config_path = run_dir / "config.json"
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Incomplete study run for {recipe_name}: {run_dir}"
            )
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(run_config["train_images"]) != expected_train_images:
            raise ValueError(f"Run train_images mismatch for {recipe_name}.")
        resolved_train_rois = run_config.get("train_rois_resolved")
        if (
            resolved_train_rois is not None
            and set(resolved_train_rois) != expected_train_rois
        ):
            raise ValueError(f"Run training-ROI mismatch for {recipe_name}.")
        expected_epochs = int(
            recipe.get("epochs_per_stage", protocol["training"]["epochs_per_stage"])
        )
        if int(run_config["epochs"]) != expected_epochs:
            raise ValueError(f"Run epoch budget mismatch for {recipe_name}.")
        expected_validation_epochs = evenly_spaced_epochs(
            expected_epochs,
            selection_evaluations,
        )
        resolved_validation_epochs = run_config.get("validation_epochs_resolved")
        if resolved_validation_epochs is not None:
            if list(resolved_validation_epochs) != expected_validation_epochs:
                raise ValueError(
                    f"Run validation schedule mismatch for {recipe_name}."
                )
        elif expected_validation_epochs != list(range(1, expected_epochs + 1)):
            raise ValueError(
                f"Run lacks the registered validation schedule for {recipe_name}."
            )
        actual_updates = optimizer_updates_per_stage(
            expected_train_images,
            int(run_config["batch_size"]),
            int(run_config.get("gradient_accumulation_steps", 1)),
            expected_epochs,
        )
        registered_updates = int(
            recipe_metadata["expected_optimizer_updates_per_stage"][fold_name]
        )
        if actual_updates != registered_updates:
            raise ValueError(f"Optimizer-update metadata mismatch for {recipe_name}.")

        evaluation_path = (
            args.evaluation_root.resolve()
            / f"{protocol['name']}_{fold_name}_{recipe_name}_eval.json"
        )
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Missing evaluation for {recipe_name}.")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if Path(evaluation["checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError(f"Evaluation checkpoint mismatch for {recipe_name}.")
        if set(evaluation["validation_rois"]) != val_rois:
            raise ValueError(f"Evaluation ROI mismatch for {recipe_name}.")
        if int(evaluation["validation_images"]) != validation_images:
            raise ValueError(f"Evaluation image-count mismatch for {recipe_name}.")
        marker_metrics = evaluation["markers"][protocol["marker"]]
        records[recipe_name] = {
            "recipe": recipe_name,
            "draw": recipe_metadata["draw"],
            "roi_count": int(recipe_metadata["roi_count"]),
            "train_rois": sorted(expected_train_rois),
            "train_images": expected_train_images,
            "epochs_per_stage": expected_epochs,
            "validation_epochs": expected_validation_epochs,
            "optimizer_updates_per_stage": actual_updates,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluation": str(evaluation_path),
            "float": marker_metrics["float"],
            "jpeg_q95": marker_metrics["jpeg_q95"],
            "roi_metrics": evaluation["roi_metrics"][protocol["marker"]],
        }

    result = {
        "version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["name"],
        "fold": fold_name,
        "validation_rois": list(fold["val_rois"]),
        "validation_images": validation_images,
        "marker": protocol["marker"],
        "metric_variant": "jpeg_q95",
        "records": records,
        "summary": aggregate_records(records, metadata, baseline_recipe),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
