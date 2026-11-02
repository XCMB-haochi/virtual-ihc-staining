from __future__ import annotations

import json
import math
import re
from pathlib import Path


SAFE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ROI_RE = re.compile(r"^ROI\d{3}$")
MODEL_ARCHITECTURES = {
    "unet",
    "nafnet-width32",
    "convnext-tiny-unet",
    "dinov2-small-unet",
    "unext2-fcmae-tiny-dinov2-residual",
    "unext2-fcmae-tiny-dinov2-fusion",
    "unext2-fcmae-tiny",
    "unext2-fcmae-base",
    "unext2-fcmae-tiny-widehead",
    "unext2-fcmae-tiny-marker-decoders",
    "unext2-fcmae-tiny-cross-marker-feedback",
    "unext2-fcmae-tiny-dapi-detail-residual",
    "unext2-fcmae-tiny-shallow-dapi-skip",
    "dualbranch-swin-s",
    "restormer-small",
}
BLOCK_TYPES = {"residual", "dense"}
MAX_CONTEXT_HALO = 256
INFERENCE_CHECKPOINT_FILENAMES = {
    "best_global_mse.pt",
    "best_mean_image.pt",
}
OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION = {
    "mode": "official-convnextv2-fcmae-imagenet1k-tiny",
    "checkpoint": (
        "runs/public_checkpoints/convnextv2_tiny_1k_224_fcmae.pt"
    ),
    "sha256": (
        "b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef"
    ),
    "source_url": (
        "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/"
        "convnextv2_tiny_1k_224_fcmae.pt"
    ),
    "license": "CC-BY-NC-4.0",
}
OFFICIAL_CONVNEXTV2_FCMAE_BASE_INITIALIZATION = {
    "mode": "official-convnextv2-fcmae-imagenet1k-base",
    "checkpoint": (
        "runs/public_checkpoints/convnextv2_base_1k_224_fcmae.pt"
    ),
    "sha256": (
        "cd59ca7d4db45a8d7bc2128dc447c0a5aee209d2b4920e79e1da4196afca6405"
    ),
    "source_url": (
        "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/"
        "convnextv2_base_1k_224_fcmae.pt"
    ),
    "license": "CC-BY-NC-4.0",
}
OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION = {
    "mode": "official-convnextv2-supervised-imagenet22k-to-1k-tiny",
    "checkpoint": (
        "runs/public_checkpoints/convnextv2_tiny_22k_224_ema.pt"
    ),
    "sha256": (
        "853d431aa9363f1b058e3c343d4bf2fca5fe2a4196621c381ddbcd4828290a96"
    ),
    "source_url": (
        "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/"
        "convnextv2_tiny_22k_224_ema.pt"
    ),
    "license": "CC-BY-NC-4.0",
}
CYTOLAND_VSCYTO2D_INITIALIZATION = {
    "mode": "biohub-cytoland-vscyto2d-unext2-tiny",
    "checkpoint": (
        "runs/public_checkpoints/cytoland_vscyto2d/epoch-399-step-23200.ckpt"
    ),
    "sha256": (
        "1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe"
    ),
    "source_url": (
        "https://huggingface.co/biohub/VSCyto2D/resolve/main/"
        "epoch%3D399-step%3D23200.ckpt"
    ),
    "license": "BSD-3-Clause",
}
AUDITED_PUBLIC_ENCODER_INITIALIZATIONS = {
    initialization["mode"]: initialization
    for initialization in (
        OFFICIAL_CONVNEXTV2_FCMAE_INITIALIZATION,
        OFFICIAL_CONVNEXTV2_FCMAE_BASE_INITIALIZATION,
        OFFICIAL_CONVNEXTV2_SUPERVISED_22K_INITIALIZATION,
        CYTOLAND_VSCYTO2D_INITIALIZATION,
    )
}


def evenly_spaced_epochs(total_epochs: int, evaluation_count: int) -> list[int]:
    if type(total_epochs) is not int or total_epochs <= 0:
        raise ValueError("total_epochs must be a positive integer.")
    if (
        type(evaluation_count) is not int
        or evaluation_count <= 0
        or evaluation_count > total_epochs
    ):
        raise ValueError(
            "evaluation_count must be a positive integer no greater than "
            "total_epochs."
        )
    return [
        math.ceil(index * total_epochs / evaluation_count)
        for index in range(1, evaluation_count + 1)
    ]


def _positive_number(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not parsed > 0:
        raise ValueError(f"{field} must be positive, got {parsed}.")
    return parsed


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {parsed}.")
    return parsed


def _nonnegative_number(value: object, field: str) -> float:
    parsed = _finite_number(value, field)
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative, got {parsed}.")
    return parsed


def _validate_gates(
    protocol: dict[str, object],
    candidates: dict[str, object],
    fold_count: int,
    roi_count: int,
) -> None:
    gates = protocol.get("gates")
    promotion_order = protocol.get("promotion_gate_order")
    if gates is None and promotion_order is None:
        return
    if not isinstance(gates, dict) or not gates:
        raise ValueError("gates must be a non-empty object.")

    required_fields = {
        "candidate",
        "baseline",
        "minimum_q95_gain",
        "require_positive_float",
        "minimum_dual_positive_folds",
        "minimum_test_like_roi_q95_gain",
        "maximum_dual_negative_test_like_rois",
    }
    allowed_fields = required_fields | {"requires"}
    dependencies: dict[str, str | None] = {}
    for gate_name, gate in gates.items():
        if (
            not isinstance(gate_name, str)
            or SAFE_NAME_RE.fullmatch(gate_name) is None
        ):
            raise ValueError(f"Invalid gate name: {gate_name!r}.")
        if not isinstance(gate, dict):
            raise ValueError(f"Gate {gate_name} must be an object.")
        missing = sorted(required_fields - set(gate))
        unexpected = sorted(set(gate) - allowed_fields)
        if missing or unexpected:
            raise ValueError(
                f"Gate {gate_name} fields mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        candidate = gate["candidate"]
        baseline = gate["baseline"]
        if candidate not in candidates or baseline not in candidates:
            raise ValueError(
                f"Gate {gate_name} must reference configured candidates."
            )
        if candidate == baseline:
            raise ValueError(f"Gate {gate_name} candidate and baseline must differ.")
        minimum_gain = _finite_number(
            gate["minimum_q95_gain"],
            f"gates.{gate_name}.minimum_q95_gain",
        )
        if minimum_gain < 0:
            raise ValueError(
                f"gates.{gate_name}.minimum_q95_gain must be non-negative."
            )
        _finite_number(
            gate["minimum_test_like_roi_q95_gain"],
            f"gates.{gate_name}.minimum_test_like_roi_q95_gain",
        )
        if type(gate["require_positive_float"]) is not bool:
            raise ValueError(
                f"gates.{gate_name}.require_positive_float must be boolean."
            )
        minimum_folds = gate["minimum_dual_positive_folds"]
        if (
            type(minimum_folds) is not int
            or not 1 <= minimum_folds <= fold_count
        ):
            raise ValueError(
                f"gates.{gate_name}.minimum_dual_positive_folds must be "
                f"from 1 to {fold_count}."
            )
        maximum_negative = gate["maximum_dual_negative_test_like_rois"]
        if (
            type(maximum_negative) is not int
            or not 0 <= maximum_negative <= roi_count
        ):
            raise ValueError(
                f"gates.{gate_name}.maximum_dual_negative_test_like_rois "
                f"must be from 0 to {roi_count}."
            )
        dependency = gate.get("requires")
        if dependency is not None and not isinstance(dependency, str):
            raise ValueError(f"gates.{gate_name}.requires must name one gate.")
        dependencies[gate_name] = dependency

    for gate_name, dependency in dependencies.items():
        if dependency is not None and dependency not in gates:
            raise ValueError(
                f"Gate {gate_name} requires unknown gate {dependency!r}."
            )
        if dependency == gate_name:
            raise ValueError(f"Gate {gate_name} cannot require itself.")

    for start in gates:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"Gate dependency cycle includes {current}.")
            seen.add(current)
            current = dependencies[current]

    if (
        not isinstance(promotion_order, list)
        or not promotion_order
        or any(not isinstance(name, str) for name in promotion_order)
        or len(set(promotion_order)) != len(promotion_order)
    ):
        raise ValueError("promotion_gate_order must be a non-empty unique list.")
    unknown = sorted(set(promotion_order) - set(gates))
    if unknown:
        raise ValueError(
            f"promotion_gate_order contains unknown gates: {unknown}."
        )


def _validate_distribution_gates(
    protocol: dict[str, object],
    candidates: dict[str, object],
    fold_count: int,
    roi_count: int,
) -> None:
    gates = protocol.get("distribution_gates")
    promotion_order = protocol.get("distribution_promotion_order")
    if gates is None and promotion_order is None:
        return
    if not isinstance(gates, dict) or not gates:
        raise ValueError("distribution_gates must be a non-empty object.")
    required_fields = {
        "candidate",
        "baseline",
        "minimum_q95_gain",
        "require_positive_float",
        "minimum_positive_folds",
        "minimum_positive_rois",
        "minimum_roi_q95_gain",
    }
    allowed_fields = required_fields | {"requires"}
    dependencies: dict[str, str | None] = {}
    for gate_name, gate in gates.items():
        if (
            not isinstance(gate_name, str)
            or SAFE_NAME_RE.fullmatch(gate_name) is None
        ):
            raise ValueError(f"Invalid distribution gate name: {gate_name!r}.")
        if not isinstance(gate, dict):
            raise ValueError(f"Distribution gate {gate_name} must be an object.")
        missing = sorted(required_fields - set(gate))
        unexpected = sorted(set(gate) - allowed_fields)
        if missing or unexpected:
            raise ValueError(
                f"Distribution gate {gate_name} fields mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        candidate = gate["candidate"]
        baseline = gate["baseline"]
        if candidate not in candidates or baseline not in candidates:
            raise ValueError(
                f"Distribution gate {gate_name} must reference configured candidates."
            )
        if candidate == baseline:
            raise ValueError(
                f"Distribution gate {gate_name} candidate and baseline must differ."
            )
        minimum_gain = _finite_number(
            gate["minimum_q95_gain"],
            f"distribution_gates.{gate_name}.minimum_q95_gain",
        )
        if minimum_gain < 0:
            raise ValueError(
                f"distribution_gates.{gate_name}.minimum_q95_gain must be "
                "non-negative."
            )
        _finite_number(
            gate["minimum_roi_q95_gain"],
            f"distribution_gates.{gate_name}.minimum_roi_q95_gain",
        )
        if type(gate["require_positive_float"]) is not bool:
            raise ValueError(
                f"distribution_gates.{gate_name}.require_positive_float must be "
                "boolean."
            )
        minimum_folds = gate["minimum_positive_folds"]
        if type(minimum_folds) is not int or not 1 <= minimum_folds <= fold_count:
            raise ValueError(
                f"distribution_gates.{gate_name}.minimum_positive_folds must be "
                f"from 1 to {fold_count}."
            )
        minimum_rois = gate["minimum_positive_rois"]
        if type(minimum_rois) is not int or not 1 <= minimum_rois <= roi_count:
            raise ValueError(
                f"distribution_gates.{gate_name}.minimum_positive_rois must be "
                f"from 1 to {roi_count}."
            )
        dependency = gate.get("requires")
        if dependency is not None and not isinstance(dependency, str):
            raise ValueError(
                f"distribution_gates.{gate_name}.requires must name one gate."
            )
        dependencies[gate_name] = dependency

    for gate_name, dependency in dependencies.items():
        if dependency is not None and dependency not in gates:
            raise ValueError(
                f"Distribution gate {gate_name} requires unknown gate "
                f"{dependency!r}."
            )
        if dependency == gate_name:
            raise ValueError(f"Distribution gate {gate_name} cannot require itself.")
    for start in gates:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(
                    f"Distribution gate dependency cycle includes {current}."
                )
            seen.add(current)
            current = dependencies[current]

    if (
        not isinstance(promotion_order, list)
        or not promotion_order
        or any(not isinstance(name, str) for name in promotion_order)
        or len(set(promotion_order)) != len(promotion_order)
    ):
        raise ValueError(
            "distribution_promotion_order must be a non-empty unique list."
        )
    unknown = sorted(set(promotion_order) - set(gates))
    if unknown:
        raise ValueError(
            f"distribution_promotion_order contains unknown gates: {unknown}."
        )


def validate_protocol(protocol: object) -> dict[str, object]:
    if not isinstance(protocol, dict) or protocol.get("version") != 1:
        raise ValueError("CV protocol must be a version-1 JSON object.")
    name = protocol.get("name")
    if not isinstance(name, str) or SAFE_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"Invalid protocol name: {name!r}.")
    marker = protocol.get("marker")
    if not isinstance(marker, str) or not marker:
        raise ValueError("CV protocol marker must be a non-empty string.")

    expected_rois = protocol.get("expected_rois")
    if not isinstance(expected_rois, list) or not expected_rois:
        raise ValueError("expected_rois must be a non-empty list.")
    if any(not isinstance(roi, str) or ROI_RE.fullmatch(roi) is None for roi in expected_rois):
        raise ValueError("expected_rois contains an invalid ROI name.")
    if len(set(expected_rois)) != len(expected_rois):
        raise ValueError("expected_rois contains duplicates.")
    roi_scope = protocol.get("roi_scope", "complete")
    if roi_scope not in {"complete", "subset"}:
        raise ValueError("roi_scope must be complete or subset.")

    folds = protocol.get("folds")
    confirmation_only = protocol.get("confirmation_only", False)
    if type(confirmation_only) is not bool:
        raise ValueError("confirmation_only must be boolean.")
    if confirmation_only:
        if roi_scope != "subset" or not isinstance(folds, list) or len(folds) != 1:
            raise ValueError("Confirmation-only protocols require one subset fold.")
    elif not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("folds must contain at least two folds.")
    fold_names: set[str] = set()
    seen_rois: set[str] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("Each fold must be an object.")
        fold_name = fold.get("name")
        if not isinstance(fold_name, str) or SAFE_NAME_RE.fullmatch(fold_name) is None:
            raise ValueError(f"Invalid fold name: {fold_name!r}.")
        if fold_name in fold_names:
            raise ValueError(f"Duplicate fold name: {fold_name}.")
        fold_names.add(fold_name)
        val_rois = fold.get("val_rois")
        if not isinstance(val_rois, list) or not val_rois:
            raise ValueError(f"{fold_name} must contain validation ROIs.")
        if any(
            not isinstance(roi, str) or ROI_RE.fullmatch(roi) is None
            for roi in val_rois
        ):
            raise ValueError(f"{fold_name} contains an invalid ROI name.")
        if len(set(val_rois)) != len(val_rois):
            raise ValueError(f"{fold_name} contains duplicate validation ROIs.")
        overlap = sorted(seen_rois.intersection(val_rois))
        if overlap:
            raise ValueError(f"Validation ROIs occur in multiple folds: {overlap}.")
        seen_rois.update(val_rois)
    if seen_rois != set(expected_rois):
        missing = sorted(set(expected_rois) - seen_rois)
        extra = sorted(seen_rois - set(expected_rois))
        raise ValueError(f"Fold ROI coverage mismatch: missing={missing}, extra={extra}.")

    recipes = protocol.get("recipes")
    if not isinstance(recipes, dict) or len(recipes) < 2:
        raise ValueError("recipes must contain at least two entries.")
    for recipe_name, recipe in recipes.items():
        if not isinstance(recipe_name, str) or SAFE_NAME_RE.fullmatch(recipe_name) is None:
            raise ValueError(f"Invalid recipe name: {recipe_name!r}.")
        if not isinstance(recipe, dict):
            raise ValueError(f"Recipe {recipe_name} must be an object.")
        context_halo = recipe.get("context_halo")
        target_halo = recipe.get("train_target_context_halo")
        if not isinstance(context_halo, int) or not isinstance(target_halo, int):
            raise ValueError(f"Recipe {recipe_name} halos must be integers.")
        if context_halo < 0 or target_halo < 0 or target_halo > context_halo:
            raise ValueError(f"Recipe {recipe_name} has invalid context halos.")
        target_offset = recipe.get("train_target_source_offset")
        validation_offset = recipe.get("validation_output_source_offset")
        if (target_offset is None) != (validation_offset is None):
            raise ValueError(
                f"Recipe {recipe_name} target and validation offsets must be "
                "configured together."
            )
        if target_offset is not None:
            for field, offset in (
                ("train_target_source_offset", target_offset),
                ("validation_output_source_offset", validation_offset),
            ):
                if (
                    not isinstance(offset, list)
                    or len(offset) != 2
                    or any(
                        type(value) is not int or not -8 <= value <= 8
                        for value in offset
                    )
                ):
                    raise ValueError(
                        f"Recipe {recipe_name} {field} must contain two integers "
                        "from -8 to 8."
                    )
            if any(
                target_offset[index] + validation_offset[index] != 0
                for index in range(2)
            ):
                raise ValueError(
                    f"Recipe {recipe_name} target and validation offsets must be "
                    "opposites."
                )
            if max(abs(value) for value in target_offset) > target_halo:
                raise ValueError(
                    f"Recipe {recipe_name} target offset exceeds its target halo."
                )
        inference_context_halo = recipe.get("inference_context_halo")
        if inference_context_halo is not None and (
            type(inference_context_halo) is not int
            or not 0 <= inference_context_halo <= MAX_CONTEXT_HALO
        ):
            raise ValueError(
                f"Recipe {recipe_name} inference_context_halo must be an integer "
                f"from 0 to {MAX_CONTEXT_HALO}."
            )
        train_stages = recipe.get("train_stages", [1, 2])
        if (
            not isinstance(train_stages, list)
            or any(stage not in {1, 2} for stage in train_stages)
            or len(set(train_stages)) != len(train_stages)
            or train_stages != sorted(train_stages)
        ):
            raise ValueError(
                f"Recipe {recipe_name} train_stages must be an ordered subset of [1, 2]."
            )
        activity_sampler = recipe.get("activity_sampler")
        if activity_sampler is not None:
            if not isinstance(activity_sampler, dict) or set(activity_sampler) != {
                "mode",
                "start_ratio",
                "hard_fraction",
                "num_bins",
            }:
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler must contain exactly "
                    "mode, start_ratio, hard_fraction, and num_bins."
                )
            if activity_sampler["mode"] != "hard-mix":
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler mode must be hard-mix."
                )
            start_ratio = _finite_number(
                activity_sampler["start_ratio"],
                f"recipes.{recipe_name}.activity_sampler.start_ratio",
            )
            hard_fraction = _finite_number(
                activity_sampler["hard_fraction"],
                f"recipes.{recipe_name}.activity_sampler.hard_fraction",
            )
            num_bins = activity_sampler["num_bins"]
            if not 0.0 <= start_ratio < 1.0:
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler.start_ratio must be in [0, 1)."
                )
            if not 0.0 < hard_fraction < 1.0:
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler.hard_fraction must be in (0, 1)."
                )
            if type(num_bins) is not int or num_bins < 2:
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler.num_bins must be at least two."
                )
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} activity_sampler requires training stages."
                )
        recipe_epochs = recipe.get("epochs_per_stage")
        if recipe_epochs is not None and (
            type(recipe_epochs) is not int or recipe_epochs <= 0
        ):
            raise ValueError(
                f"Recipe {recipe_name} epochs_per_stage must be a positive integer."
            )
        resolved_epochs = int(
            recipe_epochs
            if recipe_epochs is not None
            else protocol["training"]["epochs_per_stage"]
        )
        loss_profile = recipe.get(
            "loss_profile", protocol.get("training", {}).get("loss_profile", "hybrid")
        )
        if loss_profile not in {"hybrid", "exact-score"}:
            raise ValueError(
                f"Recipe {recipe_name} loss_profile must be hybrid or exact-score."
            )
        try:
            target_l1_strength = float(recipe.get("target_l1_weight_strength", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Recipe {recipe_name} target_l1_weight_strength must be numeric."
            ) from error
        if loss_profile == "exact-score" and target_l1_strength != 0.0:
            raise ValueError(
                f"Recipe {recipe_name} exact-score cannot use target L1 weighting."
            )
        pyramid_weight = _finite_number(
            recipe.get("pyramid_weight", 0.0),
            f"recipes.{recipe_name}.pyramid_weight",
        )
        if pyramid_weight < 0.0:
            raise ValueError(
                f"Recipe {recipe_name} pyramid_weight must be non-negative."
            )
        if loss_profile == "exact-score" and pyramid_weight != 0.0:
            raise ValueError(
                f"Recipe {recipe_name} exact-score cannot use pyramid loss."
            )
        validation_evaluations = recipe.get("validation_evaluations")
        if validation_evaluations is not None:
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} validation_evaluations requires "
                    "training stages."
                )
            try:
                evenly_spaced_epochs(resolved_epochs, validation_evaluations)
            except ValueError as error:
                raise ValueError(
                    f"Recipe {recipe_name} has invalid validation_evaluations."
                ) from error
        train_rois_by_fold = recipe.get("train_rois_by_fold")
        if train_rois_by_fold is not None:
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} train_rois_by_fold requires training stages."
                )
            if not isinstance(train_rois_by_fold, dict) or set(
                train_rois_by_fold
            ) != fold_names:
                raise ValueError(
                    f"Recipe {recipe_name} train_rois_by_fold must define every fold."
                )
            for fold in folds:
                fold_name = str(fold["name"])
                train_rois = train_rois_by_fold[fold_name]
                if (
                    not isinstance(train_rois, list)
                    or not train_rois
                    or any(
                        not isinstance(roi, str) or ROI_RE.fullmatch(roi) is None
                        for roi in train_rois
                    )
                    or len(set(train_rois)) != len(train_rois)
                ):
                    raise ValueError(
                        f"Recipe {recipe_name} has invalid training ROIs for "
                        f"{fold_name}."
                    )
                overlap = sorted(set(train_rois).intersection(fold["val_rois"]))
                if overlap:
                    raise ValueError(
                        f"Recipe {recipe_name} training ROIs overlap {fold_name} "
                        f"validation ROIs: {overlap}."
                    )
        init_source = recipe.get("init_source")
        if init_source is not None:
            _validate_run_source(init_source, f"recipes.{recipe_name}.init_source")
        if train_stages == [2] and init_source is None:
            raise ValueError(
                f"Recipe {recipe_name} trains only stage 2 but has no init_source."
            )
        training_markers = recipe.get("training_markers")
        marker_loss_weights = recipe.get("marker_loss_weights")
        target_l1_weight_strength = _finite_number(
            recipe.get("target_l1_weight_strength", 0.0),
            f"recipes.{recipe_name}.target_l1_weight_strength",
        )
        if target_l1_weight_strength < 0:
            raise ValueError(
                f"Recipe {recipe_name} target_l1_weight_strength must be "
                "non-negative."
            )
        if target_l1_weight_strength > 0 and not train_stages:
            raise ValueError(
                f"Recipe {recipe_name} target L1 weighting requires training stages."
            )
        if target_offset is not None and not train_stages:
            raise ValueError(
                f"Recipe {recipe_name} target alignment requires training stages."
            )
        auxiliary_gradient_scope = recipe.get("auxiliary_gradient_scope", "all")
        if auxiliary_gradient_scope not in {"all", "decoder-only"}:
            raise ValueError(
                f"Recipe {recipe_name} has invalid auxiliary_gradient_scope: "
                f"{auxiliary_gradient_scope!r}."
            )
        if training_markers is None:
            if marker_loss_weights is not None:
                raise ValueError(
                    f"Recipe {recipe_name} marker_loss_weights requires "
                    "training_markers."
                )
        else:
            if (
                not isinstance(training_markers, list)
                or not training_markers
                or any(
                    not isinstance(training_marker, str) or not training_marker
                    for training_marker in training_markers
                )
                or len(set(training_markers)) != len(training_markers)
            ):
                raise ValueError(
                    f"Recipe {recipe_name} training_markers must be a non-empty "
                    "unique string list."
                )
            if marker not in training_markers:
                raise ValueError(
                    f"Recipe {recipe_name} training_markers must contain the "
                    f"protocol marker {marker}."
                )
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} training_markers requires a trained recipe."
                )
            if (
                not isinstance(marker_loss_weights, list)
                or len(marker_loss_weights) != len(training_markers)
            ):
                raise ValueError(
                    f"Recipe {recipe_name} marker_loss_weights must match "
                    "training_markers."
                )
            parsed_marker_weights = [
                _nonnegative_number(
                    value,
                    f"recipes.{recipe_name}.marker_loss_weights[{index}]",
                )
                for index, value in enumerate(marker_loss_weights)
            ]
            if abs(sum(parsed_marker_weights) - 1.0) > 1e-9:
                raise ValueError(
                    f"Recipe {recipe_name} marker_loss_weights must sum to one."
                )
        if recipe.get("init_add_conditioning", False) not in {True, False}:
            raise ValueError(f"Recipe {recipe_name} init_add_conditioning must be boolean.")
        model_arch = recipe.get("model_arch", "unet")
        if not isinstance(model_arch, str) or model_arch not in MODEL_ARCHITECTURES:
            raise ValueError(
                f"Recipe {recipe_name} has invalid model_arch: {model_arch!r}."
            )
        robust_objective = recipe.get("robust_objective", "none")
        projection = recipe.get("primary_gradient_projection", False)
        if type(projection) is not bool:
            raise ValueError("primary_gradient_projection must be boolean.")
        if projection and (
            model_arch != "unext2-fcmae-tiny" or not train_stages
            or training_markers != ["CD68", "CD45RO", "HLA-DR", "Vimentin"]
            or marker_loss_weights != [0.85, 0.05, 0.05, 0.05]
            or marker != "CD68" or loss_profile != "hybrid"
            or auxiliary_gradient_scope != "all" or robust_objective != "none"
            or recipe.get("activity_sampler") is not None
            or recipe.get("feature_guidance") is not None
            or recipe.get("d4_consistency_weight", 0) != 0
            or recipe.get("pyramid_weight", protocol.get("training", {}).get("pyramid_weight", 0)) != 0
            or target_l1_weight_strength != 0 or recipe.get("ema_decay", 0) != 0
        ):
            raise ValueError("Primary gradient projection requires the fixed Tiny auxiliary15 hybrid recipe.")
        if robust_objective not in {"none", "group-dro"}:
            raise ValueError(
                f"Recipe {recipe_name} has invalid robust_objective: {robust_objective!r}."
            )
        group_dro_eta = _finite_number(
            recipe.get("group_dro_eta", 5.0),
            f"recipes.{recipe_name}.group_dro_eta",
        )
        if group_dro_eta <= 0:
            raise ValueError(f"Recipe {recipe_name} group_dro_eta must be positive.")
        if robust_objective == "group-dro":
            if not train_stages or model_arch != "unext2-fcmae-tiny":
                raise ValueError(
                    f"Recipe {recipe_name} GroupDRO requires trained Tiny UNeXt2."
                )
            if training_markers is not None or marker_loss_weights is not None:
                raise ValueError(
                    f"Recipe {recipe_name} GroupDRO requires single-marker loss without marker weights."
                )
        dino_init = recipe.get("dinov2_encoder_init", False)
        if type(dino_init) is not bool:
            raise ValueError("dinov2_encoder_init must be boolean.")
        if dino_init and (
            model_arch != "dinov2-small-unet" or 1 not in train_stages
            or init_source is not None or recipe.get("fcmae_encoder_init", False)
            or recipe.get("cytoland_initialization") is not None
        ):
            raise ValueError("DINOv2 initialization requires a fresh DINOv2 stage1.")
        feature_guidance = recipe.get("feature_guidance")
        if feature_guidance is not None:
            required_fields = {
                "marker",
                "weight",
                "stage",
                "crop_size",
                "batch_samples",
                "teachers",
            }
            if (not isinstance(feature_guidance, dict)
                or set(feature_guidance) not in (required_fields, required_fields | {"mode"})):
                raise ValueError(
                    f"Recipe {recipe_name} feature_guidance fields mismatch."
                )
            guidance_mode = feature_guidance.get("mode", "perceptual")
            if guidance_mode not in {"perceptual", "encoder-relational", "encoder-relational-robust"}:
                raise ValueError(f"Recipe {recipe_name} has unknown feature-guidance mode.")
            if guidance_mode in {"encoder-relational", "encoder-relational-robust"} and (
                feature_guidance["stage"] != 4 or feature_guidance["batch_samples"] < 3
                or recipe.get("gradient_accumulation_steps", 1) != 1
                or recipe.get("auxiliary_gradient_scope", "all") != "all"
            ):
                raise ValueError("Encoder relational guidance requires stage4 and one full batch of >=3 samples.")
            if model_arch != "unext2-fcmae-tiny" or not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} feature guidance requires trained Tiny UNeXt2."
                )
            if loss_profile != "hybrid":
                raise ValueError(
                    f"Recipe {recipe_name} feature guidance requires hybrid loss."
                )
            guidance_marker = feature_guidance["marker"]
            active_markers = training_markers or [marker]
            if guidance_marker not in active_markers:
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance marker is not trained."
                )
            guidance_weight = _positive_number(
                feature_guidance["weight"],
                f"recipes.{recipe_name}.feature_guidance.weight",
            )
            if guidance_weight > 1.0:
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance weight cannot exceed one."
                )
            stage = feature_guidance["stage"]
            crop_size = feature_guidance["crop_size"]
            batch_samples = feature_guidance["batch_samples"]
            if type(stage) is not int or not 1 <= stage <= 4:
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance stage must be from 1 to 4."
                )
            if (
                type(crop_size) is not int
                or crop_size <= 0
                or crop_size > 256
                or crop_size % 16
            ):
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance crop must be a multiple "
                    "of 16 no greater than 256."
                )
            if type(batch_samples) is not int or batch_samples <= 0:
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance batch_samples must be positive."
                )
            teachers = feature_guidance["teachers"]
            if not isinstance(teachers, dict) or set(teachers) != fold_names:
                raise ValueError(
                    f"Recipe {recipe_name} feature-guidance teachers must define every fold."
                )
            for fold_name, teacher in teachers.items():
                if not isinstance(teacher, dict) or set(teacher) != {"checkpoint", "sha256"}:
                    raise ValueError(
                        f"Recipe {recipe_name} has invalid teacher for {fold_name}."
                    )
                checkpoint = teacher["checkpoint"]
                digest = teacher["sha256"]
                if not isinstance(checkpoint, str) or not checkpoint:
                    raise ValueError(
                        f"Recipe {recipe_name} teacher checkpoint must be a path string."
                    )
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(
                        f"Recipe {recipe_name} teacher SHA-256 is invalid for {fold_name}."
                    )
        encoder_lr_multiplier = recipe.get("encoder_learning_rate_multiplier")
        if encoder_lr_multiplier is not None:
            parsed_encoder_multiplier = _finite_number(
                encoder_lr_multiplier,
                f"recipes.{recipe_name}.encoder_learning_rate_multiplier",
            )
            if not 0.0 < parsed_encoder_multiplier <= 1.0:
                raise ValueError(
                    f"Recipe {recipe_name} encoder_learning_rate_multiplier "
                    "must be in (0, 1]."
                )
            if model_arch not in {
                "convnext-tiny-unet",
                "dinov2-small-unet",
                "unext2-fcmae-tiny-dinov2-residual",
                "unext2-fcmae-tiny-dinov2-fusion",
                "unext2-fcmae-tiny",
                "unext2-fcmae-base",
                "unext2-fcmae-tiny-widehead",
                "unext2-fcmae-tiny-marker-decoders",
                "unext2-fcmae-tiny-cross-marker-feedback",
                "unext2-fcmae-tiny-dapi-detail-residual",
                "unext2-fcmae-tiny-shallow-dapi-skip",
            }:
                raise ValueError(
                    f"Recipe {recipe_name} encoder learning-rate grouping "
                    "requires ConvNeXt U-Net or UNeXt2."
                )
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} encoder learning-rate grouping "
                    "requires training stages."
                )
        encoder_drop_path_rate = _finite_number(
            recipe.get("encoder_drop_path_rate", 0.0),
            f"recipes.{recipe_name}.encoder_drop_path_rate",
        )
        if not 0.0 <= encoder_drop_path_rate < 1.0:
            raise ValueError(
                f"Recipe {recipe_name} encoder_drop_path_rate must be in [0, 1)."
            )
        if encoder_drop_path_rate > 0.0:
            if model_arch not in {
                "unext2-fcmae-tiny",
                "unext2-fcmae-tiny-dinov2-residual",
                "unext2-fcmae-tiny-dinov2-fusion",
                "unext2-fcmae-base",
                "unext2-fcmae-tiny-widehead",
                "unext2-fcmae-tiny-marker-decoders",
                "unext2-fcmae-tiny-cross-marker-feedback",
                "unext2-fcmae-tiny-dapi-detail-residual",
                "unext2-fcmae-tiny-shallow-dapi-skip",
            }:
                raise ValueError(
                    f"Recipe {recipe_name} encoder drop path requires UNeXt2."
                )
            if not train_stages:
                raise ValueError(
                    f"Recipe {recipe_name} encoder drop path requires training stages."
                )
        if auxiliary_gradient_scope == "decoder-only":
            if model_arch != "unext2-fcmae-tiny":
                raise ValueError(
                    f"Recipe {recipe_name} decoder-only auxiliary gradients "
                    "require unext2-fcmae-tiny."
                )
            if training_markers is None or marker_loss_weights is None:
                raise ValueError(
                    f"Recipe {recipe_name} decoder-only auxiliary gradients "
                    "require weighted multi-marker training."
                )
            if len(training_markers) < 2 or training_markers[0] != marker:
                raise ValueError(
                    f"Recipe {recipe_name} decoder-only auxiliary gradients "
                    "require the protocol marker first."
                )
        mixstyle_probability = _finite_number(
            recipe.get("mixstyle_probability", 0.0),
            f"recipes.{recipe_name}.mixstyle_probability",
        )
        if not 0.0 <= mixstyle_probability <= 1.0:
            raise ValueError(
                f"Recipe {recipe_name} mixstyle_probability must be in [0, 1]."
            )
        mixstyle_alpha = _finite_number(
            recipe.get("mixstyle_alpha", 0.1),
            f"recipes.{recipe_name}.mixstyle_alpha",
        )
        if mixstyle_alpha <= 0.0:
            raise ValueError(
                f"Recipe {recipe_name} mixstyle_alpha must be positive."
            )
        if mixstyle_probability > 0.0 and model_arch != "unet":
            raise ValueError(
                f"Recipe {recipe_name} enables MixStyle for non-U-Net architecture."
            )
        fcmae_encoder_init = recipe.get("fcmae_encoder_init", False)
        if type(fcmae_encoder_init) is not bool:
            raise ValueError(
                f"Recipe {recipe_name} fcmae_encoder_init must be boolean."
            )
        if fcmae_encoder_init:
            if model_arch not in {
                "unext2-fcmae-tiny",
                "unext2-fcmae-tiny-dinov2-residual",
                "unext2-fcmae-tiny-dinov2-fusion",
                "unext2-fcmae-base",
                "unext2-fcmae-tiny-widehead",
                "unext2-fcmae-tiny-marker-decoders",
                "unext2-fcmae-tiny-cross-marker-feedback",
                "unext2-fcmae-tiny-dapi-detail-residual",
                "unext2-fcmae-tiny-shallow-dapi-skip",
            }:
                raise ValueError(
                    f"Recipe {recipe_name} FCMAE initialization requires UNeXt2."
                )
            source_override = recipe.get("fcmae_encoder_source")
            if source_override is not None:
                if not isinstance(source_override, dict) or set(source_override) != {"checkpoint", "sha256"}:
                    raise ValueError(
                        f"Recipe {recipe_name} fcmae_encoder_source must contain checkpoint and sha256."
                    )
                checkpoint = source_override["checkpoint"]
                digest = source_override["sha256"]
                if (
                    not isinstance(checkpoint, str) or not checkpoint
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"Recipe {recipe_name} fcmae_encoder_source is invalid.")
            if 1 not in train_stages or init_source is not None:
                raise ValueError(
                    f"Recipe {recipe_name} FCMAE initialization requires fresh stage1."
                )
        cytoland_initialization = recipe.get("cytoland_initialization")
        cytoland_scope = recipe.get("cytoland_initialization_scope")
        if (cytoland_initialization is None) != (cytoland_scope is None):
            raise ValueError(
                f"Recipe {recipe_name} Cytoland initialization requires artifact "
                "and scope together."
            )
        if cytoland_initialization is not None:
            if cytoland_initialization != CYTOLAND_VSCYTO2D_INITIALIZATION:
                raise ValueError(
                    f"Recipe {recipe_name} Cytoland initialization does not match "
                    "the audited public artifact."
                )
            if cytoland_scope not in {"encoder", "encoder-decoder"}:
                raise ValueError(
                    f"Recipe {recipe_name} has invalid Cytoland initialization scope."
                )
            if model_arch != "unext2-fcmae-tiny":
                raise ValueError(
                    f"Recipe {recipe_name} Cytoland initialization requires narrow "
                    "Tiny UNeXt2."
                )
            if 1 not in train_stages or init_source is not None or fcmae_encoder_init:
                raise ValueError(
                    f"Recipe {recipe_name} Cytoland initialization requires fresh "
                    "stage1 without FCMAE initialization."
                )
            if training_markers is not None and training_markers != [marker]:
                raise ValueError(
                    f"Recipe {recipe_name} Cytoland initialization requires the "
                    "single protocol marker."
                )
        source_photometric_augmentation = recipe.get(
            "source_photometric_augmentation", False
        )
        if type(source_photometric_augmentation) is not bool:
            raise ValueError(
                f"Recipe {recipe_name} source_photometric_augmentation must be boolean."
            )
        paired_affine = recipe.get("paired_affine_augmentation")
        if paired_affine is not None:
            required_affine_fields = {
                "probability",
                "outer_halo",
                "degrees",
                "scale_min",
                "scale_max",
            }
            if not isinstance(paired_affine, dict):
                raise ValueError(
                    f"Recipe {recipe_name} paired_affine_augmentation must be an object."
                )
            missing = sorted(required_affine_fields - set(paired_affine))
            unexpected = sorted(set(paired_affine) - required_affine_fields)
            if missing or unexpected:
                raise ValueError(
                    f"Recipe {recipe_name} paired affine fields mismatch: "
                    f"missing={missing}, unexpected={unexpected}."
                )
            probability = _finite_number(
                paired_affine["probability"],
                f"recipes.{recipe_name}.paired_affine_augmentation.probability",
            )
            if not 0.0 < probability <= 1.0:
                raise ValueError(
                    f"Recipe {recipe_name} paired affine probability must be in (0, 1]."
                )
            outer_halo = paired_affine["outer_halo"]
            if (
                type(outer_halo) is not int
                or not context_halo <= outer_halo <= MAX_CONTEXT_HALO
            ):
                raise ValueError(
                    f"Recipe {recipe_name} paired affine outer_halo must be an integer "
                    f"from {context_halo} to {MAX_CONTEXT_HALO}."
                )
            degrees = _finite_number(
                paired_affine["degrees"],
                f"recipes.{recipe_name}.paired_affine_augmentation.degrees",
            )
            if not 0.0 < degrees <= 45.0:
                raise ValueError(
                    f"Recipe {recipe_name} paired affine degrees must be in (0, 45]."
                )
            scale_min = _positive_number(
                paired_affine["scale_min"],
                f"recipes.{recipe_name}.paired_affine_augmentation.scale_min",
            )
            scale_max = _positive_number(
                paired_affine["scale_max"],
                f"recipes.{recipe_name}.paired_affine_augmentation.scale_max",
            )
            if scale_min > scale_max:
                raise ValueError(
                    f"Recipe {recipe_name} paired affine scale_min exceeds scale_max."
                )
            angle = math.radians(degrees)
            coverage_factor = math.cos(angle) + math.sin(angle)
            loaded_side = 256 + 2 * outer_halo
            requested_side = 256 + 2 * context_halo
            covered_side = loaded_side * scale_min / coverage_factor
            if covered_side + 1e-9 < requested_side:
                raise ValueError(
                    f"Recipe {recipe_name} paired affine outer context cannot cover "
                    "the requested crop at minimum scale and maximum rotation."
                )
        base_channels = recipe.get("base_channels", 32)
        if (
            isinstance(base_channels, bool)
            or not isinstance(base_channels, int)
            or base_channels <= 0
        ):
            raise ValueError(
                f"Recipe {recipe_name} has invalid base_channels: {base_channels!r}."
            )
        block_type = recipe.get("block_type", "residual")
        if not isinstance(block_type, str) or block_type not in BLOCK_TYPES:
            raise ValueError(
                f"Recipe {recipe_name} has invalid block_type: {block_type!r}."
            )
        try:
            ema_decay = float(recipe.get("ema_decay", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Recipe {recipe_name} ema_decay must be numeric."
            ) from error
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                f"Recipe {recipe_name} ema_decay must be in [0, 1), got {ema_decay}."
            )
        d4_consistency_weight = _finite_number(
            recipe.get("d4_consistency_weight", 0.0),
            f"recipes.{recipe_name}.d4_consistency_weight",
        )
        if d4_consistency_weight < 0.0:
            raise ValueError(
                f"Recipe {recipe_name} d4_consistency_weight must be non-negative."
            )
        if d4_consistency_weight > 0.0 and not train_stages:
            raise ValueError(
                f"Recipe {recipe_name} D4 consistency requires training stages."
            )

    fcmae_recipes = [
        name
        for name, recipe in recipes.items()
        if recipe.get("fcmae_encoder_init", False)
    ]
    fcmae_pretraining = protocol.get("fcmae_pretraining")
    if fcmae_recipes:
        required_pretraining_fields = {
            "name",
            "epochs",
            "batch_size",
            "workers",
            "learning_rate",
            "minimum_learning_rate",
            "warmup_epochs",
            "weight_decay",
            "mask_ratio",
            "checkpoint_interval",
            "seed",
        }
        if not isinstance(fcmae_pretraining, dict):
            raise ValueError(
                "fcmae_pretraining is required by FCMAE-initialized recipes."
            )
        missing = sorted(required_pretraining_fields - set(fcmae_pretraining))
        optional_pretraining_fields = {
            "data_scope",
            "encoder_initialization",
            "gradient_accumulation_steps",
            "model_size",
            "pretraining_markers",
            "shared_across_folds",
        }
        unexpected = sorted(
            set(fcmae_pretraining)
            - required_pretraining_fields
            - optional_pretraining_fields
        )
        if missing or unexpected:
            raise ValueError(
                "fcmae_pretraining fields mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        pretrain_name = fcmae_pretraining["name"]
        if (
            not isinstance(pretrain_name, str)
            or SAFE_NAME_RE.fullmatch(pretrain_name) is None
        ):
            raise ValueError("fcmae_pretraining.name is invalid.")
        for key in ("epochs", "batch_size", "checkpoint_interval"):
            value = fcmae_pretraining[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"fcmae_pretraining.{key} must be positive.")
        gradient_accumulation_steps = fcmae_pretraining.get(
            "gradient_accumulation_steps",
            1,
        )
        if (
            type(gradient_accumulation_steps) is not int
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError(
                "fcmae_pretraining.gradient_accumulation_steps must be positive."
            )
        model_size = fcmae_pretraining.get("model_size", "tiny")
        if model_size not in {"tiny", "base"}:
            raise ValueError("fcmae_pretraining.model_size must be tiny or base.")
        data_scope = fcmae_pretraining.get("data_scope", "fold-train")
        if data_scope not in {"fold-train", "all-train"}:
            raise ValueError(
                "fcmae_pretraining.data_scope must be fold-train or all-train."
            )
        pretraining_markers = fcmae_pretraining.get(
            "pretraining_markers",
            ["DAPI"],
        )
        allowed_pretraining_markers = {
            "DAPI",
            "CD45RO",
            "CD68",
            "HLA-DR",
            "Vimentin",
        }
        if (
            not isinstance(pretraining_markers, list)
            or not pretraining_markers
            or any(not isinstance(marker, str) for marker in pretraining_markers)
            or len(set(pretraining_markers)) != len(pretraining_markers)
            or set(pretraining_markers) - allowed_pretraining_markers
            or "DAPI" not in pretraining_markers
        ):
            raise ValueError(
                "fcmae_pretraining.pretraining_markers must be a unique non-empty "
                "list of official channels including DAPI."
            )
        shared_across_folds = fcmae_pretraining.get(
            "shared_across_folds",
            False,
        )
        if type(shared_across_folds) is not bool:
            raise ValueError(
                "fcmae_pretraining.shared_across_folds must be boolean."
            )
        if shared_across_folds and data_scope != "all-train":
            raise ValueError(
                "Shared FCMAE pretraining requires data_scope=all-train."
            )
        compatible_architectures = {
            "tiny": {
                "unext2-fcmae-tiny",
                "unext2-fcmae-tiny-widehead",
                "unext2-fcmae-tiny-marker-decoders",
                "unext2-fcmae-tiny-cross-marker-feedback",
                "unext2-fcmae-tiny-dapi-detail-residual",
                "unext2-fcmae-tiny-shallow-dapi-skip",
                "unext2-fcmae-tiny-dinov2-residual",
                "unext2-fcmae-tiny-dinov2-fusion",
            },
            "base": {"unext2-fcmae-base"},
        }[model_size]
        incompatible_recipes = sorted(
            recipe_name
            for recipe_name in fcmae_recipes
            if recipes[recipe_name].get("model_arch", "unet")
            not in compatible_architectures
        )
        if incompatible_recipes:
            raise ValueError(
                "FCMAE pretraining model_size disagrees with initialized recipes: "
                f"{incompatible_recipes}."
            )
        workers = fcmae_pretraining["workers"]
        if type(workers) is not int or workers < 0:
            raise ValueError("fcmae_pretraining.workers must be non-negative.")
        warmup_epochs = fcmae_pretraining["warmup_epochs"]
        if (
            type(warmup_epochs) is not int
            or not 0 <= warmup_epochs < fcmae_pretraining["epochs"]
        ):
            raise ValueError(
                "fcmae_pretraining.warmup_epochs must be below epochs."
            )
        if type(fcmae_pretraining["seed"]) is not int:
            raise ValueError("fcmae_pretraining.seed must be an integer.")
        learning_rate = _positive_number(
            fcmae_pretraining["learning_rate"],
            "fcmae_pretraining.learning_rate",
        )
        minimum_learning_rate = _nonnegative_number(
            fcmae_pretraining["minimum_learning_rate"],
            "fcmae_pretraining.minimum_learning_rate",
        )
        if minimum_learning_rate >= learning_rate:
            raise ValueError(
                "fcmae_pretraining.minimum_learning_rate must be below learning_rate."
            )
        _nonnegative_number(
            fcmae_pretraining["weight_decay"],
            "fcmae_pretraining.weight_decay",
        )
        mask_ratio = _finite_number(
            fcmae_pretraining["mask_ratio"],
            "fcmae_pretraining.mask_ratio",
        )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("fcmae_pretraining.mask_ratio must be in (0, 1).")
        encoder_initialization = fcmae_pretraining.get("encoder_initialization")
        if encoder_initialization is not None:
            if not isinstance(encoder_initialization, dict):
                raise ValueError(
                    "fcmae_pretraining.encoder_initialization must be an object."
                )
            audited_initialization = AUDITED_PUBLIC_ENCODER_INITIALIZATIONS.get(
                encoder_initialization.get("mode")
            )
            if encoder_initialization != audited_initialization:
                raise ValueError(
                    "fcmae_pretraining.encoder_initialization does not match the "
                    "audited official/public encoder artifacts."
                )
    elif fcmae_pretraining is not None:
        raise ValueError(
            "fcmae_pretraining is present but no recipe consumes its encoder."
        )

    run_sources = protocol.get("run_sources")
    if run_sources is not None:
        if not isinstance(run_sources, dict) or set(run_sources) != set(recipes):
            raise ValueError("run_sources must define exactly one source for each recipe.")
        for recipe_name, source in run_sources.items():
            _validate_run_source(source, f"run_sources.{recipe_name}")

    training = protocol.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object.")
    epochs = training.get("epochs_per_stage")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs_per_stage must be a positive integer.")
    for key in (
        "stage1_learning_rate",
        "stage2_learning_rate",
        "weight_decay",
        "ssim_weight",
        "l1_weight",
        "mse_weight",
    ):
        _positive_number(training.get(key), f"training.{key}")
    for key in ("batch_size", "workers"):
        value = training.get(key)
        if not isinstance(value, int) or value < (0 if key == "workers" else 1):
            raise ValueError(f"training.{key} is invalid: {value!r}.")
    if not isinstance(training.get("seed"), int):
        raise ValueError("training.seed must be an integer.")
    _nonnegative_number(
        training.get("minimum_learning_rate", 0.0),
        "training.minimum_learning_rate",
    )
    _nonnegative_number(
        training.get("gradient_clip_norm", 0.0),
        "training.gradient_clip_norm",
    )
    if type(training.get("save_every_epoch", True)) is not bool:
        raise ValueError("training.save_every_epoch must be boolean.")
    effective_batch_size = int(training["batch_size"])
    for recipe_name, recipe in recipes.items():
        micro_batch_size = recipe.get("batch_size", effective_batch_size)
        accumulation_steps = recipe.get("gradient_accumulation_steps", 1)
        if (
            isinstance(micro_batch_size, bool)
            or not isinstance(micro_batch_size, int)
            or micro_batch_size <= 0
        ):
            raise ValueError(
                f"Recipe {recipe_name} has invalid batch_size: {micro_batch_size!r}."
            )
        if (
            isinstance(accumulation_steps, bool)
            or not isinstance(accumulation_steps, int)
            or accumulation_steps <= 0
        ):
            raise ValueError(
                "Recipe "
                f"{recipe_name} has invalid gradient_accumulation_steps: "
                f"{accumulation_steps!r}."
            )
        if micro_batch_size * accumulation_steps != effective_batch_size:
            raise ValueError(
                f"Recipe {recipe_name} effective batch must equal "
                f"training.batch_size={effective_batch_size}, got "
                f"{micro_batch_size} * {accumulation_steps}."
            )

    inference = protocol.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference must be an object.")
    if (
        type(inference.get("context_halo")) is not int
        or not 0 <= inference["context_halo"] <= MAX_CONTEXT_HALO
    ):
        raise ValueError(
            "inference.context_halo must be an integer from 0 to "
            f"{MAX_CONTEXT_HALO}."
        )
    if inference.get("tta") not in {"none", "d4"}:
        raise ValueError("inference.tta must be none or d4.")
    if not isinstance(inference.get("jpeg_quality"), int) or not 1 <= inference["jpeg_quality"] <= 100:
        raise ValueError("inference.jpeg_quality must be from 1 to 100.")
    if not isinstance(inference.get("batch_size"), int) or inference["batch_size"] <= 0:
        raise ValueError("inference.batch_size must be positive.")
    checkpoint_filename = inference.get("checkpoint_filename", "best_global_mse.pt")
    if (
        not isinstance(checkpoint_filename, str)
        or checkpoint_filename not in INFERENCE_CHECKPOINT_FILENAMES
    ):
        raise ValueError(
            "inference.checkpoint_filename must be best_global_mse.pt or "
            "best_mean_image.pt."
        )
    if type(inference.get("test_like_diagnostic", True)) is not bool:
        raise ValueError("inference.test_like_diagnostic must be boolean.")
    for recipe_name, recipe in recipes.items():
        recipe_halo = recipe.get("inference_context_halo", inference["context_halo"])
        if recipe_halo > inference["context_halo"]:
            raise ValueError(
                f"Recipe {recipe_name} inference_context_halo={recipe_halo} exceeds "
                f"the loader halo inference.context_halo={inference['context_halo']}."
            )

    candidates = protocol.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("candidates must be a non-empty object.")
    for candidate_name, weights in candidates.items():
        if not isinstance(candidate_name, str) or SAFE_NAME_RE.fullmatch(candidate_name) is None:
            raise ValueError(f"Invalid candidate name: {candidate_name!r}.")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Candidate {candidate_name} must define weights.")
        unknown = sorted(set(weights) - set(recipes))
        if unknown:
            raise ValueError(f"Candidate {candidate_name} uses unknown recipes: {unknown}.")
        parsed = [_positive_number(value, f"candidates.{candidate_name}.{key}") for key, value in weights.items()]
        if abs(sum(parsed) - 1.0) > 1e-9:
            raise ValueError(f"Candidate {candidate_name} weights must sum to one.")
    candidate_offsets = protocol.get("candidate_output_source_offsets", {})
    if not isinstance(candidate_offsets, dict):
        raise ValueError("candidate_output_source_offsets must be an object.")
    unknown_offset_candidates = sorted(set(candidate_offsets) - set(candidates))
    if unknown_offset_candidates:
        raise ValueError(
            "candidate_output_source_offsets references unknown candidates: "
            f"{unknown_offset_candidates}."
        )
    for candidate_name, offset in candidate_offsets.items():
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(value) is not int or not -8 <= value <= 8 for value in offset)
        ):
            raise ValueError(
                f"Candidate {candidate_name} output source offset must contain "
                "two integers from -8 to 8."
            )
    for field in ("baseline_candidate", "primary_candidate"):
        value = protocol.get(field)
        if value not in candidates:
            raise ValueError(f"{field} must name a configured candidate.")
    _validate_gates(protocol, candidates, len(folds), len(expected_rois))
    _validate_distribution_gates(
        protocol,
        candidates,
        len(folds),
        len(expected_rois),
    )
    return protocol


def load_protocol(path: Path) -> dict[str, object]:
    return validate_protocol(json.loads(path.read_text(encoding="utf-8")))


def recipe_inference_context_halo(protocol: dict[str, object], recipe_name: str) -> int:
    recipes = protocol["recipes"]
    if recipe_name not in recipes:
        raise ValueError(f"Unknown recipe: {recipe_name}.")
    recipe = recipes[recipe_name]
    return int(
        recipe.get(
            "inference_context_halo",
            protocol["inference"]["context_halo"],
        )
    )


def inference_checkpoint_filename(protocol: dict[str, object]) -> str:
    filename = str(
        protocol["inference"].get(
            "checkpoint_filename",
            "best_global_mse.pt",
        )
    )
    if filename not in INFERENCE_CHECKPOINT_FILENAMES:
        raise ValueError(f"Unsupported inference checkpoint filename: {filename!r}.")
    return filename


def candidate_output_source_offset(
    protocol: dict[str, object], candidate_name: str
) -> tuple[int, int]:
    if candidate_name not in protocol["candidates"]:
        raise ValueError(f"Unknown candidate: {candidate_name}.")
    offsets = protocol.get("candidate_output_source_offsets", {})
    raw = offsets.get(candidate_name, [0, 0]) if isinstance(offsets, dict) else [0, 0]
    return int(raw[0]), int(raw[1])


def run_directory(
    output_root: Path,
    protocol_name: str,
    fold_name: str,
    recipe_name: str,
    stage: int,
) -> Path:
    if stage not in {1, 2}:
        raise ValueError(f"Stage must be 1 or 2, got {stage}.")
    return output_root / f"{protocol_name}_{fold_name}_{recipe_name}_stage{stage}"


def fcmae_pretraining_directory(
    output_root: Path,
    protocol: dict[str, object],
    fold_name: str,
) -> Path:
    pretraining = protocol.get("fcmae_pretraining")
    if not isinstance(pretraining, dict):
        raise ValueError("Protocol does not define fcmae_pretraining.")
    fold_component = (
        "shared"
        if bool(pretraining.get("shared_across_folds", False))
        else fold_name
    )
    return output_root / (
        f"{protocol['name']}_{fold_component}_{pretraining['name']}_pretraining"
    )


def fcmae_encoder_checkpoint(
    output_root: Path,
    protocol: dict[str, object],
    fold_name: str,
) -> Path:
    return fcmae_pretraining_directory(output_root, protocol, fold_name) / (
        "final_encoder.pt"
    )


def _validate_run_source(source: object, field: str) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"{field} must be an object.")
    if set(source) != {"protocol_name", "recipe_name", "stage"}:
        raise ValueError(
            f"{field} must contain protocol_name, recipe_name, and stage only."
        )
    for key in ("protocol_name", "recipe_name"):
        value = source[key]
        if not isinstance(value, str) or SAFE_NAME_RE.fullmatch(value) is None:
            raise ValueError(f"{field}.{key} is invalid: {value!r}.")
    if source["stage"] not in {1, 2}:
        raise ValueError(f"{field}.stage must be 1 or 2.")


def sourced_run_directory(
    output_root: Path,
    protocol: dict[str, object],
    fold_name: str,
    recipe_name: str,
) -> Path:
    sources = protocol.get("run_sources")
    source = (
        sources[recipe_name]
        if isinstance(sources, dict)
        else {
            "protocol_name": protocol["name"],
            "recipe_name": recipe_name,
            "stage": 2,
        }
    )
    return run_directory(
        output_root,
        str(source["protocol_name"]),
        fold_name,
        str(source["recipe_name"]),
        int(source["stage"]),
    )


def initialization_checkpoint(
    output_root: Path,
    protocol: dict[str, object],
    fold_name: str,
    recipe_name: str,
    stage: int,
) -> Path | None:
    recipe = protocol["recipes"][recipe_name]
    source = recipe.get("init_source")
    if source is not None:
        return (
            run_directory(
                output_root,
                str(source["protocol_name"]),
                fold_name,
                str(source["recipe_name"]),
                int(source["stage"]),
            )
            / "best_global_mse.pt"
        )
    if stage == 2:
        return (
            run_directory(
                output_root,
                str(protocol["name"]),
                fold_name,
                recipe_name,
                1,
            )
            / "best_global_mse.pt"
        )
    return None
