from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cv_protocol import load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be a {qualifier} integer.")
    return value


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric.")
    parsed = float(value)
    if not (-float("inf") < parsed < float("inf")):
        raise ValueError(f"{field} must be finite.")
    return parsed


def resolve_project_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty project-relative path.")
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{field} must stay inside the project root.") from error
    return path


def validate_refiner_protocol(protocol: object) -> dict[str, object]:
    if not isinstance(protocol, dict):
        raise ValueError("Refiner protocol must be a JSON object.")
    if protocol.get("version") != 1:
        raise ValueError("Refiner protocol version must be 1.")
    for field in ("name", "marker", "base_protocol", "base_candidate"):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")

    recipe_order = protocol.get("recipe_order")
    if (
        not isinstance(recipe_order, list)
        or len(recipe_order) != 2
        or any(not isinstance(item, str) or not item for item in recipe_order)
        or len(set(recipe_order)) != len(recipe_order)
    ):
        raise ValueError("recipe_order must contain exactly two unique recipe names.")

    weights = protocol.get("baseline_weights")
    if not isinstance(weights, dict) or set(weights) != set(recipe_order):
        raise ValueError("baseline_weights must match recipe_order exactly.")
    parsed_weights = {
        name: _finite_number(weights[name], f"baseline_weights.{name}")
        for name in recipe_order
    }
    if any(value < 0 for value in parsed_weights.values()):
        raise ValueError("baseline_weights must be non-negative.")
    if abs(sum(parsed_weights.values()) - 1.0) > 1e-9:
        raise ValueError("baseline_weights must sum to one.")

    cache = protocol.get("cache")
    if not isinstance(cache, dict):
        raise ValueError("cache must be an object.")
    expected_dtypes = {
        "prediction_dtype": "float32",
        "dapi_dtype": "uint8",
        "target_dtype": "uint8",
    }
    for field, expected in expected_dtypes.items():
        if cache.get(field) != expected:
            raise ValueError(f"cache.{field} must be {expected}.")

    model = protocol.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object.")
    width = _positive_int(model.get("width"), "model.width")
    if width % 8:
        raise ValueError("model.width must be divisible by 8 for GroupNorm.")
    dilations = model.get("dilations")
    if (
        not isinstance(dilations, list)
        or not dilations
        or any(type(value) is not int or value <= 0 for value in dilations)
    ):
        raise ValueError("model.dilations must contain positive integers.")
    residual_scale = _finite_number(
        model.get("residual_scale"), "model.residual_scale"
    )
    if not 0 < residual_scale <= 1:
        raise ValueError("model.residual_scale must be in (0, 1].")

    training = protocol.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object.")
    for field in ("epochs", "batch_size", "workers", "seed"):
        _positive_int(
            training.get(field),
            f"training.{field}",
            allow_zero=field == "workers",
        )
    for field in (
        "learning_rate",
        "weight_decay",
        "ssim_weight",
        "l1_weight",
        "mse_weight",
    ):
        value = _finite_number(training.get(field), f"training.{field}")
        if value < 0 or (field == "learning_rate" and value == 0):
            raise ValueError(f"training.{field} has an invalid range.")
    loss_sum = sum(
        float(training[field]) for field in ("ssim_weight", "l1_weight", "mse_weight")
    )
    if abs(loss_sum - 1.0) > 1e-9:
        raise ValueError("The three training loss weights must sum to one.")
    if training.get("ssim_mode") != "uniform7":
        raise ValueError("training.ssim_mode must be uniform7.")
    if training.get("selection_psnr_aggregation") != "global-mse":
        raise ValueError("selection_psnr_aggregation must be global-mse.")
    quality = _positive_int(
        training.get("selection_jpeg_quality"),
        "training.selection_jpeg_quality",
    )
    if quality > 100:
        raise ValueError("training.selection_jpeg_quality cannot exceed 100.")
    if training.get("selection_tta") not in {"none", "d4"}:
        raise ValueError("training.selection_tta must be none or d4.")

    inference = protocol.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference must be an object.")
    if inference.get("tta") not in {"none", "d4"}:
        raise ValueError("inference.tta must be none or d4.")
    quality = _positive_int(
        inference.get("jpeg_quality"), "inference.jpeg_quality"
    )
    if quality > 100:
        raise ValueError("inference.jpeg_quality cannot exceed 100.")
    _positive_int(inference.get("batch_size"), "inference.batch_size")

    gate = protocol.get("promotion_gate")
    if not isinstance(gate, dict):
        raise ValueError("promotion_gate must be an object.")
    for field in (
        "minimum_overall_q95_gain",
        "minimum_test_like_q95_gain",
        "minimum_float_gain",
        "minimum_test_like_roi_q95_gain",
    ):
        _finite_number(gate.get(field), f"promotion_gate.{field}")
    dual_folds = _positive_int(
        gate.get("minimum_dual_positive_folds"),
        "promotion_gate.minimum_dual_positive_folds",
    )
    if dual_folds > 5:
        raise ValueError("minimum_dual_positive_folds cannot exceed five.")
    _positive_int(
        gate.get("maximum_dual_negative_test_like_rois"),
        "promotion_gate.maximum_dual_negative_test_like_rois",
        allow_zero=True,
    )

    base_path = resolve_project_path(protocol["base_protocol"], "base_protocol")
    base_protocol = load_protocol(base_path)
    if base_protocol["marker"] != protocol["marker"]:
        raise ValueError("Refiner and base protocols must use the same marker.")
    candidate_name = str(protocol["base_candidate"])
    if candidate_name not in base_protocol["candidates"]:
        raise ValueError(f"Unknown base candidate: {candidate_name}.")
    candidate = base_protocol["candidates"][candidate_name]
    if list(candidate) != recipe_order:
        raise ValueError("recipe_order must match the base candidate recipe order.")
    for name in recipe_order:
        if abs(float(candidate[name]) - parsed_weights[name]) > 1e-12:
            raise ValueError("baseline_weights must match the base candidate exactly.")
    if len(base_protocol["folds"]) != 5:
        raise ValueError("Round37 requires exactly five base folds.")
    return protocol


def load_refiner_protocol(path: Path) -> dict[str, object]:
    return validate_refiner_protocol(json.loads(path.read_text(encoding="utf-8")))


def load_base_protocol(protocol: dict[str, object]) -> dict[str, object]:
    return load_protocol(resolve_project_path(protocol["base_protocol"], "base_protocol"))


def validate_cache_manifest(
    cache_root: Path,
    protocol_path: Path,
    protocol: dict[str, object],
    base_protocol: dict[str, object],
) -> dict[str, object]:
    manifest_path = cache_root.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing OOF cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_folds = {
        str(fold["name"]): set(fold["val_rois"]) for fold in base_protocol["folds"]
    }
    fold_entries = manifest.get("folds")
    if not isinstance(fold_entries, list):
        raise ValueError("OOF cache manifest has no fold entries.")
    actual_folds = {
        str(fold.get("name")): set(fold.get("val_rois", []))
        for fold in fold_entries
        if isinstance(fold, dict)
    }
    if (
        manifest.get("version") != 1
        or manifest.get("refiner_protocol_sha256") != sha256_file(protocol_path)
        or manifest.get("protocol_name") != protocol["name"]
        or manifest.get("base_protocol_name") != base_protocol["name"]
        or manifest.get("base_candidate") != protocol["base_candidate"]
        or manifest.get("recipe_order") != protocol["recipe_order"]
        or manifest.get("baseline_weights") != protocol["baseline_weights"]
        or actual_folds != expected_folds
    ):
        raise ValueError("OOF cache manifest does not match the fixed protocol.")
    if sum(int(fold.get("images", 0)) for fold in fold_entries) != int(
        manifest.get("images", -1)
    ):
        raise ValueError("OOF cache manifest image counts are inconsistent.")
    return manifest
