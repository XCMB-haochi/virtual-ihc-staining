from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_torch import (
    build_model_from_checkpoint,
    checkpoint_conditioning,
    checkpoint_model_specification,
)


REQUIRED_KEYS = {
    "model",
    "epoch",
    "best_score",
    "markers",
    "base_channels",
    "block_type",
    "args",
    "selection_psnr_aggregation",
}
BUFFER_POLICIES = ("error", "latest", "average")
LOCAL_PATH_TYPE = type(Path())


def _canonical_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _same_existing_file(first: Path, second: Path) -> bool:
    try:
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def _read_checkpoint(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise ValueError(f"Checkpoint is not a file: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with torch.serialization.safe_globals([LOCAL_PATH_TYPE]):
        checkpoint = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    missing = sorted(REQUIRED_KEYS - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint {path} is missing required keys: {missing}")
    return checkpoint, digest


def _validate_metadata(checkpoint: Mapping[str, object], path: Path) -> None:
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Checkpoint {path} has an invalid epoch: {epoch!r}")

    markers = checkpoint["markers"]
    if not isinstance(markers, (list, tuple)) or not markers or not all(
        isinstance(marker, str) and marker for marker in markers
    ):
        raise ValueError(f"Checkpoint {path} has invalid markers: {markers!r}")

    base_channels = checkpoint["base_channels"]
    if (
        isinstance(base_channels, bool)
        or not isinstance(base_channels, int)
        or base_channels <= 0
    ):
        raise ValueError(
            f"Checkpoint {path} has invalid base_channels: {base_channels!r}"
        )
    if not isinstance(checkpoint["block_type"], str):
        raise ValueError(f"Checkpoint {path} has an invalid block_type.")
    if not isinstance(checkpoint["args"], dict):
        raise ValueError(f"Checkpoint {path} has an invalid args dictionary.")
    if not isinstance(checkpoint["selection_psnr_aggregation"], str):
        raise ValueError(
            f"Checkpoint {path} has an invalid selection_psnr_aggregation."
        )

    args = checkpoint["args"]
    conditioning = checkpoint_conditioning(checkpoint)
    checkpoint_model_specification(checkpoint)
    if args.get("base_channels") != base_channels:
        raise ValueError(
            f"Checkpoint {path} disagrees on base_channels between metadata and args."
        )
    if args.get("block_type") != checkpoint["block_type"]:
        raise ValueError(
            f"Checkpoint {path} disagrees on block_type between metadata and args."
        )
    if args.get("condition_mode", "none") != conditioning["mode"]:
        raise ValueError(
            f"Checkpoint {path} disagrees on conditioning mode between metadata and args."
        )
    if "conditioning_spec" in args and args["conditioning_spec"] != conditioning:
        raise ValueError(
            f"Checkpoint {path} disagrees on conditioning metadata and args."
        )
    argument_markers = args.get("markers")
    if not isinstance(argument_markers, str) or tuple(
        item.strip() for item in argument_markers.split(",") if item.strip()
    ) != tuple(markers):
        raise ValueError(
            f"Checkpoint {path} disagrees on markers between metadata and args."
        )


def _normalized_weights(weights: Sequence[float], count: int) -> list[float]:
    if len(weights) != count:
        raise ValueError(
            f"Expected one weight per checkpoint ({count}), received {len(weights)}."
        )
    converted = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight <= 0 for weight in converted):
        raise ValueError("Checkpoint weights must all be positive and finite.")
    total = math.fsum(converted)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Checkpoint weight sum must be positive and finite.")
    return [weight / total for weight in converted]


def _validate_compatible_metadata(
    checkpoints: Sequence[Mapping[str, object]], paths: Sequence[Path]
) -> None:
    reference = checkpoints[0]
    reference_architecture = (
        tuple(reference["markers"]),
        checkpoint_model_specification(reference),
        checkpoint_conditioning(reference),
        reference["selection_psnr_aggregation"],
    )
    epochs: set[int] = set()
    for checkpoint, path in zip(checkpoints, paths):
        epoch = int(checkpoint["epoch"])
        if epoch in epochs:
            raise ValueError(
                f"Same-trajectory soup requires unique epochs; duplicate epoch {epoch}."
            )
        epochs.add(epoch)
        architecture = (
            tuple(checkpoint["markers"]),
            checkpoint_model_specification(checkpoint),
            checkpoint_conditioning(checkpoint),
            checkpoint["selection_psnr_aggregation"],
        )
        if architecture != reference_architecture:
            raise ValueError(f"Checkpoint architecture metadata differs: {path}")
        if checkpoint["args"] != reference["args"]:
            raise ValueError(
                f"Checkpoint args differ; inputs are not from the same trajectory: {path}"
            )


def _validate_state_dicts(
    checkpoints: Sequence[Mapping[str, object]],
    paths: Sequence[Path],
    expected_state: Mapping[str, torch.Tensor],
) -> None:
    expected_keys = set(expected_state)
    for checkpoint, path in zip(checkpoints, paths):
        state = checkpoint["model"]
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint {path} has an invalid model state dictionary.")
        actual_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Checkpoint {path} model keys differ; missing={missing}, extra={extra}"
            )
        finite_groups: dict[torch.dtype, list[torch.Tensor]] = {}
        for name, expected in expected_state.items():
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Checkpoint {path} state {name!r} is not a tensor.")
            if value.shape != expected.shape:
                raise ValueError(
                    f"Checkpoint {path} state {name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected.shape)}."
                )
            if value.dtype != expected.dtype:
                raise ValueError(
                    f"Checkpoint {path} state {name!r} has dtype {value.dtype}, "
                    f"expected {expected.dtype}."
                )
            if value.layout != expected.layout:
                raise ValueError(
                    f"Checkpoint {path} state {name!r} has layout {value.layout}, "
                    f"expected {expected.layout}."
                )
            if value.is_floating_point() or value.is_complex():
                finite_groups.setdefault(value.dtype, []).append(value.reshape(-1))
        for dtype, values in finite_groups.items():
            flattened = torch.cat(values)
            if not bool(torch.isfinite(flattened).all()):
                raise ValueError(
                    f"Checkpoint {path} has non-finite values in {dtype} model state."
                )


def _weighted_average(
    values: Sequence[torch.Tensor], weights: Sequence[float], name: str
) -> torch.Tensor:
    reference = values[0]
    if not reference.is_floating_point():
        raise ValueError(f"State {name!r} cannot be averaged because it is not floating point.")
    accumulator = torch.zeros(reference.shape, dtype=torch.float64, device="cpu")
    for value, weight in zip(values, weights):
        accumulator.add_(value.detach().to(device="cpu", dtype=torch.float64), alpha=weight)
    return accumulator.to(dtype=reference.dtype)


def _average_parameters(
    states: Sequence[Mapping[str, torch.Tensor]],
    parameter_names: Sequence[str],
    weights: Sequence[float],
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    groups: dict[torch.dtype, list[str]] = {}
    for name in parameter_names:
        reference = expected_state[name]
        if not reference.is_floating_point():
            raise ValueError(
                f"Parameter {name!r} cannot be averaged because it is not floating point."
            )
        groups.setdefault(reference.dtype, []).append(name)

    result: dict[str, torch.Tensor] = {}
    for dtype, names in groups.items():
        element_count = sum(expected_state[name].numel() for name in names)
        accumulator = torch.zeros(element_count, dtype=torch.float64, device="cpu")
        for state, weight in zip(states, weights):
            flattened = torch.cat([state[name].reshape(-1) for name in names])
            accumulator.add_(flattened.to(dtype=torch.float64), alpha=weight)
        averaged = accumulator.to(dtype=dtype)
        offset = 0
        for name in names:
            reference = expected_state[name]
            next_offset = offset + reference.numel()
            result[name] = averaged[offset:next_offset].reshape(reference.shape).clone()
            offset = next_offset
    return result


def average_checkpoints(
    input_paths: Sequence[Path],
    weights: Sequence[float],
    output_path: Path,
    buffer_policy: str = "error",
) -> dict[str, object]:
    paths = [Path(path) for path in input_paths]
    output_path = Path(output_path)
    if len(paths) < 2:
        raise ValueError("Checkpoint soup requires at least two input checkpoints.")
    if buffer_policy not in BUFFER_POLICIES:
        raise ValueError(
            f"Invalid buffer policy {buffer_policy!r}; expected one of {BUFFER_POLICIES}."
        )

    canonical_inputs = [_canonical_path(path) for path in paths]
    if len(canonical_inputs) != len(set(canonical_inputs)):
        raise ValueError("Input checkpoint paths must be unique.")
    if _canonical_path(output_path) in set(canonical_inputs) or any(
        _same_existing_file(output_path, path) for path in paths
    ):
        raise ValueError("Output checkpoint cannot overwrite an input checkpoint.")
    normalized_weights = _normalized_weights(weights, len(paths))

    loaded = [_read_checkpoint(path) for path in paths]
    checkpoints = [item[0] for item in loaded]
    hashes = [item[1] for item in loaded]
    for checkpoint, path in zip(checkpoints, paths):
        _validate_metadata(checkpoint, path)
    _validate_compatible_metadata(checkpoints, paths)

    reference = checkpoints[0]
    conditioning = checkpoint_conditioning(reference)
    model = build_model_from_checkpoint(reference, torch.device("cpu"))
    expected_state = model.state_dict()
    _validate_state_dicts(checkpoints, paths, expected_state)

    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    if set(expected_state) != parameter_names | buffer_names:
        raise ValueError("Model state contains entries that are neither parameters nor buffers.")
    if buffer_names and buffer_policy == "error":
        raise ValueError(
            "Model contains buffers; select --buffer-policy latest or average explicitly."
        )

    states = [checkpoint["model"] for checkpoint in checkpoints]
    averaged_parameters = _average_parameters(
        states,
        [name for name in expected_state if name in parameter_names],
        normalized_weights,
        expected_state,
    )
    averaged_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    latest_index = max(range(len(checkpoints)), key=lambda index: checkpoints[index]["epoch"])
    for name in expected_state:
        values = [state[name] for state in states]
        if name in parameter_names:
            averaged_state[name] = averaged_parameters[name]
        elif buffer_policy == "latest":
            averaged_state[name] = values[latest_index].detach().cpu().clone()
        elif buffer_policy == "average":
            if values[0].is_floating_point():
                averaged_state[name] = _weighted_average(
                    values, normalized_weights, name
                )
            else:
                averaged_state[name] = (
                    values[latest_index].detach().cpu().clone()
                )
        else:  # Covered above, retained as a defensive invariant.
            raise AssertionError(f"Unhandled buffer state: {name}")

    # Load once before writing so the result is guaranteed to be inference-compatible.
    model.load_state_dict(averaged_state, strict=True)
    epochs = [int(checkpoint["epoch"]) for checkpoint in checkpoints]
    provenance_inputs = [
        {
            "path": str(path.resolve()),
            "sha256": digest,
            "epoch": epoch,
            "weight": weight,
        }
        for path, digest, epoch, weight in zip(
            paths, hashes, epochs, normalized_weights
        )
    ]
    output_checkpoint = {
        "epoch": max(epochs),
        "model": averaged_state,
        "best_score": None,
        "markers": copy.deepcopy(reference["markers"]),
        "base_channels": reference["base_channels"],
        "block_type": reference["block_type"],
        "args": copy.deepcopy(reference["args"]),
        "selection_psnr_aggregation": reference["selection_psnr_aggregation"],
        "soup": {
            "format_version": 1,
            "method": "weighted_parameter_average",
            "buffer_policy": buffer_policy,
            "inputs": provenance_inputs,
        },
    }
    if "conditioning" in reference:
        output_checkpoint["conditioning"] = copy.deepcopy(conditioning)
    if "model_spec" in reference:
        output_checkpoint["model_spec"] = copy.deepcopy(
            checkpoint_model_specification(reference)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"Output path is not a file: {output_path}")
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as handle:
            torch.save(output_checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "output": str(output_path.resolve()),
        "sha256": output_hash,
        "epoch": max(epochs),
        "buffer_policy": buffer_policy,
        "inputs": provenance_inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a weighted parameter soup from checkpoints on one trajectory."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--weights", nargs="+", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--buffer-policy",
        choices=BUFFER_POLICIES,
        default="error",
        help="How to handle model buffers; default rejects models that have any.",
    )
    args = parser.parse_args()

    report = average_checkpoints(
        args.inputs,
        args.weights,
        args.output,
        buffer_policy=args.buffer_policy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
