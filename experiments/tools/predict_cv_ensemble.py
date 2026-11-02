from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cv_protocol import (
    candidate_output_source_offset,
    inference_checkpoint_filename,
    load_protocol,
    recipe_inference_context_halo,
    sourced_run_directory,
)
from evaluate_cv_ensemble import (
    sha256_file,
    shift_prediction_edge,
    validate_checkpoint_fold,
)
from ihc_data import (
    image_names,
    load_context_gray,
    load_gray,
    resolve_data_root,
)
from ihc_torch import (
    batch_condition_tensor,
    build_model_from_checkpoint,
    center_crop_like,
    checkpoint_conditioning,
    crop_context_halo,
    predict_with_tta,
    save_grayscale_rgb,
)
from ihc_spatial import apply_complete_roi_conditioning as apply_input_conditioning


def effective_member_weights(
    protocol: dict[str, object], candidate_name: str
) -> dict[str, float]:
    candidates = protocol["candidates"]
    if candidate_name not in candidates:
        raise ValueError(f"Unknown CV candidate: {candidate_name}.")
    recipe_weights = candidates[candidate_name]
    fold_count = len(protocol["folds"])
    weights = {
        f"{fold['name']}/{recipe_name}": float(recipe_weight) / fold_count
        for fold in protocol["folds"]
        for recipe_name, recipe_weight in recipe_weights.items()
    }
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Effective CV member weights must sum to one.")
    return weights


def ensure_empty_prediction_directory(output: Path) -> None:
    existing = list(output.glob("*.jpg")) if output.is_dir() else []
    if existing:
        raise FileExistsError(
            f"Refusing to mix predictions with {len(existing)} existing JPEGs in {output}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict test images with a fixed fold and recipe ensemble."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--candidate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    candidate_name = args.candidate or str(protocol["primary_candidate"])
    member_weights = effective_member_weights(protocol, candidate_name)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_root = resolve_data_root(args.dataset)
    names = image_names(data_root / "test" / "DAPI")
    if not names:
        raise ValueError("Test DAPI directory contains no JPEG images.")
    ensure_empty_prediction_directory(args.output)
    inference = protocol["inference"]
    checkpoint_filename = inference_checkpoint_filename(protocol)
    marker = str(protocol["marker"])
    output_root = args.output_root.resolve()

    members: list[dict[str, object]] = []
    for fold in protocol["folds"]:
        expected_rois = set(fold["val_rois"])
        for recipe_name in protocol["candidates"][candidate_name]:
            member_id = f"{fold['name']}/{recipe_name}"
            checkpoint_path = (
                sourced_run_directory(
                    output_root,
                    protocol,
                    str(fold["name"]),
                    recipe_name,
                )
                / checkpoint_filename
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing CV checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            validate_checkpoint_fold(
                checkpoint, expected_rois, marker, checkpoint_path
            )
            markers = tuple(checkpoint["markers"])
            conditioning = checkpoint_conditioning(checkpoint)
            condition_by_roi = apply_input_conditioning(
                data_root / "test" / "DAPI",
                names,
                conditioning,
            )
            members.append(
                {
                    "id": member_id,
                    "fold": fold["name"],
                    "recipe": recipe_name,
                    "weight": member_weights[member_id],
                    "checkpoint": checkpoint_path,
                    "sha256": sha256_file(checkpoint_path),
                    "model": build_model_from_checkpoint(checkpoint, device).eval(),
                    "marker_index": markers.index(marker),
                    "conditioning": conditioning,
                    "condition_by_roi": condition_by_roi,
                    "inference_context_halo": recipe_inference_context_halo(
                        protocol, recipe_name
                    ),
                }
            )

    if {member["id"] for member in members} != set(member_weights):
        raise RuntimeError("Loaded CV members do not match the effective weight plan.")
    batch_size = int(inference["batch_size"])
    source_context_halo = int(inference["context_halo"])
    tta = str(inference["tta"])
    jpeg_quality = int(inference["jpeg_quality"])
    output_source_offset = candidate_output_source_offset(protocol, candidate_name)
    args.output.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            batch_names = names[start : start + batch_size]
            source_center = np.stack(
                [load_gray(data_root / "test" / "DAPI" / name) for name in batch_names]
            )
            source_context = np.stack(
                [
                    load_context_gray(
                        data_root / "test" / "DAPI", name, source_context_halo
                    )
                    for name in batch_names
                ]
            )
            tensor = (
                torch.from_numpy(source_context[:, None, ...] / 255.0)
                .float()
                .to(device)
            )
            reference = torch.from_numpy(source_center[:, None, ...])
            blended = None
            for member in members:
                member_tensor = crop_context_halo(
                    tensor,
                    source_context_halo,
                    int(member["inference_context_halo"]),
                )
                condition = batch_condition_tensor(
                    batch_names, member["condition_by_roi"], device
                )
                prediction = center_crop_like(
                    predict_with_tta(member["model"], member_tensor, tta, condition),
                    reference,
                )[:, member["marker_index"] : member["marker_index"] + 1]
                contribution = prediction * float(member["weight"])
                blended = contribution if blended is None else blended + contribution
            if blended is None:
                raise RuntimeError("CV prediction has no ensemble members.")
            blended_np = shift_prediction_edge(
                blended[:, 0].cpu().numpy(), output_source_offset
            )
            for index, name in enumerate(batch_names):
                save_grayscale_rgb(
                    blended_np[index],
                    args.output / name,
                    quality=jpeg_quality,
                )
            print(
                json.dumps(
                    {
                        "completed": min(start + batch_size, len(names)),
                        "total": len(names),
                    }
                ),
                flush=True,
            )

    actual = image_names(args.output)
    if actual != names:
        missing = sorted(set(names) - set(actual))
        extra = sorted(set(actual) - set(names))
        raise RuntimeError(
            f"Prediction filename mismatch: missing={missing}, extra={extra}."
        )
    report = {
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["name"],
        "candidate": candidate_name,
        "marker": marker,
        "test_images": len(names),
        "context_halo": source_context_halo,
        "tta": tta,
        "jpeg_quality": jpeg_quality,
        "output_source_offset": list(output_source_offset),
        "output": str(args.output.resolve()),
        "members": [
            {
                "id": member["id"],
                "fold": member["fold"],
                "recipe": member["recipe"],
                "weight": member["weight"],
                "checkpoint": str(member["checkpoint"].resolve()),
                "sha256": member["sha256"],
                "conditioning": member["conditioning"],
                "inference_context_halo": member["inference_context_halo"],
            }
            for member in members
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
