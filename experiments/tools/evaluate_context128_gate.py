from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from cv_protocol import load_protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(container: object, candidate: str, variant: str, label: str) -> float:
    if not isinstance(container, dict):
        raise ValueError(f"{label} must be an object.")
    candidate_metrics = container.get(candidate)
    if not isinstance(candidate_metrics, dict):
        raise ValueError(f"{label}.{candidate} is missing.")
    metrics = candidate_metrics.get(variant)
    if not isinstance(metrics, dict):
        raise ValueError(f"{label}.{candidate}.{variant} is missing.")
    value = metrics.get("competition_score_official")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.{candidate}.{variant} has no score.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label}.{candidate}.{variant} score is not finite.")
    return result


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a larger-context OOF with an immutable baseline OOF."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("candidate_oof", type=Path)
    parser.add_argument("baseline_oof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol.resolve())
    candidate_report = json.loads(args.candidate_oof.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_oof.read_text(encoding="utf-8"))
    if not isinstance(candidate_report, dict) or not isinstance(baseline_report, dict):
        raise ValueError("OOF reports must be JSON objects.")
    expected_candidate = str(protocol["primary_candidate"])
    candidate_protocol_name = str(protocol["name"])
    gate_config = protocol.get("context_scale_gate")
    legacy_context128_gate = gate_config is None
    if legacy_context128_gate:
        gate_config = protocol.get("context128_gate")
    if not isinstance(gate_config, dict):
        raise ValueError("Protocol lacks context-scale gate metadata.")
    baseline_candidate = str(gate_config["baseline_report_candidate"])
    candidate_context_halo = int(
        gate_config.get("candidate_context_halo", protocol["inference"]["context_halo"])
    )
    baseline_context_halo = int(gate_config.get("baseline_context_halo", 64))
    baseline_protocol_name = str(
        gate_config.get("baseline_protocol_name", "round81_cv5_fixed_output_shift")
    )
    comparison = str(
        gate_config.get(
            "comparison",
            (
                "context128_vs_context64_fixed_shift"
                if legacy_context128_gate
                else f"context{candidate_context_halo}_vs_context{baseline_context_halo}"
            ),
        )
    )
    expected_baseline_sha = str(args.baseline_sha256).lower()
    actual_baseline_sha = sha256_file(args.baseline_oof.resolve())
    if actual_baseline_sha != expected_baseline_sha:
        raise ValueError(
            f"Baseline OOF SHA mismatch: expected {expected_baseline_sha}, "
            f"got {actual_baseline_sha}."
        )

    for report, expected_name, label in (
        (candidate_report, candidate_protocol_name, "candidate"),
        (baseline_report, baseline_protocol_name, "baseline"),
    ):
        if report.get("marker") != protocol["marker"]:
            raise ValueError(f"{label} marker does not match protocol.")
        if report.get("validation_images") != 6296:
            raise ValueError(f"{label} OOF image count is incomplete.")
        if report.get("validation_rois") != sorted(protocol["expected_rois"]):
            raise ValueError(f"{label} OOF ROI coverage is incomplete.")
    if candidate_report.get("protocol_name") != candidate_protocol_name:
        raise ValueError("Candidate OOF protocol name does not match.")
    if baseline_report.get("protocol_name") != baseline_protocol_name:
        raise ValueError("Baseline OOF protocol does not match the bound report.")
    if (
        candidate_report.get("inference", {}).get("context_halo")
        != candidate_context_halo
    ):
        raise ValueError("Candidate OOF context halo does not match the protocol.")
    if (
        baseline_report.get("inference", {}).get("context_halo")
        != baseline_context_halo
    ):
        raise ValueError("Baseline OOF context halo does not match the gate.")
    if candidate_report.get("inference", {}).get("tta") != "d4":
        raise ValueError("Candidate OOF TTA drifted from D4.")

    candidate_provenance = candidate_report.get("checkpoint_provenance")
    baseline_provenance = baseline_report.get("checkpoint_provenance")
    if not isinstance(candidate_provenance, dict) or not isinstance(
        baseline_provenance, dict
    ):
        raise ValueError("Checkpoint provenance is missing.")
    expected_recipes = set(protocol["recipes"])
    for fold in (str(item["name"]) for item in protocol["folds"]):
        current = candidate_provenance.get(fold)
        reference = baseline_provenance.get(fold)
        if not isinstance(current, dict) or not isinstance(reference, dict):
            raise ValueError(f"Missing checkpoint provenance for {fold}.")
        if set(current) != expected_recipes or set(reference) != expected_recipes:
            raise ValueError(f"Checkpoint recipe coverage mismatch for {fold}.")
        for recipe in expected_recipes:
            current_record = current[recipe]
            reference_record = reference[recipe]
            if current_record.get("sha256") != reference_record.get("sha256"):
                raise ValueError(f"Checkpoint drift for {fold}/{recipe}.")
            if current_record.get("inference_context_halo") != candidate_context_halo:
                raise ValueError(f"Context halo drift for {fold}/{recipe}.")
            if reference_record.get("inference_context_halo") != baseline_context_halo:
                raise ValueError(f"Baseline context halo drift for {fold}/{recipe}.")

    variant = f"jpeg_q{protocol['inference']['jpeg_quality']}"
    overall_q95_delta = score(
        candidate_report["overall"], expected_candidate, variant, "candidate.overall"
    ) - score(
        baseline_report["overall"], baseline_candidate, variant, "baseline.overall"
    )
    overall_float_delta = score(
        candidate_report["overall"], expected_candidate, "float", "candidate.overall"
    ) - score(
        baseline_report["overall"], baseline_candidate, "float", "baseline.overall"
    )
    expected_folds = [str(item["name"]) for item in protocol["folds"]]
    fold_q95_deltas = {
        fold: score(
            candidate_report["fold_metrics"][fold],
            expected_candidate,
            variant,
            f"candidate.fold_metrics.{fold}",
        )
        - score(
            baseline_report["fold_metrics"][fold],
            baseline_candidate,
            variant,
            f"baseline.fold_metrics.{fold}",
        )
        for fold in expected_folds
    }
    positive_folds = sorted(
        fold for fold, delta in fold_q95_deltas.items() if delta > 0.0
    )
    expected_rois = sorted(str(roi) for roi in protocol["expected_rois"])
    candidate_roi = candidate_report["roi_metrics"][expected_candidate][variant]
    baseline_roi = baseline_report["roi_metrics"][baseline_candidate][variant]
    if set(candidate_roi) != set(expected_rois) or set(baseline_roi) != set(expected_rois):
        raise ValueError("ROI metric coverage mismatch.")
    roi_q95_deltas = {
        roi: float(candidate_roi[roi]["competition_score_official"])
        - float(baseline_roi[roi]["competition_score_official"])
        for roi in expected_rois
    }
    positive_rois = sorted(roi for roi, delta in roi_q95_deltas.items() if delta > 0.0)
    minimum_roi_gain = min(roi_q95_deltas.values())
    checks = {
        "overall_q95": overall_q95_delta >= float(gate_config["minimum_q95_gain"]),
        "overall_float": (
            overall_float_delta > 0.0
            if bool(gate_config["require_positive_float"])
            else True
        ),
        "positive_folds": len(positive_folds)
        >= int(gate_config["minimum_positive_folds"]),
        "positive_rois": len(positive_rois)
        >= int(gate_config["minimum_positive_rois"]),
        "roi_floor": minimum_roi_gain >= float(gate_config["minimum_roi_q95_gain"]),
    }
    result = {
        "version": 1,
        "protocol_name": protocol["name"],
        "comparison": comparison,
        "quality_variant": variant,
        "candidate": expected_candidate,
        "baseline": baseline_candidate,
        "context_halos": {
            "candidate": candidate_context_halo,
            "baseline": baseline_context_halo,
        },
        "overall_q95_delta": overall_q95_delta,
        "overall_float_delta": overall_float_delta,
        "fold_q95_deltas": fold_q95_deltas,
        "positive_folds": positive_folds,
        "roi_q95_deltas": roi_q95_deltas,
        "positive_rois": positive_rois,
        "minimum_roi_q95_gain": minimum_roi_gain,
        "thresholds": gate_config,
        "checks": checks,
        "promotion": {
            "allowed": all(checks.values()),
            "candidate": expected_candidate if all(checks.values()) else None,
        },
        "candidate_oof": str(args.candidate_oof.resolve()),
        "candidate_oof_sha256": sha256_file(args.candidate_oof.resolve()),
        "baseline_oof": str(args.baseline_oof.resolve()),
        "baseline_oof_sha256": actual_baseline_sha,
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
