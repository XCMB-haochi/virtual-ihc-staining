from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from cv_protocol import load_protocol


SCORE_KEYS = ("competition_score_official", "competition_score_mean_image_psnr")
EXPECTED_RECIPES = {
    "round35base32",
    "round35base48",
    "fcmae800",
    "officialfcmae800",
    "officialauxiliary15",
    "alldapiofficialfcmae630",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(container: object, candidate: str, variant: str, label: str) -> float:
    if not isinstance(container, dict):
        raise ValueError(f"{label} must be an object.")
    metrics = container.get(candidate)
    if not isinstance(metrics, dict) or not isinstance(metrics.get(variant), dict):
        raise ValueError(f"{label}.{candidate}.{variant} is missing.")
    values = metrics[variant]
    for key in SCORE_KEYS:
        value = values.get(key)
        if value is not None and not isinstance(value, bool):
            result = float(value)
            if math.isfinite(result):
                return result
    raise ValueError(f"{label}.{candidate}.{variant} has no finite official score.")


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
        description="Compare fixed Round90 context128 composition with Round67."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("candidate_oof", type=Path)
    parser.add_argument("baseline_oof", type=Path)
    parser.add_argument("baseline_gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-oof-sha256", required=True)
    parser.add_argument("--baseline-gate-sha256", required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    candidate_path = args.candidate_oof.resolve()
    baseline_path = args.baseline_oof.resolve()
    baseline_gate_path = args.baseline_gate.resolve()
    protocol = load_protocol(protocol_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_gate = json.loads(baseline_gate_path.read_text(encoding="utf-8"))
    expected_candidate = "tri_auxhalf_residual_half_context128"
    baseline_candidate = "tri_auxhalf_residual_half"
    expected_oof_sha = str(args.baseline_oof_sha256).lower()
    expected_gate_sha = str(args.baseline_gate_sha256).lower()
    if sha256_file(baseline_path) != expected_oof_sha:
        raise ValueError("Round90 baseline OOF SHA-256 mismatch.")
    if sha256_file(baseline_gate_path) != expected_gate_sha:
        raise ValueError("Round90 baseline gate SHA-256 mismatch.")
    if protocol["name"] != "round90_cv5_auxiliary_context128":
        raise ValueError("Round90 protocol identity mismatch.")
    if protocol["primary_candidate"] != expected_candidate:
        raise ValueError("Round90 primary candidate mismatch.")
    if protocol["inference"]["context_halo"] != 128:
        raise ValueError("Round90 inference halo is not 128.")
    if protocol["inference"]["tta"] != "d4":
        raise ValueError("Round90 TTA is not D4.")
    if candidate.get("protocol_name") != protocol["name"]:
        raise ValueError("Round90 candidate protocol name mismatch.")
    if baseline.get("protocol_name") != "round67_cv5_auxiliary_all_dapi_composition":
        raise ValueError("Round90 baseline protocol name mismatch.")
    for report, label in ((candidate, "candidate"), (baseline, "baseline")):
        if report.get("marker") != protocol["marker"]:
            raise ValueError(f"Round90 {label} marker mismatch.")
        if report.get("validation_images") != 6296:
            raise ValueError(f"Round90 {label} image coverage is incomplete.")
        if report.get("validation_rois") != sorted(protocol["expected_rois"]):
            raise ValueError(f"Round90 {label} ROI coverage is incomplete.")
    if candidate.get("checkpoint_filename") != "best_global_mse.pt":
        raise ValueError("Round90 candidate checkpoint selector mismatch.")
    if candidate.get("inference", {}).get("context_halo") != 128:
        raise ValueError("Round90 candidate report did not use context128.")
    if baseline.get("inference", {}).get("context_halo") != 64:
        raise ValueError("Round90 baseline report did not use context64.")
    if baseline_gate.get("promotion") != {
        "allowed": True,
        "gate": "auxhalf_residual_half",
        "candidate": baseline_candidate,
    }:
        raise ValueError("Round90 baseline promotion has changed.")
    provenance = candidate.get("checkpoint_provenance")
    baseline_provenance = baseline.get("checkpoint_provenance")
    if not isinstance(provenance, dict) or not isinstance(baseline_provenance, dict):
        raise ValueError("Round90 checkpoint provenance is missing.")
    expected_folds = [str(item["name"]) for item in protocol["folds"]]
    if set(provenance) != set(expected_folds) or set(baseline_provenance) != set(expected_folds):
        raise ValueError("Round90 checkpoint fold coverage mismatch.")
    bad_hashes: list[str] = []
    for fold in expected_folds:
        current = provenance[fold]
        reference = baseline_provenance[fold]
        if set(current) != EXPECTED_RECIPES or set(reference) != EXPECTED_RECIPES:
            raise ValueError(f"Round90 recipe coverage mismatch for {fold}.")
        for recipe in sorted(EXPECTED_RECIPES):
            current_record = current[recipe]
            reference_record = reference[recipe]
            if current_record.get("sha256") != reference_record.get("sha256"):
                raise ValueError(f"Round90 checkpoint drift for {fold}/{recipe}.")
            if current_record.get("inference_context_halo") != 128:
                raise ValueError(f"Round90 candidate halo drift for {fold}/{recipe}.")
            checkpoint = Path(current_record["path"])
            actual = sha256_file(checkpoint) if checkpoint.is_file() else None
            if actual != current_record.get("sha256"):
                bad_hashes.append(f"{fold}/{recipe}")
    if bad_hashes:
        raise ValueError(f"Round90 checkpoint hashes mismatch: {bad_hashes}")

    variant = f"jpeg_q{protocol['inference']['jpeg_quality']}"
    overall_q95_delta = score(
        candidate["overall"], expected_candidate, variant, "candidate.overall"
    ) - score(
        baseline["overall"], baseline_candidate, variant, "baseline.overall"
    )
    overall_float_delta = score(
        candidate["overall"], expected_candidate, "float", "candidate.overall"
    ) - score(
        baseline["overall"], baseline_candidate, "float", "baseline.overall"
    )
    fold_q95_deltas = {
        fold: score(
            candidate["fold_metrics"][fold], expected_candidate, variant, f"candidate.{fold}"
        )
        - score(
            baseline["fold_metrics"][fold], baseline_candidate, variant, f"baseline.{fold}"
        )
        for fold in expected_folds
    }
    positive_folds = sorted(fold for fold, delta in fold_q95_deltas.items() if delta > 0.0)
    candidate_rois = candidate["roi_metrics"][expected_candidate][variant]
    baseline_rois = baseline["roi_metrics"][baseline_candidate][variant]
    expected_rois = sorted(str(roi) for roi in protocol["expected_rois"])
    if set(candidate_rois) != set(expected_rois) or set(baseline_rois) != set(expected_rois):
        raise ValueError("Round90 ROI metric coverage mismatch.")
    roi_q95_deltas = {
        roi: float(candidate_rois[roi]["competition_score_official"])
        - float(baseline_rois[roi]["competition_score_official"])
        for roi in expected_rois
    }
    positive_rois = sorted(roi for roi, delta in roi_q95_deltas.items() if delta > 0.0)
    minimum_roi_gain = min(roi_q95_deltas.values())
    gate_config = protocol["context128_gate"]
    checks = {
        "overall_q95": overall_q95_delta >= float(gate_config["minimum_q95_gain"]),
        "overall_float": (
            overall_float_delta > 0.0
            if bool(gate_config["require_positive_float"])
            else True
        ),
        "positive_folds": len(positive_folds) >= int(gate_config["minimum_positive_folds"]),
        "positive_rois": len(positive_rois) >= int(gate_config["minimum_positive_rois"]),
        "roi_floor": minimum_roi_gain >= float(gate_config["minimum_roi_q95_gain"]),
    }
    report = {
        "version": 1,
        "protocol_name": protocol["name"],
        "comparison": "round90_context128_vs_round67_context64",
        "quality_variant": variant,
        "candidate": expected_candidate,
        "baseline": baseline_candidate,
        "context_halos": {"candidate": 128, "baseline": 64},
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
        "candidate_oof": str(candidate_path),
        "candidate_oof_sha256": sha256_file(candidate_path),
        "baseline_oof": str(baseline_path),
        "baseline_oof_sha256": expected_oof_sha,
        "baseline_gate": str(baseline_gate_path),
        "baseline_gate_sha256": expected_gate_sha,
    }
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
