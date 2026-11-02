from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from cv_protocol import load_protocol


SCORE_KEYS = (
    "competition_score_official",
    "competition_score_mean_image_psnr",
)


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def official_score(metrics: object, label: str) -> float:
    values = mapping(metrics, label)
    for key in SCORE_KEYS:
        raw = values.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool):
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    raise ValueError(f"{label} has no finite official score.")


def candidate_variant(
    container: object,
    candidate: str,
    variant: str,
    label: str,
) -> dict[str, object]:
    candidates = mapping(container, label)
    candidate_metrics = mapping(
        candidates.get(candidate),
        f"{label}.{candidate}",
    )
    return mapping(
        candidate_metrics.get(variant),
        f"{label}.{candidate}.{variant}",
    )


def variant_delta(
    container: object,
    candidate: str,
    baseline: str,
    variant: str,
    label: str,
) -> float:
    return official_score(
        candidate_variant(container, candidate, variant, label),
        f"{label}.{candidate}.{variant}",
    ) - official_score(
        candidate_variant(container, baseline, variant, label),
        f"{label}.{baseline}.{variant}",
    )


def evaluate_gate(
    protocol: dict[str, object],
    report: dict[str, object],
    gate_name: str,
    dependency_passed: bool,
) -> dict[str, object]:
    gates = mapping(protocol["distribution_gates"], "distribution_gates")
    gate = mapping(gates[gate_name], f"distribution_gates.{gate_name}")
    candidate = str(gate["candidate"])
    baseline = str(gate["baseline"])
    quality_variant = f"jpeg_q{protocol['inference']['jpeg_quality']}"
    overall_q95_delta = variant_delta(
        report.get("overall"),
        candidate,
        baseline,
        quality_variant,
        "overall",
    )
    overall_float_delta = variant_delta(
        report.get("overall"),
        candidate,
        baseline,
        "float",
        "overall",
    )

    expected_folds = [str(fold["name"]) for fold in protocol["folds"]]
    fold_metrics = mapping(report.get("fold_metrics"), "fold_metrics")
    if set(fold_metrics) != set(expected_folds):
        raise ValueError("OOF fold metrics do not match the protocol.")
    fold_q95_deltas = {
        fold: variant_delta(
            fold_metrics[fold],
            candidate,
            baseline,
            quality_variant,
            f"fold_metrics.{fold}",
        )
        for fold in expected_folds
    }
    positive_folds = sorted(
        fold for fold, delta in fold_q95_deltas.items() if delta > 0.0
    )

    expected_rois = sorted(str(roi) for roi in protocol["expected_rois"])
    roi_metrics = mapping(report.get("roi_metrics"), "roi_metrics")
    candidate_rois = candidate_variant(
        roi_metrics,
        candidate,
        quality_variant,
        "roi_metrics",
    )
    baseline_rois = candidate_variant(
        roi_metrics,
        baseline,
        quality_variant,
        "roi_metrics",
    )
    if set(candidate_rois) != set(expected_rois) or set(baseline_rois) != set(
        expected_rois
    ):
        raise ValueError("OOF ROI metrics do not match the protocol.")
    roi_q95_deltas = {
        roi: official_score(
            candidate_rois[roi],
            f"roi_metrics.{candidate}.{quality_variant}.{roi}",
        )
        - official_score(
            baseline_rois[roi],
            f"roi_metrics.{baseline}.{quality_variant}.{roi}",
        )
        for roi in expected_rois
    }
    positive_rois = sorted(roi for roi, delta in roi_q95_deltas.items() if delta > 0.0)
    minimum_roi_gain = min(roi_q95_deltas.values())
    checks = {
        "overall_q95": overall_q95_delta >= float(gate["minimum_q95_gain"]),
        "overall_float": (
            overall_float_delta > 0.0 if bool(gate["require_positive_float"]) else True
        ),
        "positive_folds": len(positive_folds) >= int(gate["minimum_positive_folds"]),
        "positive_rois": len(positive_rois) >= int(gate["minimum_positive_rois"]),
        "roi_floor": minimum_roi_gain >= float(gate["minimum_roi_q95_gain"]),
        "dependency": dependency_passed,
    }
    return {
        "passed": all(checks.values()),
        "candidate": candidate,
        "baseline": baseline,
        "requires": gate.get("requires"),
        "checks": checks,
        "overall_q95_delta": overall_q95_delta,
        "overall_float_delta": overall_float_delta,
        "fold_q95_deltas": fold_q95_deltas,
        "positive_folds": positive_folds,
        "roi_q95_deltas": roi_q95_deltas,
        "positive_rois": positive_rois,
        "minimum_roi_q95_gain": minimum_roi_gain,
        "thresholds": gate,
    }


def evaluate_gates(
    protocol: dict[str, object],
    report: dict[str, object],
) -> dict[str, object]:
    if report.get("protocol_name") != protocol["name"]:
        raise ValueError("OOF report protocol name does not match the protocol.")
    if report.get("marker") != protocol["marker"]:
        raise ValueError("OOF report marker does not match the protocol.")
    if report.get("validation_rois") != sorted(protocol["expected_rois"]):
        raise ValueError("OOF report validation ROIs do not match the protocol.")
    gates = mapping(protocol["distribution_gates"], "distribution_gates")
    results: dict[str, dict[str, object]] = {}
    active: set[str] = set()

    def evaluate_named(gate_name: str) -> dict[str, object]:
        if gate_name in results:
            return results[gate_name]
        if gate_name in active:
            raise ValueError(
                f"Distribution gate dependency cycle includes {gate_name}."
            )
        active.add(gate_name)
        gate = mapping(gates[gate_name], f"distribution_gates.{gate_name}")
        dependency = gate.get("requires")
        dependency_passed = (
            True
            if dependency is None
            else bool(evaluate_named(str(dependency))["passed"])
        )
        result = evaluate_gate(
            protocol,
            report,
            gate_name,
            dependency_passed,
        )
        active.remove(gate_name)
        results[gate_name] = result
        return result

    for name in gates:
        evaluate_named(name)
    selected_gate = next(
        (
            str(name)
            for name in protocol["distribution_promotion_order"]
            if results[str(name)]["passed"]
        ),
        None,
    )
    result = {
        "version": 1,
        "protocol_name": protocol["name"],
        "quality_variant": f"jpeg_q{protocol['inference']['jpeg_quality']}",
        "gates": results,
        "promotion": {
            "allowed": selected_gate is not None,
            "gate": selected_gate,
            "candidate": (
                None if selected_gate is None else results[selected_gate]["candidate"]
            ),
        },
    }
    if protocol.get("confirmation_only", False):
        result["confirmation"] = {
            "passed": selected_gate is not None,
            "gate": selected_gate,
            "candidate": result["promotion"]["candidate"],
        }
        result["promotion"] = {"allowed": False, "gate": None, "candidate": None}
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
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
        description="Evaluate preregistered complete-distribution OOF gates."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("oof_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    report_path = args.oof_report.resolve()
    protocol = load_protocol(protocol_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("OOF report must be an object.")
    result = evaluate_gates(protocol, report)
    result["protocol"] = str(protocol_path)
    result["oof_report"] = str(report_path)
    result["oof_report_sha256"] = sha256_file(report_path)
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
