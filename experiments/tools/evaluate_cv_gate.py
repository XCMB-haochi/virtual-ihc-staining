from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping

from cv_protocol import load_protocol


SCORE_KEYS = (
    "competition_score_mean_image_psnr",
    "competition_score_global_mse_psnr",
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def score_pair(metrics: object, label: str) -> tuple[float, float]:
    values = _mapping(metrics, label)
    result: list[float] = []
    for key in SCORE_KEYS:
        raw = values.get(key)
        if isinstance(raw, bool):
            raise ValueError(f"{label}.{key} must be numeric.")
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.{key} must be numeric.") from error
        if not math.isfinite(parsed):
            raise ValueError(f"{label}.{key} must be finite.")
        result.append(parsed)
    return result[0], result[1]


def subtract_scores(
    candidate: object,
    baseline: object,
    label: str,
) -> list[float]:
    candidate_scores = score_pair(candidate, f"{label}.candidate")
    baseline_scores = score_pair(baseline, f"{label}.baseline")
    return [
        candidate_scores[index] - baseline_scores[index]
        for index in range(len(SCORE_KEYS))
    ]


def _candidate_variant(
    container: object,
    candidate: str,
    variant: str,
    label: str,
) -> Mapping[str, object]:
    candidates = _mapping(container, label)
    candidate_metrics = _mapping(
        candidates.get(candidate),
        f"{label}.{candidate}",
    )
    return _mapping(
        candidate_metrics.get(variant),
        f"{label}.{candidate}.{variant}",
    )


def _variant_delta(
    container: object,
    candidate: str,
    baseline: str,
    variant: str,
    label: str,
) -> list[float]:
    return subtract_scores(
        _candidate_variant(container, candidate, variant, label),
        _candidate_variant(container, baseline, variant, label),
        f"{label}.{variant}.{candidate}-vs-{baseline}",
    )


def evaluate_gate(
    protocol: dict[str, object],
    report: Mapping[str, object],
    gate_name: str,
    dependency_passed: bool,
) -> dict[str, object]:
    gates = _mapping(protocol["gates"], "protocol.gates")
    gate = _mapping(gates[gate_name], f"protocol.gates.{gate_name}")
    candidate = str(gate["candidate"])
    baseline = str(gate["baseline"])
    quality_key = f"jpeg_q{protocol['inference']['jpeg_quality']}"

    overall_q95 = _variant_delta(
        report.get("overall"), candidate, baseline, quality_key, "overall"
    )
    test_like_q95 = _variant_delta(
        report.get("test_like"), candidate, baseline, quality_key, "test_like"
    )
    overall_float = _variant_delta(
        report.get("overall"), candidate, baseline, "float", "overall"
    )
    test_like_float = _variant_delta(
        report.get("test_like"), candidate, baseline, "float", "test_like"
    )

    fold_metrics = _mapping(report.get("fold_metrics"), "fold_metrics")
    expected_folds = [str(fold["name"]) for fold in protocol["folds"]]
    if set(fold_metrics) != set(expected_folds):
        raise ValueError(
            "OOF fold metrics do not match the protocol: "
            f"expected={expected_folds}, observed={sorted(fold_metrics)}."
        )
    fold_deltas = {
        fold_name: _variant_delta(
            fold_metrics[fold_name],
            candidate,
            baseline,
            quality_key,
            f"fold_metrics.{fold_name}",
        )
        for fold_name in expected_folds
    }
    dual_positive_folds = sum(
        all(value > 0 for value in delta) for delta in fold_deltas.values()
    )

    raw_test_like_rois = report.get("test_like_train_rois")
    if (
        not isinstance(raw_test_like_rois, list)
        or not raw_test_like_rois
        or any(not isinstance(roi, str) for roi in raw_test_like_rois)
        or len(set(raw_test_like_rois)) != len(raw_test_like_rois)
    ):
        raise ValueError("test_like_train_rois must be a non-empty unique list.")
    test_like_rois = sorted(raw_test_like_rois)
    unknown_rois = sorted(set(test_like_rois) - set(protocol["expected_rois"]))
    if unknown_rois:
        raise ValueError(f"OOF report contains unknown test-like ROIs: {unknown_rois}.")
    roi_metrics = _mapping(report.get("roi_metrics"), "roi_metrics")
    candidate_roi = _candidate_variant(
        roi_metrics,
        candidate,
        quality_key,
        "roi_metrics",
    )
    baseline_roi = _candidate_variant(
        roi_metrics,
        baseline,
        quality_key,
        "roi_metrics",
    )
    roi_deltas = {
        roi: subtract_scores(
            candidate_roi.get(roi),
            baseline_roi.get(roi),
            f"roi_metrics.{quality_key}.{roi}.{candidate}-vs-{baseline}",
        )
        for roi in test_like_rois
    }
    dual_negative_rois = [
        roi for roi, delta in roi_deltas.items() if all(value < 0 for value in delta)
    ]
    minimum_roi_gain = min(
        value for delta in roi_deltas.values() for value in delta
    )

    minimum_q95_gain = float(gate["minimum_q95_gain"])
    require_positive_float = bool(gate["require_positive_float"])
    checks = {
        "overall_q95": min(overall_q95) >= minimum_q95_gain,
        "test_like_q95": min(test_like_q95) >= minimum_q95_gain,
        "float": (
            not require_positive_float
            or min(overall_float + test_like_float) > 0
        ),
        "folds": dual_positive_folds
        >= int(gate["minimum_dual_positive_folds"]),
        "roi_floor": minimum_roi_gain
        >= float(gate["minimum_test_like_roi_q95_gain"]),
        "roi_count": len(dual_negative_rois)
        <= int(gate["maximum_dual_negative_test_like_rois"]),
        "dependency": dependency_passed,
    }
    return {
        "passed": all(checks.values()),
        "candidate": candidate,
        "baseline": baseline,
        "requires": gate.get("requires"),
        "checks": checks,
        "overall_q95_delta": overall_q95,
        "test_like_q95_delta": test_like_q95,
        "overall_float_delta": overall_float,
        "test_like_float_delta": test_like_float,
        "fold_q95_deltas": fold_deltas,
        "dual_positive_folds": dual_positive_folds,
        "test_like_roi_q95_deltas": roi_deltas,
        "dual_negative_test_like_rois": dual_negative_rois,
        "minimum_test_like_roi_q95_gain": minimum_roi_gain,
    }


def evaluate_gates(
    protocol: dict[str, object],
    report: Mapping[str, object],
) -> dict[str, object]:
    if report.get("protocol_name") != protocol["name"]:
        raise ValueError(
            "OOF report protocol mismatch: "
            f"{report.get('protocol_name')!r} != {protocol['name']!r}."
        )
    gates = _mapping(protocol["gates"], "protocol.gates")
    results: dict[str, dict[str, object]] = {}
    active: set[str] = set()

    def evaluate_named(gate_name: str) -> dict[str, object]:
        if gate_name in results:
            return results[gate_name]
        if gate_name in active:
            raise ValueError(f"Gate dependency cycle includes {gate_name}.")
        active.add(gate_name)
        gate = _mapping(gates[gate_name], f"protocol.gates.{gate_name}")
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

    selected_gate: str | None = None
    for name in protocol["promotion_gate_order"]:
        if results[str(name)]["passed"]:
            selected_gate = str(name)
            break
    selected_candidate = (
        None if selected_gate is None else results[selected_gate]["candidate"]
    )
    return {
        "version": 1,
        "protocol_name": protocol["name"],
        "quality_variant": f"jpeg_q{protocol['inference']['jpeg_quality']}",
        "gates": results,
        "promotion": {
            "allowed": selected_gate is not None,
            "gate": selected_gate,
            "candidate": selected_candidate,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        description="Apply pre-registered promotion gates to a CV OOF report."
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
