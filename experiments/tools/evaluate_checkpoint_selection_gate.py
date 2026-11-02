from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cv_protocol import load_protocol


SCORE_KEYS = (
    "competition_score_official",
    "competition_score_mean_image_psnr",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def official_score(metrics: object) -> float:
    if not isinstance(metrics, dict):
        raise ValueError("Metric entry must be an object.")
    for key in SCORE_KEYS:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"Metric entry has none of the official score keys: {SCORE_KEYS}.")


def report_checkpoint_filename(report: dict[str, object]) -> str:
    direct = report.get("checkpoint_filename")
    if direct is not None:
        return str(direct)
    inference = report.get("inference")
    if isinstance(inference, dict):
        return str(inference.get("checkpoint_filename", "best_global_mse.pt"))
    return "best_global_mse.pt"


def candidate_metric(
    report: dict[str, object],
    section: str,
    candidate: str,
    variant: str,
) -> float:
    container = report.get(section)
    if not isinstance(container, dict) or candidate not in container:
        raise ValueError(f"Report {section} has no candidate {candidate!r}.")
    candidate_entry = container[candidate]
    if not isinstance(candidate_entry, dict) or variant not in candidate_entry:
        raise ValueError(f"Report {section}.{candidate} has no variant {variant!r}.")
    return official_score(candidate_entry[variant])


def nested_metric(
    report: dict[str, object],
    section: str,
    outer_name: str,
    candidate: str,
    variant: str,
) -> float:
    container = report.get(section)
    if not isinstance(container, dict) or outer_name not in container:
        raise ValueError(f"Report {section} has no entry {outer_name!r}.")
    outer = container[outer_name]
    if not isinstance(outer, dict) or candidate not in outer:
        raise ValueError(
            f"Report {section}.{outer_name} has no candidate {candidate!r}."
        )
    candidate_entry = outer[candidate]
    if not isinstance(candidate_entry, dict) or variant not in candidate_entry:
        raise ValueError(
            f"Report {section}.{outer_name}.{candidate} has no variant {variant!r}."
        )
    return official_score(candidate_entry[variant])


def roi_metric(
    report: dict[str, object], candidate: str, variant: str, roi: str
) -> float:
    container = report.get("roi_metrics")
    if not isinstance(container, dict) or candidate not in container:
        raise ValueError(f"Report roi_metrics has no candidate {candidate!r}.")
    candidate_entry = container[candidate]
    if not isinstance(candidate_entry, dict) or variant not in candidate_entry:
        raise ValueError(f"Report roi_metrics.{candidate} has no variant {variant!r}.")
    variants = candidate_entry[variant]
    if not isinstance(variants, dict) or roi not in variants:
        raise ValueError(
            f"Report roi_metrics.{candidate}.{variant} has no ROI {roi!r}."
        )
    return official_score(variants[roi])


def compare_reports(
    protocol: dict[str, object],
    baseline: dict[str, object],
    candidate_report: dict[str, object],
) -> dict[str, object]:
    gate = protocol.get("checkpoint_selection_gate")
    if not isinstance(gate, dict):
        raise ValueError("Protocol has no checkpoint_selection_gate object.")
    required = {
        "baseline_protocol_name",
        "candidate",
        "baseline_checkpoint_filename",
        "candidate_checkpoint_filename",
        "changed_folds",
        "minimum_overall_q95_gain",
        "require_positive_float",
        "minimum_changed_fold_q95_gain",
        "minimum_positive_changed_rois",
        "minimum_changed_roi_q95_gain",
        "maximum_unchanged_fold_abs_delta",
    }
    if set(gate) != required:
        raise ValueError(
            "checkpoint_selection_gate fields mismatch: "
            f"missing={sorted(required - set(gate))}, "
            f"unexpected={sorted(set(gate) - required)}."
        )
    if type(gate["require_positive_float"]) is not bool:
        raise ValueError("require_positive_float must be boolean.")
    for field in (
        "minimum_overall_q95_gain",
        "minimum_changed_fold_q95_gain",
        "minimum_changed_roi_q95_gain",
        "maximum_unchanged_fold_abs_delta",
    ):
        if isinstance(gate[field], bool):
            raise ValueError(f"{field} must be numeric.")
        try:
            float(gate[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be numeric.") from error
    for field in (
        "minimum_overall_q95_gain",
        "minimum_changed_fold_q95_gain",
        "maximum_unchanged_fold_abs_delta",
    ):
        if float(gate[field]) < 0:
            raise ValueError(f"{field} must be non-negative.")
    minimum_positive_rois = gate["minimum_positive_changed_rois"]
    if type(minimum_positive_rois) is not int or minimum_positive_rois < 1:
        raise ValueError("minimum_positive_changed_rois must be a positive integer.")

    if baseline.get("protocol_name") != gate["baseline_protocol_name"]:
        raise ValueError("Baseline report protocol name does not match the gate.")
    if candidate_report.get("protocol_name") != protocol["name"]:
        raise ValueError("Candidate report protocol name does not match the protocol.")
    for field in ("marker", "validation_images", "validation_rois"):
        if baseline.get(field) != candidate_report.get(field):
            raise ValueError(f"Baseline and candidate report disagree on {field}.")
    if report_checkpoint_filename(baseline) != gate["baseline_checkpoint_filename"]:
        raise ValueError("Baseline report uses the wrong checkpoint filename.")
    if (
        report_checkpoint_filename(candidate_report)
        != gate["candidate_checkpoint_filename"]
    ):
        raise ValueError("Candidate report uses the wrong checkpoint filename.")

    candidate = str(gate["candidate"])
    quality = int(protocol["inference"]["jpeg_quality"])
    q_variant = f"jpeg_q{quality}"
    overall_q95_gain = candidate_metric(
        candidate_report, "overall", candidate, q_variant
    ) - candidate_metric(baseline, "overall", candidate, q_variant)
    overall_float_gain = candidate_metric(
        candidate_report, "overall", candidate, "float"
    ) - candidate_metric(baseline, "overall", candidate, "float")

    folds = {str(fold["name"]): fold for fold in protocol["folds"]}
    changed_folds_raw = gate["changed_folds"]
    if (
        not isinstance(changed_folds_raw, list)
        or not changed_folds_raw
        or any(not isinstance(name, str) for name in changed_folds_raw)
        or any(name not in folds for name in changed_folds_raw)
        or len(set(changed_folds_raw)) != len(changed_folds_raw)
    ):
        raise ValueError("checkpoint_selection_gate.changed_folds is invalid.")
    changed_folds = set(changed_folds_raw)
    fold_q95_deltas = {
        name: nested_metric(
            candidate_report, "fold_metrics", name, candidate, q_variant
        )
        - nested_metric(baseline, "fold_metrics", name, candidate, q_variant)
        for name in folds
    }
    changed_fold_deltas = {
        name: value for name, value in fold_q95_deltas.items() if name in changed_folds
    }
    unchanged_fold_deltas = {
        name: value
        for name, value in fold_q95_deltas.items()
        if name not in changed_folds
    }

    changed_rois = sorted(
        {roi for name in changed_folds for roi in folds[name]["val_rois"]}
    )
    if minimum_positive_rois > len(changed_rois):
        raise ValueError("minimum_positive_changed_rois exceeds the changed ROI count.")
    roi_q95_deltas = {
        roi: roi_metric(candidate_report, candidate, q_variant, roi)
        - roi_metric(baseline, candidate, q_variant, roi)
        for roi in changed_rois
    }
    positive_rois = sorted(roi for roi, value in roi_q95_deltas.items() if value > 0)
    minimum_roi_gain = min(roi_q95_deltas.values())
    maximum_unchanged_fold_delta = max(
        (abs(value) for value in unchanged_fold_deltas.values()),
        default=0.0,
    )

    checks = {
        "overall_q95": overall_q95_gain >= float(gate["minimum_overall_q95_gain"]),
        "overall_float": (
            overall_float_gain > 0 if gate["require_positive_float"] else True
        ),
        "changed_folds": all(
            value >= float(gate["minimum_changed_fold_q95_gain"])
            for value in changed_fold_deltas.values()
        ),
        "positive_changed_rois": len(positive_rois)
        >= int(gate["minimum_positive_changed_rois"]),
        "changed_roi_floor": minimum_roi_gain
        >= float(gate["minimum_changed_roi_q95_gain"]),
        "unchanged_folds_exact": maximum_unchanged_fold_delta
        <= float(gate["maximum_unchanged_fold_abs_delta"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate": candidate,
        "q95_variant": q_variant,
        "baseline_checkpoint_filename": report_checkpoint_filename(baseline),
        "candidate_checkpoint_filename": report_checkpoint_filename(candidate_report),
        "overall_q95_gain": overall_q95_gain,
        "overall_float_gain": overall_float_gain,
        "fold_q95_deltas": fold_q95_deltas,
        "changed_folds": sorted(changed_folds),
        "changed_fold_q95_deltas": changed_fold_deltas,
        "maximum_unchanged_fold_abs_delta": maximum_unchanged_fold_delta,
        "changed_roi_q95_deltas": roi_q95_deltas,
        "positive_changed_rois": positive_rois,
        "minimum_changed_roi_q95_gain": minimum_roi_gain,
        "thresholds": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two CV reports under a preregistered checkpoint gate."
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    baseline_path = args.baseline_report.resolve()
    candidate_path = args.candidate_report.resolve()
    protocol = load_protocol(protocol_path)
    report = {
        "version": 1,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": sha256_file(baseline_path),
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": sha256_file(candidate_path),
        "promotion_gate": compare_reports(
            protocol,
            load_json(baseline_path),
            load_json(candidate_path),
        ),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
