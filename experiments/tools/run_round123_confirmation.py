"""Train a fold-excluded teacher, then confirm the unchanged Round122 recipe."""
from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from cv_protocol import load_protocol, validate_protocol
from run_dapi_fcmae_pretraining import build_command, completed_pretraining
from run_training_screen import digest, fcmae_source, gpu_environment, preflight, smoke, wait_for_gpu

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/round123_fold1_relational_confirmation.json"
TEACHER_PLAN = ROOT / "configs/round123_teacher.json"
RECIPE = "encoderrelational15"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def validate_frozen_recipe(template, original, plan, original_pretraining):
    for key in ("training", "inference", "fcmae_pretraining", "candidates", "candidate_output_source_offsets"):
        if template[key] != original[key]:
            raise ValueError(f"Confirmation changed {key}.")
    recipes = copy.deepcopy(template["recipes"])
    recipes[RECIPE]["feature_guidance"]["teachers"] = original["recipes"][RECIPE]["feature_guidance"]["teachers"]
    if recipes != original["recipes"]:
        raise ValueError("Confirmation changed the trained recipe.")
    if not template.get("confirmation_only") or set(template["expected_rois"]) & set(original["expected_rois"]):
        raise ValueError("Confirmation must use additional held-out ROIs.")
    if plan["fold"] != template["folds"][0] or plan["pretraining"] != original_pretraining:
        raise ValueError("Teacher changed its fold or pretraining recipe.")


def materialize(template, checkpoint, checksum):
    result = copy.deepcopy(template)
    result["recipes"][RECIPE]["feature_guidance"]["teachers"] = {
        "fold1": {"checkpoint": str(checkpoint), "sha256": checksum}
    }
    return validate_protocol(result)


def audit_teacher(directory, plan, epochs):
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from ihc_feature_guidance import load_frozen_multimarker_fcmae_teacher
    if not completed_pretraining(directory, epochs):
        raise RuntimeError("Teacher is incomplete.")
    rows = [json.loads(row) for row in (directory / "metrics.jsonl").read_text().splitlines() if row]
    if [row["epoch"] for row in rows] != list(range(1, epochs + 1)) or not all(math.isfinite(row["loss"]) for row in rows):
        raise RuntimeError("Teacher trajectory is invalid.")
    path = directory / "final_encoder.pt"
    teacher, report = load_frozen_multimarker_fcmae_teacher(
        path, digest(path), set(plan["fold"]["val_rois"]), torch.device("cpu"))
    if not all(bool(torch.isfinite(p).all()) for p in teacher.parameters()):
        raise RuntimeError("Teacher weights are nonfinite.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(TEMPLATE)
    plan = read(TEACHER_PLAN)
    validate_frozen_recipe(protocol, load_protocol(ROOT / "configs/round122_cv2_encoder_relational_robust.json"),
                           plan, read(ROOT / "configs/round96_cv2_multimarker_fcmae.json")["fcmae_pretraining"])
    public = plan["pretraining"]["encoder_initialization"]
    if digest(ROOT / public["checkpoint"]) != public["sha256"]:
        raise ValueError("Public initialization hash differs.")
    if digest(ROOT / plan["baseline"]["checkpoint"]) != plan["baseline"]["sha256"]:
        raise ValueError("Matched fold1 baseline hash differs.")
    fcmae_source(protocol["recipes"][RECIPE], "fold1")
    preflight(protocol, "round123")
    directory = ROOT / "runs" / ("round123_teacher_smoke" if args.smoke else "round123_fold1_teacher")
    if directory.exists():
        raise FileExistsError(f"Refusing to reuse teacher directory without recovery audit: {directory}")
    if args.preflight:
        print("PREFLIGHT_PASS: frozen Round122, new fold1, fold-local teacher", flush=True)
        return
    epochs = 3 if args.smoke else 160  # Keep the registered two warmup epochs valid.
    settings = {**plan["pretraining"], "epochs": epochs}
    command = build_command(ROOT / "dataset", directory, plan["fold"]["val_rois"], settings, "cuda:0")
    if args.smoke:
        command.extend(["--max-train-images", "64"])
    log = ROOT / "reports" / ("round123_teacher_smoke.log" if args.smoke else "round123_teacher.log")
    if not args.smoke:
        if args.source_commit is None or len(args.source_commit) != 40:
            raise ValueError("Formal launch needs the exact source commit.")
        receipt = {
            "source_commit": args.source_commit, "gpu_candidates": [1],
            "protocol_template_sha256": digest(TEMPLATE), "teacher_plan_sha256": digest(TEACHER_PLAN),
            "source_sha256": {name: digest(ROOT / name) for name in (
                "src/ihc_unext2.py", "src/ihc_feature_guidance.py", "src/ihc_torch.py", "src/ihc_spatial.py",
                "tools/train_dapi_fcmae.py", "tools/train_unet.py", "tools/run_dapi_fcmae_pretraining.py",
                "tools/cv_protocol.py", "tools/run_cv_training.py", "tools/run_training_screen.py",
                "tools/run_round123_confirmation.py", "tools/evaluate_cv_ensemble.py")},
            "teacher_command": command,
        }
        write_new(ROOT / "reports/round123_orchestration.json", receipt)
    gpu = wait_for_gpu(36000, [1], True)
    print(f"TEACHER_LAUNCH GPU{gpu}: {epochs} epochs", flush=True)
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=gpu_environment(gpu), stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = audit_teacher(directory, plan, epochs)
    checkpoint = directory / "final_encoder.pt"
    resolved = materialize(protocol, str(checkpoint), digest(checkpoint))
    resolved_path = ROOT / "reports" / ("round123_smoke_protocol.json" if args.smoke else "round123_resolved_protocol.json")
    write_new(resolved_path, resolved)
    if args.smoke:
        smoke(SimpleNamespace(protocol=resolved_path, label="round123_preflight", min_train_free_mib=36000,
                              gpus=[1], wait_for_empty_gpu=True), resolved, RECIPE, resolved["recipes"][RECIPE])
        write_new(ROOT / "reports/round123_preflight.json", {"teacher": report, "template_sha256": digest(TEMPLATE)})
        print("PREFLIGHT_SMOKE_PASS", flush=True)
    else:
        write_new(ROOT / "reports/round123_teacher_audit.json", report)
        subprocess.run([sys.executable, "tools/run_training_screen.py", str(resolved_path), "--label", "round123",
                        "--source-commit", args.source_commit, "--gpus", "1", "--wait-for-empty-gpu",
                        "--min-train-free-mib", "36000", "--queue"], cwd=ROOT, check=True)


if __name__ == "__main__":
    # The external Screen command is not dependent on the interactive SSH session.
    formal = not ({"--preflight", "--smoke"} & set(sys.argv))
    code = 1
    try:
        main()
        code = 0
    finally:
        if formal:
            (ROOT / "reports/round123_pipeline.exit").write_text(str(code) + "\n")
