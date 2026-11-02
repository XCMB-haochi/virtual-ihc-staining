from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from cv_protocol import (
    fcmae_pretraining_directory,
    load_protocol,
    run_directory,
    sourced_run_directory,
)
from run_cv_training import build_train_command, completed_run

ROOT = Path(__file__).resolve().parents[1]
FCMAE_ENCODER_SOURCES = {
    "fold0": ("runs/round52_cv2_unext2_official_fcmae_fold0_dapi_officialfcmae800_pretraining/final_encoder.pt",
               "50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9"),
    "fold1": ("runs/round52_cv2_unext2_official_fcmae_fold1_dapi_officialfcmae800_pretraining/final_encoder.pt",
               "55f821b76fbc7d257a2aebc02ad41736a3e130f08af6607fa6f698ac6030aaf6"),
    "fold2": ("runs/round52_cv2_unext2_official_fcmae_fold2_dapi_officialfcmae800_pretraining/final_encoder.pt",
               "c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compute_busy_gpus() -> set[int]:
    mapping = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    indices = {uuid.strip(): int(index.strip()) for index, uuid in
               (line.split(",", 1) for line in mapping.splitlines() if line.strip())}
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True)
    uuids = {line.strip() for line in processes.splitlines() if line.strip()}
    if uuids - set(indices):
        return set(indices.values())  # Unknown/MIG mapping: do not assume idle.
    return {indices[uuid] for uuid in uuids}


def eligible_gpus(required_mib: int, excluded=(), candidates=None, require_empty=False) -> list[int]:
    readings = {}
    busy = set()
    for sample in range(5):
        if require_empty:
            busy.update(compute_busy_gpus())
        text = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        for line in text.splitlines():
            index, util, free = (int(value.strip()) for value in line.split(","))
            previous = readings.get(index, (0, free))
            readings[index] = (max(previous[0], util), min(previous[1], free))
        if sample < 4:
            time.sleep(1)
    return sorted(
        (gpu for gpu, (util, free) in readings.items()
         if gpu not in excluded and gpu not in busy and (candidates is None or gpu in candidates)
         and util <= 30 and free >= required_mib),
        key=lambda gpu: (readings[gpu][0], -readings[gpu][1]),
    )


def gpu_environment(gpu: int) -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def wait_for_gpu(required_mib: int, candidates=None, require_empty=False) -> int:
    while True:
        available = eligible_gpus(required_mib, candidates=candidates, require_empty=require_empty)
        if available:
            return available[0]
        time.sleep(30)


def verified_stage_complete(path: Path, epochs: int) -> bool:
    if not completed_run(path, epochs) or (path/"best_global_mse.pt").stat().st_size == 0:
        return False
    rows = [json.loads(line) for line in (path/"metrics.jsonl").read_text().splitlines() if line]
    return [row.get("epoch") for row in rows] == list(range(1, epochs+1))


def pending_folds(protocol, recipe_name):
    recipe = protocol["recipes"][recipe_name]
    epochs = int(recipe.get("epochs_per_stage", protocol["training"]["epochs_per_stage"]))
    return [fold["name"] for fold in protocol["folds"] if not all(
        verified_stage_complete(run_directory(ROOT/"runs", protocol["name"], fold["name"], recipe_name, stage), epochs)
        for stage in recipe["train_stages"]
    )]


def fcmae_source(recipe: dict[str, object], fold_name: str) -> tuple[Path, str]:
    source = recipe.get("fcmae_encoder_source")
    if source is None:
        source_name, source_hash = FCMAE_ENCODER_SOURCES[fold_name]
    else:
        if set(source) != {"checkpoint", "sha256"}:
            raise ValueError("fcmae_encoder_source must contain checkpoint and sha256.")
        source_name, source_hash = str(source["checkpoint"]), str(source["sha256"])
    path = ROOT / source_name
    if not path.is_file() or digest(path) != source_hash:
        raise FileNotFoundError(f"Missing or invalid FCMAE source: {path}")
    return path, source_hash


def preflight(protocol, label, allow_completed=False):
    trained = [(name, recipe) for name, recipe in protocol["recipes"].items() if recipe.get("train_stages")]
    if len(trained) != 1 or trained[0][1]["train_stages"] != [1, 2]:
        raise ValueError("This launcher supports one fresh two-stage candidate.")
    if not re.fullmatch(r"round[0-9]+", label) or not protocol["name"].startswith(label + "_"):
        raise ValueError("Run label and protocol name do not match.")
    for fold in protocol["folds"]:
        for name, recipe in protocol["recipes"].items():
            if recipe.get("train_stages"):
                if recipe.get("fcmae_encoder_init", False):
                    source, source_hash = fcmae_source(recipe, str(fold["name"]))
                    linked = (
                        ROOT / fcmae_pretraining_directory(ROOT / "runs", protocol, str(fold["name"]))
                        / "final_encoder.pt"
                    )
                    linked.parent.mkdir(parents=True, exist_ok=True)
                    if linked.exists():
                        if digest(linked) != source_hash:
                            raise ValueError(f"Existing FCMAE source link differs: {linked}")
                    else:
                        linked.hardlink_to(source)
                for stage in recipe["train_stages"]:
                    path = run_directory(ROOT/"runs", protocol["name"], fold["name"], name, stage)
                    if path.exists():
                        epochs = int(recipe.get("epochs_per_stage", protocol["training"]["epochs_per_stage"]))
                        if not allow_completed or not verified_stage_complete(path, epochs):
                            raise FileExistsError(f"Training output already exists or is incomplete: {path}")
            else:
                path = sourced_run_directory(ROOT/"runs", protocol, fold["name"], name) / protocol["inference"]["checkpoint_filename"]
                if not path.is_file():
                    raise FileNotFoundError(path)
    for suffix in ("oof", "gate"):
        path = ROOT / "reports" / f"{protocol['name']}_{suffix}.json"
        if path.exists():
            raise FileExistsError(path)
    return trained[0]


def smoke(args, protocol, recipe_name, recipe):
    gpu = wait_for_gpu(args.min_train_free_mib, args.gpus, args.wait_for_empty_gpu)
    results = []
    previous = None
    for stage in (1, 2):
        directory = ROOT / "runs" / f"{args.label}_smoke_stage{stage}"
        if directory.exists():
            raise FileExistsError(directory)
        command = build_train_command(sys.executable, args.dataset, directory,
            protocol["folds"][0]["val_rois"], protocol["marker"], recipe,
            protocol["training"], stage, previous, "cuda:0", fold_name=protocol["folds"][0]["name"])
        if stage == 1 and recipe.get("fcmae_encoder_init", False):
            command.extend(("--fcmae-encoder-init-checkpoint", str(fcmae_source(recipe, str(protocol["folds"][0]["name"]))[0].resolve())))
        command[command.index("--epochs")+1] = "1"
        command.extend(["--max-train-images", "64", "--max-val-images", "32"])
        with (ROOT/"reports"/f"{args.label}_smoke_stage{stage}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=gpu_environment(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
        config = read_json(directory/"config.json")
        if len(rows) != 1 or rows[0]["epoch"] != 1 or rows[0]["optimizer_steps"] != 4:
            raise RuntimeError("Smoke optimizer/epoch contract failed.")
        previous = directory / "best_global_mse.pt"
        if not previous.is_file():
            raise FileNotFoundError(previous)
        if not all(math.isfinite(row["train_loss"]) for row in rows):
            raise RuntimeError("Non-finite smoke loss.")
        if recipe.get("primary_gradient_projection", False):
            import torch
            checkpoint = torch.load(previous, map_location="cpu", weights_only=False)
            stats = rows[0]["primary_gradient_projection"]
            if not config.get("primary_gradient_projection") or stats["finite_updates"] < 1:
                raise RuntimeError("Projection smoke did not perform a finite update.")
            if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
                raise RuntimeError("Projection smoke checkpoint is non-finite.")
            results_stats = {"projection": stats}
        else:
            results_stats = {}
        if recipe.get("feature_guidance") is not None:
            import torch
            checkpoint = torch.load(previous, map_location="cpu", weights_only=False)
            feature_loss = rows[0]["train_feature_guidance_loss"]
            if not math.isfinite(feature_loss) or feature_loss <= 0:
                raise RuntimeError("Feature-guidance smoke did not produce a finite positive loss.")
            if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
                raise RuntimeError("Feature-guidance smoke checkpoint is non-finite.")
            results_stats["feature_guidance"] = {
                "loss": feature_loss, "teacher": config["feature_guidance_report"]}
        results.append({"stage":stage,"seconds":rows[0]["seconds"],"peak_gib":rows[0]["peak_gpu_gib"],
                        "init":config["init_checkpoint_report"],"parameters":config["parameters"], **results_stats})
    receipt = {"protocol_sha256":digest(args.protocol),"stages":results}
    (ROOT/"runs"/f"{args.label}_smoke.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


def run(args, protocol, recipe_name):
    receipt = read_json(ROOT/"runs"/f"{args.label}_smoke.json")
    if receipt["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Smoke used a different protocol.")
    peak_mib = max(row["peak_gib"] for row in receipt["stages"]) * 1024
    if args.min_train_free_mib < peak_mib + 4096:
        raise ValueError("Training memory threshold needs at least 4 GiB above measured smoke peak.")
    pending = pending_folds(protocol, recipe_name)
    previous_launch_path = ROOT/"reports"/f"{args.label}_launch.json"
    if args.recover:
        previous = read_json(previous_launch_path)
        if previous["protocol_sha256"] != digest(args.protocol):
            raise ValueError("Recovery protocol differs from the original launch.")
        for name, checksum in previous["source_sha256"].items():
            if name != "tools/run_training_screen.py" and digest(ROOT/name) != checksum:
                raise ValueError(f"Recovery training source differs: {name}")
    launch = {"protocol":protocol["name"],"source_commit":args.source_commit,
              "protocol_sha256":digest(args.protocol), "smoke":receipt,
              "gpu_candidates":args.gpus, "pending_folds":pending,
              "wait_for_empty_gpu": args.wait_for_empty_gpu,
              "source_sha256": {name:digest(ROOT/name) for name in (
                  "src/ihc_dinov2.py", "src/ihc_dino_residual.py", "src/ihc_dino_fusion.py", "src/ihc_torch.py", "tools/train_unet.py",
                  "tools/cv_protocol.py", "tools/run_cv_training.py", "tools/run_training_screen.py",
                  "tools/evaluate_cv_ensemble.py", "tools/evaluate_distribution_gate.py",
                  "src/ihc_gradient_projection.py", "src/ihc_spatial.py",
                  "src/ihc_unext2.py", "src/ihc_feature_guidance.py")}}
    if args.recover:
        launch["previous_launch_sha256"] = digest(previous_launch_path)
        launch_path = ROOT/"reports"/f"{args.label}_recovery_launch.json"
        with launch_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(launch, indent=2)+"\n")
    else:
        previous_launch_path.write_text(json.dumps(launch,indent=2)+"\n")
    for fold in protocol["folds"]:
        if fold["name"] not in pending:
            print(f"Reusing completed {fold['name']}", flush=True)
    active = {}
    failed = False
    with contextlib.ExitStack() as stack:
        while pending or active:
            for gpu, (fold, process) in list(active.items()):
                code = process.poll()
                if code is not None:
                    print(f"Completed {fold}: exit={code}", flush=True)
                    del active[gpu]
                    if code != 0:
                        failed = True
                        pending.clear()
            if pending:
                for gpu in eligible_gpus(args.min_train_free_mib, active, args.gpus, args.wait_for_empty_gpu):
                    if not pending:
                        break
                    fold = pending.pop(0)
                    log = stack.enter_context((ROOT/"reports"/f"{args.label}_{fold}.log").open("w"))
                    command = [sys.executable, "tools/run_cv_training.py", str(args.dataset), str(args.protocol),
                               "--folds", fold, "--recipes", recipe_name, "--device", "cuda:0"]
                    process = subprocess.Popen(command, cwd=ROOT, env=gpu_environment(gpu), stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (fold, process)
                    print(f"Launched {fold} on GPU{gpu}: pid={process.pid}", flush=True)
            if pending or active:
                time.sleep(15)
    if failed:
        raise RuntimeError("A training fold failed; OOF was not started.")
    gpu = wait_for_gpu(12000, args.gpus, args.wait_for_empty_gpu)
    oof_path = ROOT/"reports"/f"{protocol['name']}_oof.json"
    gate_path = ROOT/"reports"/f"{protocol['name']}_gate.json"
    subprocess.run([sys.executable, "tools/evaluate_cv_ensemble.py", str(args.dataset), str(args.protocol),
                    "--device", "cuda:0", "--output", str(oof_path)], cwd=ROOT, env=gpu_environment(gpu), check=True)
    subprocess.run([sys.executable, "tools/evaluate_distribution_gate.py", str(args.protocol), str(oof_path),
                    "--output", str(gate_path)], cwd=ROOT, check=True)
    oof, gate = read_json(oof_path), read_json(gate_path)
    if set(oof["validation_rois"]) != set(protocol["expected_rois"]) or gate["oof_report_sha256"] != digest(oof_path):
        raise RuntimeError("OOF coverage or gate provenance mismatch.")
    print("COMPLETE", protocol["name"], gate["promotion"], flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run a preflighted two-stage candidate with resource-aware fold scheduling.")
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--min-train-free-mib", type=int, default=24000)
    parser.add_argument("--source-commit")
    parser.add_argument("--recover", action="store_true", help="Reuse completed stages of the unchanged protocol; incomplete outputs must be archived first.")
    parser.add_argument("--gpus", type=int, nargs="+", help="Physical GPU candidates for this invocation; default checks all cards.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--queue", action="store_true", help="Queue smoke, then formal training only after smoke succeeds.")
    parser.add_argument("--wait-for-empty-gpu", action="store_true", help="Wait for existing compute processes to exit, even at low utilization.")
    args = parser.parse_args()
    if args.queue and (args.smoke or args.preflight or args.recover or args.gpus is None or len(args.gpus) != 1):
        parser.error("Queue mode requires one explicit GPU and a fresh run.")
    if args.gpus is not None and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
        parser.error("--gpus requires unique nonnegative physical GPU indices.")
    args.protocol = args.protocol.resolve()
    args.dataset = args.dataset.resolve()
    protocol = load_protocol(args.protocol)
    recipe_name, recipe = preflight(protocol, args.label, allow_completed=args.recover)
    if args.preflight:
        print("PREFLIGHT_PASS", protocol["name"])
        return
    if args.smoke:
        smoke(args, protocol, recipe_name, recipe)
        return
    if args.source_commit is None or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("Formal launch requires the synchronized local source commit.")
    import fcntl
    with (ROOT/"runs"/f"{args.label}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = ROOT/"reports"/f"{args.label}_pipeline.exit"
        status.unlink(missing_ok=True)
        code = 1
        try:
            if args.queue:
                queue_path = ROOT/"reports"/f"{args.label}_queue.json"
                queued = {"state": "waiting_for_smoke", "protocol": protocol["name"],
                          "protocol_sha256": digest(args.protocol), "source_commit": args.source_commit,
                          "gpu_candidates": args.gpus, "wait_for_empty_gpu": args.wait_for_empty_gpu}
                with queue_path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(queued, indent=2)+"\n")
                print("QUEUED", json.dumps(queued), flush=True)
                smoke(args, protocol, recipe_name, recipe)
                queued["state"] = "smoke_passed"
                queue_path.write_text(json.dumps(queued, indent=2)+"\n")
            run(args, protocol, recipe_name)
            if args.queue:
                queued["state"] = "completed"
                queue_path.write_text(json.dumps(queued, indent=2)+"\n")
            code = 0
        finally:
            status.write_text(str(code)+"\n")


if __name__ == "__main__":
    main()
