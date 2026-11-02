"""Stage-wide grouped validation clock for a CD45RO-primary baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATASET = "dataset/复赛数据集(包括训练集和测试集输入)"
ENCODER = "runs/semifinal_public_fcmae_tiny/final_encoder.pt"
ENCODER_SHA = "98256dc1aca47cb6e553a1c82d7f343d328e14d2c7a1b1019dc34273669d1578"


def training_command(fold, stage, stage1_epoch=None, smoke=False):
    output = f"runs/round204_fold{fold['name'][-1]}_stage{stage}" if not smoke else "runs/round204_smoke"
    command = [sys.executable, "tools/train_unet.py", DATASET,
        "--output-dir", output, "--val-rois", ",".join(fold["val_rois"]),
        "--markers", "CD68,CD45RO,HLA-DR,Vimentin",
        "--marker-loss-weights", "0.05,0.85,0.05,0.05",
        "--epochs", "1" if smoke else "14", "--batch-size", "4",
        "--gradient-accumulation-steps", "4", "--model-arch", "unext2-fcmae-tiny",
        "--base-channels", "32", "--block-type", "residual",
        "--condition-mode", "spatial-thumbnail-film", "--context-halo", "64",
        "--train-target-context-halo", "64", "--learning-rate", "0.0003" if stage == 1 else "0.0001",
        "--minimum-learning-rate", "0", "--weight-decay", "0.0001",
        "--ssim-weight", "0.7", "--l1-weight", "0.2", "--mse-weight", "0.1",
        "--loss-profile", "hybrid", "--ssim-mode", "uniform7",
        "--selection-marker", "CD45RO", "--selection-ssim-mode", "gaussian11",
        "--selection-psnr-aggregation", "mean-image", "--selection-jpeg-quality", "95",
        "--workers", "0", "--seed", "2026", "--save-every-epoch", "--device", "cuda:0"]
    if stage == 1:
        command += ["--fcmae-encoder-init-checkpoint", ENCODER]
    else:
        if stage1_epoch is None:
            raise ValueError("A common stage1 clock is required")
        command += ["--init-checkpoint", f"runs/round204_{fold['name']}_stage1/epochs/epoch_{stage1_epoch:03d}.pt"]
    if smoke:
        command += ["--max-train-images", "128", "--max-val-images", "8"]
    return command, Path(output)


def common_clock(folds, stage):
    trajectories = {}
    for fold in folds:
        path = Path(f"runs/round204_{fold['name']}_stage{stage}/metrics.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["epoch"] for r in rows] != list(range(1, 15)):
            raise ValueError(f"Incomplete trajectory: {path}")
        if any(r["validation"]["ssim_mode"] != "gaussian11" for r in rows):
            raise ValueError("Wrong selection metric")
        trajectories[fold["name"]] = rows
    curve = []
    for index in range(14):
        numerator = denominator = 0
        scores = {}
        for name, rows in trajectories.items():
            val = rows[index]["validation"]
            score = val["markers"]["CD45RO"]["competition_score_mean_image_psnr"]
            numerator += score * val["count"]
            denominator += val["count"]
            scores[name] = score
        if denominator != 2380:
            raise ValueError("Incomplete grouped validation")
        curve.append({"epoch": index + 1, "score": numerator / denominator, "fold_scores": scores})
    epoch = max(curve, key=lambda row: row["score"])["epoch"]
    return {"stage": stage, "selected_epoch": epoch, "curve": curve,
            "rule": "Maximum image-weighted mean of all folds at the same epoch; earliest exact tie. No fold-best median."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with Path(ENCODER).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != ENCODER_SHA:
            raise ValueError("Initialization artifact changed")
    folds = json.loads(Path("configs/semifinal_cv5_baseline.json").read_text())["folds"]
    if args.smoke:
        command, output = training_command(folds[0], 1, smoke=True)
        if output.exists():
            raise FileExistsError(output)
        subprocess.run(command, check=True)
        import torch
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        steps = [float(state["step"]) for state in checkpoint["optimizer"]["state"].values() if "step" in state]
        if not steps or min(steps) <= 0:
            raise RuntimeError("Smoke did not perform actual optimizer updates")
        print(f"Verified optimizer updates: {min(steps)}", flush=True)
        return
    for fold in folds:
        for stage in (1, 2):
            if Path(f"runs/round204_{fold['name']}_stage{stage}").exists():
                raise FileExistsError("Refusing to overwrite a training trajectory")
    first_clock = None
    for stage in (1, 2):
        for fold in folds:
            command, _ = training_command(fold, stage, first_clock)
            print(f"Starting stage{stage} {fold['name']}", flush=True)
            with Path(f"reports/round204_{fold['name']}_stage{stage}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        clock = common_clock(folds, stage)
        Path(f"reports/round204_stage{stage}_clock.json").write_text(json.dumps(clock, indent=2) + "\n")
        print(json.dumps(clock), flush=True)
        if stage == 1:
            first_clock = clock["selected_epoch"]


if __name__ == "__main__":
    main()
