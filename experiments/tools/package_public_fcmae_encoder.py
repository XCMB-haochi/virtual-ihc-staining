from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_unext2 import (
    DapiFCMAE,
    initialize_fcmae_encoder_from_public_checkpoint,
    unext2_encoder_specification,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("public_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite encoder: {args.output}")
    torch.manual_seed(args.seed)
    model = DapiFCMAE("tiny")
    report = initialize_fcmae_encoder_from_public_checkpoint(
        model.encoder,
        args.public_checkpoint,
        args.sha256,
        "official-convnextv2-fcmae-imagenet1k-tiny",
    )
    payload = {
        "version": 1,
        "task": "dapi-fcmae-pretraining",
        "epoch": 0,
        "val_rois": [],
        "encoder_spec": unext2_encoder_specification("tiny"),
        "encoder": model.encoder.state_dict(),
        "config": {
            "data_scope": "public-only",
            "seed": args.seed,
            "pretraining_markers": ["DAPI"],
            "epochs": 0,
            "encoder_initialization": report,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256(args.output), "initialization": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
