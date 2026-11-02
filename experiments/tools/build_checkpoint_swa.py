from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from average_checkpoints import average_checkpoints


def selected_epochs(metrics_path: Path) -> tuple[int, int, int, int]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(row["epoch"]) for row in rows] != list(range(1, 15)):
        raise ValueError(f"Expected source epochs 1..14 in {metrics_path}")
    best = max(rows, key=lambda row: float(row["selection_score"]))
    center = int(best["epoch"])
    if not 2 <= center <= 13:
        raise ValueError(f"Best epoch must be interior for centered soup, got {center}")
    return center - 1, center, center + 1, center


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one fixed 1:2:1 local checkpoint soup.")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    first, center, last, _ = selected_epochs(source_dir / "metrics.jsonl")
    inputs = [source_dir / "epochs" / f"epoch_{epoch:03d}.pt" for epoch in (first, center, last)]
    if any(not path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise FileNotFoundError(f"Missing source checkpoints: {missing}")
    output = output_dir / "best_global_mse.pt"
    result = average_checkpoints(
        inputs,
        [1.0, 2.0, 1.0],
        output,
        buffer_policy="error",
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    soup = checkpoint.get("soup")
    if not isinstance(soup, dict):
        raise RuntimeError("Averaged checkpoint has no soup metadata")
    payload = {
        "version": 1,
        "source_dir": str(source_dir),
        "source_best_epoch": center,
        "input_epochs": [first, center, last],
        "input_weights": [0.25, 0.5, 0.25],
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "inputs": soup.get("inputs", []),
        "average_result": result,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
