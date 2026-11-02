from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_data import image_names, load_gray, parse_patch_name, resolve_data_root
from ihc_unext2 import DapiFCMAE, unext2_encoder_specification


EXPECTED_PRETRAINING_MARKERS = ["DAPI", "CD68", "CD45RO", "HLA-DR", "Vimentin"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PairedModalities(Dataset):
    def __init__(self, data_root: Path, names: list[str], marker: str) -> None:
        self.data_root = data_root
        self.names = names
        self.marker = marker

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]
        dapi = load_gray(self.data_root / "train" / "DAPI" / name)
        target = load_gray(self.data_root / "train" / self.marker / name)
        return (
            torch.from_numpy(dapi[None] / 255.0).float(),
            torch.from_numpy(target[None] / 255.0).float(),
            name,
        )


def load_encoder(
    checkpoint_path: Path,
    expected_sha256: str,
    expected_val_rois: set[str],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Encoder SHA-256 mismatch for {checkpoint_path}: {actual_sha256}."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("FCMAE encoder checkpoint must be a mapping.")
    if checkpoint.get("version") != 1 or checkpoint.get("task") != "dapi-fcmae-pretraining":
        raise ValueError("Unexpected FCMAE encoder checkpoint format.")
    if checkpoint.get("encoder_spec") != unext2_encoder_specification("tiny"):
        raise ValueError("Round109 requires a Tiny UNeXt2 encoder.")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("FCMAE encoder checkpoint lacks configuration metadata.")
    if list(config.get("pretraining_markers", ())) != EXPECTED_PRETRAINING_MARKERS:
        raise ValueError("Round109 requires the fixed Round96 five-modality encoder.")
    checkpoint_val_rois = set(checkpoint.get("val_rois", ()))
    if checkpoint_val_rois != expected_val_rois:
        raise ValueError(
            "Encoder validation exclusion does not match the diagnostic fold: "
            f"{sorted(checkpoint_val_rois)} != {sorted(expected_val_rois)}."
        )
    model = DapiFCMAE("tiny").encoder
    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping):
        raise ValueError("FCMAE checkpoint lacks encoder parameters.")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    provenance = {
        "path": str(checkpoint_path.resolve()),
        "sha256": actual_sha256,
        "epoch": int(checkpoint["epoch"]),
        "pretraining_markers": list(config["pretraining_markers"]),
        "excluded_val_rois": sorted(checkpoint_val_rois),
    }
    return model, provenance


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dapi_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    names: list[str] = []
    use_amp = device.type == "cuda"
    for dapi, target, batch_names in loader:
        dapi = dapi.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            dapi_features, _ = encoder(dapi)
            target_features, _ = encoder(target)
        dapi_embedding = dapi_features[-1].float().mean(dim=(-2, -1))
        target_embedding = target_features[-1].float().mean(dim=(-2, -1))
        dapi_parts.append(dapi_embedding.cpu().numpy())
        target_parts.append(target_embedding.cpu().numpy())
        names.extend(batch_names)
    return np.concatenate(dapi_parts), np.concatenate(target_parts), names


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Cosine inputs must be equal two-dimensional arrays.")
    epsilon = np.finfo(np.float64).eps
    left_norm = np.linalg.norm(left, axis=1).clip(min=epsilon)
    right_norm = np.linalg.norm(right, axis=1).clip(min=epsilon)
    return np.sum(left * right, axis=1) / (left_norm * right_norm)


def deterministic_mismatch_indices(names: list[str]) -> np.ndarray:
    if len(names) < 2:
        raise ValueError("Mismatch comparison requires at least two images.")
    rois = np.asarray([parse_patch_name(name).roi for name in names])
    indices = np.arange(len(names))
    output = np.empty(len(names), dtype=np.int64)
    for roi in sorted(set(rois)):
        selected = indices[rois == roi]
        if len(selected) < 2:
            raise ValueError(f"ROI {roi} has too few images for mismatch comparison.")
        output[selected] = np.roll(selected, 1)
    return output


def ridge_feature_report(
    train_dapi: np.ndarray,
    train_target: np.ndarray,
    val_dapi: np.ndarray,
    val_target: np.ndarray,
    val_names: list[str],
    alpha: float,
    solver: str,
) -> dict[str, float]:
    x_scaler = StandardScaler().fit(train_dapi)
    y_scaler = StandardScaler().fit(train_target)
    x_train = x_scaler.transform(train_dapi)
    y_train = y_scaler.transform(train_target)
    x_val = x_scaler.transform(val_dapi)
    y_val = y_scaler.transform(val_target)
    model = Ridge(alpha=alpha, solver=solver, fit_intercept=False)
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    residual_sum = float(np.square(prediction - y_val).sum(dtype=np.float64))
    baseline_sum = float(np.square(y_val).sum(dtype=np.float64))
    r2 = 1.0 - residual_sum / baseline_sum
    paired_cosine = float(cosine_rows(val_dapi, val_target).mean())
    mismatched = deterministic_mismatch_indices(val_names)
    mismatched_cosine = float(cosine_rows(val_dapi, val_target[mismatched]).mean())
    return {
        "standardized_r2": r2,
        "standardized_residual_sum": residual_sum,
        "standardized_baseline_sum": baseline_sum,
        "mapped_centered_cosine": float(cosine_rows(prediction, y_val).mean()),
        "paired_cosine": paired_cosine,
        "within_roi_mismatched_cosine": mismatched_cosine,
        "paired_cosine_advantage": paired_cosine - mismatched_cosine,
    }


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("version") != 1:
        raise ValueError("Round109 protocol must be a version-1 object.")
    if protocol.get("name") != "round109_cross_modal_feature_diagnostic":
        raise ValueError("Unexpected Round109 protocol name.")
    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != 2:
        raise ValueError("Round109 requires exactly two folds.")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fold-safe DAPI-to-CD68 FCMAE feature predictability."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    data_root = resolve_data_root(args.dataset)
    all_names = image_names(data_root / "train" / "DAPI")
    marker = str(protocol["marker"])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    embedding = protocol["embedding"]
    mapping = protocol["mapping"]
    if not isinstance(embedding, Mapping) or not isinstance(mapping, Mapping):
        raise ValueError("Round109 embedding and mapping sections must be objects.")

    started = time.perf_counter()
    folds_report: dict[str, object] = {}
    total_residual = 0.0
    total_baseline = 0.0
    for fold in protocol["folds"]:
        if not isinstance(fold, Mapping):
            raise ValueError("Round109 fold entries must be objects.")
        fold_name = str(fold["name"])
        val_rois = set(fold["val_rois"])
        train_names = [
            name for name in all_names if parse_patch_name(name).roi not in val_rois
        ]
        val_names = [
            name for name in all_names if parse_patch_name(name).roi in val_rois
        ]
        if args.max_train_images > 0:
            train_names = train_names[: args.max_train_images]
        if args.max_val_images > 0:
            val_names = val_names[: args.max_val_images]
        checkpoint_path = Path(str(fold["encoder"]))
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        encoder, provenance = load_encoder(
            checkpoint_path,
            str(fold["encoder_sha256"]),
            val_rois,
            device,
        )
        train_loader = DataLoader(
            PairedModalities(data_root, train_names, marker),
            batch_size=int(embedding["batch_size"]),
            shuffle=False,
            num_workers=int(embedding["workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=int(embedding["workers"]) > 0,
        )
        val_loader = DataLoader(
            PairedModalities(data_root, val_names, marker),
            batch_size=int(embedding["batch_size"]),
            shuffle=False,
            num_workers=int(embedding["workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=int(embedding["workers"]) > 0,
        )
        train_dapi, train_target, extracted_train_names = extract_embeddings(
            encoder, train_loader, device
        )
        val_dapi, val_target, extracted_val_names = extract_embeddings(
            encoder, val_loader, device
        )
        if extracted_train_names != train_names or extracted_val_names != val_names:
            raise RuntimeError("Round109 embedding order changed during extraction.")
        if train_dapi.shape[1] != int(embedding["dimension"]):
            raise RuntimeError("Round109 embedding dimension does not match protocol.")
        metrics = ridge_feature_report(
            train_dapi,
            train_target,
            val_dapi,
            val_target,
            val_names,
            float(mapping["alpha"]),
            str(mapping["solver"]),
        )
        total_residual += metrics["standardized_residual_sum"]
        total_baseline += metrics["standardized_baseline_sum"]
        folds_report[fold_name] = {
            "train_images": len(train_names),
            "validation_images": len(val_names),
            "train_rois": sorted(
                {parse_patch_name(name).roi for name in train_names}
            ),
            "validation_rois": sorted(val_rois),
            "encoder": provenance,
            "metrics": metrics,
        }
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    weighted_r2 = 1.0 - total_residual / total_baseline
    gate = protocol["gate"]
    if not isinstance(gate, Mapping):
        raise ValueError("Round109 gate must be an object.")
    fold_r2 = {
        name: float(report["metrics"]["standardized_r2"])
        for name, report in folds_report.items()
    }
    fold_advantage = {
        name: float(report["metrics"]["paired_cosine_advantage"])
        for name, report in folds_report.items()
    }
    checks = {
        "weighted_r2": weighted_r2 >= float(gate["minimum_weighted_r2"]),
        "fold_r2": min(fold_r2.values()) >= float(gate["minimum_fold_r2"]),
        "paired_cosine_advantage": min(fold_advantage.values())
        >= float(gate["minimum_fold_paired_cosine_advantage"]),
    }
    report = {
        "version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_name": protocol["name"],
        "formal": args.max_train_images == 0 and args.max_val_images == 0,
        "data_root": str(data_root),
        "marker": marker,
        "embedding": dict(embedding),
        "mapping": dict(mapping),
        "folds": folds_report,
        "overall": {
            "weighted_standardized_r2": weighted_r2,
            "fold_r2": fold_r2,
            "fold_paired_cosine_advantage": fold_advantage,
        },
        "gate": {
            "thresholds": dict(gate),
            "checks": checks,
            "feature_guidance_actionable": bool(all(checks.values())),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["overall"], ensure_ascii=False), flush=True)
    print(json.dumps(report["gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
