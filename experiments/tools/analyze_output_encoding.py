from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ihc_torch import competition_score


def decode_jpeg(image: np.ndarray, quality: int, *, optimize: bool = False) -> np.ndarray:
    buffer = io.BytesIO()
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").convert("RGB").save(
        buffer, format="JPEG", quality=quality, optimize=optimize
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def decode_png(image: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("L"), dtype=np.float32)


def evaluate_variant(predictions: list[np.ndarray], targets: list[np.ndarray], decoder) -> dict[str, float]:
    ssim_sum = 0.0
    psnr_sum = 0.0
    score_sum = 0.0
    for prediction, target in zip(predictions, targets):
        candidate = decoder(prediction)
        ssim = float(structural_similarity(target, candidate, data_range=255, win_size=7))
        psnr = float(peak_signal_noise_ratio(target, candidate, data_range=255))
        ssim_sum += ssim
        psnr_sum += psnr
        score_sum += competition_score(ssim, psnr)
    count = len(predictions)
    return {
        "images": count,
        "ssim": ssim_sum / count,
        "psnr": psnr_sum / count,
        "competition_score_official": score_sum / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare output encoding paths on leakage-safe OOF cache.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cache_root / "manifest.json").read_text(encoding="utf-8"))
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    fold_dirs = sorted(args.cache_root.glob("fold*/"))
    for fold_dir in fold_dirs:
        fold_manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
        fold_names = list(fold_manifest["names"])
        indices = np.arange(len(fold_names), dtype=np.int64)
        if args.max_images > 0 and len(indices) > args.max_images:
            indices = np.linspace(0, len(indices) - 1, args.max_images, dtype=np.int64)
        dapi = np.load(fold_dir / "prediction_thumbnailbase32.npy", mmap_mode="r")
        base48 = np.load(fold_dir / "prediction_thumbnailbase48continue.npy", mmap_mode="r")
        target = np.load(fold_dir / "target.npy", mmap_mode="r")
        for index in indices:
            predictions.append(
                (0.7 * np.asarray(dapi[index], dtype=np.float32)
                 + 0.3 * np.asarray(base48[index], dtype=np.float32))
            )
            targets.append(np.asarray(target[index], dtype=np.float32))
            names.append(fold_names[int(index)])
    if not predictions:
        raise RuntimeError("OOF cache contains no selected images.")

    variants = {
        "jpeg_q75": lambda image: decode_jpeg(image, 75),
        "jpeg_q90": lambda image: decode_jpeg(image, 90),
        "jpeg_q95": lambda image: decode_jpeg(image, 95),
        "jpeg_q98": lambda image: decode_jpeg(image, 98),
        "jpeg_q100": lambda image: decode_jpeg(image, 100),
        "jpeg_q95_optimize": lambda image: decode_jpeg(image, 95, optimize=True),
        "png_lossless": decode_png,
    }
    results = {
        name: evaluate_variant(predictions, targets, decoder)
        for name, decoder in variants.items()
    }
    results["metadata"] = {
        "cache_protocol": manifest.get("protocol_name"),
        "selected_images": len(predictions),
        "source": "0.7*thumbnailbase32 + 0.3*thumbnailbase48continue",
        "selection": "deterministic evenly spaced sample per fold; no test data",
        "names_sha256": __import__("hashlib").sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
