from __future__ import annotations
import os,random,math,json,tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ihc_data import load_gray


class MarkerPatchDataset(Dataset):
    def __init__(
        self,
        train_directory: Path,
        samples: list[tuple[str, str]],
        augment: bool = True,
    ) -> None:
        self.train_directory = train_directory
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        marker, name = self.samples[index]
        image = load_gray(self.train_directory / marker / name)[None, ...]
        if self.augment:
            transform_index = random.randrange(8)
            image = np.rot90(image, transform_index % 4, axes=(-2, -1))
            if transform_index >= 4:
                image = image[..., ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(image) / 255.0).float()
        return tensor, f"{marker}/{name}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_rng_states_for_restore(value: object) -> list[torch.Tensor]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("CUDA RNG state must be a non-empty tensor sequence.")
    states = []
    for state in value:
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
        ):
            raise ValueError("CUDA RNG states must be one-dimensional ByteTensors.")
        states.append(state.detach().cpu())
    return states


def learning_rate_for_epoch(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch must be between 1 and epochs.")
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be non-negative and below epochs.")
    if warmup_epochs and epoch <= warmup_epochs:
        return base_learning_rate * epoch / warmup_epochs
    progress = (epoch - warmup_epochs - 1) / max(epochs - warmup_epochs - 1, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(
        minimum_learning_rate
        + (base_learning_rate - minimum_learning_rate) * cosine
    )


def gradient_accumulation_sample_scale(
    batch_index: int,
    micro_batch_size: int,
    current_batch_size: int,
    accumulation_steps: int,
    dataset_size: int,
) -> float:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    group_first_sample = group_start * micro_batch_size
    group_sample_count = min(
        accumulation_steps * micro_batch_size,
        dataset_size - group_first_sample,
    )
    if group_sample_count <= 0 or current_batch_size > group_sample_count:
        raise ValueError("FCMAE gradient accumulation batch index exceeds the dataset.")
    return current_batch_size / group_sample_count


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def truncate_metrics(path: Path, maximum_epoch: int) -> None:
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if int(payload["epoch"]) <= maximum_epoch:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def build_epoch_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=generator,
    )

