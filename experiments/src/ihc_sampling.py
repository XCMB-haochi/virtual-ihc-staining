from __future__ import annotations

import bisect
import math
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from ihc_data import load_gray


def target_activity(image: np.ndarray) -> float:
    """Compute a deterministic target-only activity score in uint8 units."""

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError(f"Activity expects a non-empty grayscale image, got {values.shape}.")
    horizontal = np.abs(values[:, 1:] - values[:, :-1]).mean() if values.shape[1] > 1 else 0.0
    vertical = np.abs(values[1:, :] - values[:-1, :]).mean() if values.shape[0] > 1 else 0.0
    mean = float(values.mean())
    std = float(values.std())
    p95 = float(np.quantile(values, 0.95))
    gradient = 0.5 * (float(horizontal) + float(vertical))
    foreground = float(np.mean(values > mean))
    value = std + gradient + 0.25 * foreground + 0.25 * p95
    if not math.isfinite(value):
        raise ValueError("Activity score is not finite.")
    return value


def training_target_activities(
    data_root: Path,
    names: Sequence[str],
    marker: str,
) -> list[float]:
    """Read activity only from the labelled training images used by a fold."""

    if not names:
        raise ValueError("Activity sampling requires non-empty training names.")
    directory = data_root / "train" / marker
    values = [target_activity(load_gray(directory / name)) for name in names]
    if not values or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("Training target activities contain non-finite values.")
    return values


class ActivityHardMixSampler(Sampler[int]):
    """Shuffle all samples, then replace a fixed late-epoch fraction with hard samples."""

    def __init__(
        self,
        activities: Sequence[float],
        *,
        seed: int,
        num_bins: int = 4,
        start_epoch: int,
        hard_fraction: float,
    ) -> None:
        if len(activities) == 0:
            raise ValueError("activities cannot be empty")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be at least two")
        if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 0:
            raise ValueError("start_epoch must be a non-negative integer")
        if not math.isfinite(float(hard_fraction)) or not 0.0 < float(hard_fraction) < 1.0:
            raise ValueError("hard_fraction must lie in (0, 1)")
        values = tuple(float(value) for value in activities)
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("activities must be finite")
        self.activities = values
        self.seed = seed
        self.num_bins = num_bins
        self.start_epoch = start_epoch
        self.hard_fraction = float(hard_fraction)
        sorted_values = sorted(values)
        boundaries: list[float] = []
        for index in range(1, num_bins):
            position = (len(sorted_values) - 1) * index / num_bins
            lower = math.floor(position)
            upper = math.ceil(position)
            fraction = position - lower
            boundary = sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
            if not boundaries or boundary > boundaries[-1]:
                boundaries.append(boundary)
        self.quantile_boundaries = tuple(boundaries)
        assignments = tuple(bisect.bisect_left(self.quantile_boundaries, value) for value in values)
        highest_bin = max(assignments)
        self.hard_indices = tuple(index for index, group in enumerate(assignments) if group == highest_bin)
        if not self.hard_indices:
            raise ValueError("The activity sampler has no hard-sample bin.")
        self.epoch = 0
        self.last_epoch_indices: tuple[int, ...] = ()
        self.last_hard_indices: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.activities)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        all_indices = list(range(len(self.activities)))
        rng.shuffle(all_indices)
        selected_hard: list[int] = []
        if self.epoch - 1 >= self.start_epoch:
            hard_count = int(round(len(all_indices) * self.hard_fraction))
            hard_count = max(1, min(len(all_indices) - 1, hard_count))
            selected_hard = [
                self.hard_indices[rng.randrange(len(self.hard_indices))]
                for _ in range(hard_count)
            ]
            all_indices = all_indices[: len(all_indices) - hard_count] + selected_hard
            rng.shuffle(all_indices)
        self.last_epoch_indices = tuple(all_indices)
        self.last_hard_indices = tuple(selected_hard)
        return iter(all_indices)
