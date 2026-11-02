from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import numpy as np

from ihc_sampling import ActivityHardMixSampler, target_activity


class ActivitySamplerTests(unittest.TestCase):
    def test_activity_is_finite_and_sensitive_to_structure(self) -> None:
        flat = np.full((8, 8), 10, dtype=np.uint8)
        checker = np.indices((8, 8)).sum(axis=0).astype(np.uint8) * 20
        self.assertGreater(target_activity(checker), target_activity(flat))

    def test_pre_start_epoch_is_a_permutation(self) -> None:
        sampler = ActivityHardMixSampler(
            [float(index) for index in range(20)],
            seed=2044,
            num_bins=4,
            start_epoch=2,
            hard_fraction=0.30,
        )
        first = list(iter(sampler))
        second = list(iter(sampler))
        self.assertEqual(sorted(first), list(range(20)))
        self.assertEqual(sorted(second), list(range(20)))
        self.assertEqual(len(set(first)), 20)
        self.assertEqual(len(set(second)), 20)

    def test_late_epoch_replaces_only_fixed_hard_fraction(self) -> None:
        sampler = ActivityHardMixSampler(
            [float(index) for index in range(20)],
            seed=2044,
            num_bins=4,
            start_epoch=0,
            hard_fraction=0.30,
        )
        indices = list(iter(sampler))
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(sampler.last_hard_indices), 6)
        self.assertTrue(all(index in range(20) for index in indices))

    def test_same_seed_is_reproducible(self) -> None:
        values = [float(index * index) for index in range(16)]
        left = ActivityHardMixSampler(
            values, seed=7, num_bins=4, start_epoch=0, hard_fraction=0.25
        )
        right = ActivityHardMixSampler(
            values, seed=7, num_bins=4, start_epoch=0, hard_fraction=0.25
        )
        self.assertEqual(list(iter(left)), list(iter(right)))
        self.assertEqual(list(iter(left)), list(iter(right)))


if __name__ == "__main__":
    unittest.main()
