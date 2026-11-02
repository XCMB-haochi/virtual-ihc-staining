from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from summarize_label_scale import aggregate_records, optimizer_updates_per_stage


def record(recipe: str, score: float) -> dict[str, object]:
    return {
        "recipe": recipe,
        "train_images": 100,
        "jpeg_q95": {"competition_score": score},
        "float": {"competition_score": score + 0.1},
    }


class LabelScaleSummaryTests(unittest.TestCase):
    def test_optimizer_update_count_includes_partial_batches(self) -> None:
        self.assertEqual(optimizer_updates_per_stage(33, 8, 2, 3), 9)

    def test_aggregate_reports_scale_spread_and_monotonic_draws(self) -> None:
        scores = {
            "full20": 80.0,
            "a5": 70.0,
            "a10": 74.0,
            "a15": 78.0,
            "b5": 71.0,
            "b10": 76.0,
            "b15": 79.0,
            "c5": 75.0,
            "c10": 74.0,
            "c15": 79.0,
        }
        records = {name: record(name, score) for name, score in scores.items()}
        metadata = {
            "full20": {"draw": "baseline", "roi_count": 20},
            **{
                name: {"draw": name[0], "roi_count": int(name[1:])}
                for name in scores
                if name != "full20"
            },
        }

        summary = aggregate_records(records, metadata, "full20")

        self.assertAlmostEqual(summary["scale_summary"]["5"]["jpeg_q95_mean"], 72.0)
        self.assertAlmostEqual(
            summary["scale_summary"]["15"]["gap_to_full20"], 4.0 / 3.0
        )
        self.assertEqual(summary["monotonic_draws"], ["a", "b"])
        self.assertFalse(summary["draw_curves"]["c"]["strictly_monotonic"])


if __name__ == "__main__":
    unittest.main()
