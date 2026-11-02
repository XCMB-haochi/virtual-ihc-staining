from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_training_screen as screen
from run_training_screen import eligible_gpus


class ResourceSchedulingTests(unittest.TestCase):
    @mock.patch("run_training_screen.time.sleep")
    @mock.patch("run_training_screen.subprocess.check_output")
    def test_transient_compute_and_low_memory_exclude_cards(self, query, sleep):
        query.side_effect = ["0, 0, 40000\n1, 0, 25000"] * 4 + ["0, 91, 40000\n1, 0, 15000"]
        self.assertEqual(eligible_gpus(24000), [])

    @mock.patch("run_training_screen.time.sleep")
    @mock.patch("run_training_screen.subprocess.check_output", return_value="0, 0, 40000\n1, 0, 35000")
    def test_both_available_cards_are_used_but_active_gpu_is_excluded(self, query, sleep):
        self.assertEqual(eligible_gpus(24000), [0, 1])
        self.assertEqual(eligible_gpus(24000, excluded={0}), [1])

    @mock.patch("run_training_screen.time.sleep")
    @mock.patch("run_training_screen.subprocess.check_output", return_value="0, 0, 40000\n1, 0, 35000")
    def test_recovery_gpu_pool_excludes_unselected_card(self, query, sleep):
        self.assertEqual(eligible_gpus(24000, candidates=[1]), [1])


class RecoveryTests(unittest.TestCase):
    def test_only_complete_fold_is_reused_and_partial_stage_is_rejected(self):
        protocol = {
            "name": "round999_recovery", "training": {"epochs_per_stage": 2},
            "folds": [{"name": "fold0"}, {"name": "fold2"}],
            "recipes": {"candidate": {"train_stages": [1, 2]}},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(screen, "ROOT", Path(directory)):
            for stage in (1, 2):
                path = screen.run_directory(Path(directory)/"runs", protocol["name"], "fold0", "candidate", stage)
                path.mkdir(parents=True)
                (path/"best_global_mse.pt").write_bytes(b"checkpoint")
                (path/"metrics.jsonl").write_text('\n'.join(json.dumps({"epoch":i}) for i in (1,2)))
            with self.assertRaises(FileExistsError):
                screen.preflight(protocol, "round999")
            screen.preflight(protocol, "round999", allow_completed=True)
            self.assertEqual(screen.pending_folds(protocol, "candidate"), ["fold2"])
            (path/"metrics.jsonl").write_text('{"epoch":2}\n{"epoch":2}\n')
            with self.assertRaises(FileExistsError):
                screen.preflight(protocol, "round999", allow_completed=True)


if __name__ == "__main__":
    unittest.main()
