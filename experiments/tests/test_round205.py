import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_round205_full_data import full_command


class FullDataClockTests(unittest.TestCase):
    def test_full_data_does_not_shorten_cosine_to_selected_epoch(self):
        for stage in (1, 2):
            command = full_command(stage)
            self.assertEqual(command[command.index('--epochs') + 1], '14')
            self.assertIn('--train-all', command)
            self.assertEqual(command[command.index('--val-rois') + 1], '')
            self.assertEqual(command[command.index('--seed') + 1], '2026')
        command = full_command(2)
        self.assertEqual(command[command.index('--init-checkpoint') + 1],
                         'runs/round205_full_stage1/epochs/epoch_014.pt')
        self.assertNotIn('--distillation-target-dir', command)


if __name__ == '__main__':
    unittest.main()
