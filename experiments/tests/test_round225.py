import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round225_signal_full import command
from run_round208_full_domain import supervised_command

class FixedAuxiliaryTests(unittest.TestCase):
    def test_only_auxiliary_task_and_paths_change_from_round208(self):
        for stage in (1,2):
            expected=supervised_command(stage)
            expected[expected.index('--model-arch')+1]='unext2-fcmae-tiny-signal-aux'
            expected[expected.index('--output-dir')+1]=f'runs/round225_full_stage{stage}'
            if stage==2:expected[expected.index('--init-checkpoint')+1]='runs/round225_full_stage1/epochs/epoch_014.pt'
            expected+=['--signal-auxiliary-weight','0.05']
            self.assertEqual(command(stage),expected)
            self.assertNotIn('--distillation-weight',command(stage))

if __name__=='__main__':unittest.main()
