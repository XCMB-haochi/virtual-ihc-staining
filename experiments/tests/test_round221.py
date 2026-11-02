import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round221_texture_full import command
from run_round208_full_domain import supervised_command
from run_round216_distill import student_command

class FixedCandidateTests(unittest.TestCase):
    def test_only_texture_and_artifact_paths_change(self):
        for stage in (1,2):
            expected=supervised_command(1) if stage==1 else student_command()
            expected[expected.index('--condition-mode')+1]='spatial-thumbnail-texture-film'
            expected[expected.index('--output-dir')+1]=f'runs/round221_full_stage{stage}'
            if stage==2:expected[expected.index('--init-checkpoint')+1]='runs/round221_full_stage1/epochs/epoch_014.pt'
            self.assertEqual(command(stage),expected)

if __name__=='__main__':unittest.main()
