import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round245_teacher_transfer import command
from run_round240_teacher_refresh import command as baseline

class TeacherTransferTests(unittest.TestCase):
    def test_only_cache_and_output_change(self):
        expected=baseline()
        expected[expected.index('--output-dir')+1]='runs/round245_full_stage2'
        expected[expected.index('--distillation-target-dir')+1]='runs/round245_oof_targets/targets'
        self.assertEqual(command(),expected)
        self.assertEqual(command()[command().index('--init-checkpoint')+1],'runs/round239_full_stage1/epochs/epoch_014.pt')

if __name__=='__main__':unittest.main()
