import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round211_batch16 import command_for
from run_round210_relational import student_command


class Batch16Tests(unittest.TestCase):
    def test_fresh_students_keep_teachers_and_domain_initialization(self):
        fold={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
        teacher='runs/round210_fold3_teacher160/final_encoder.pt'
        for recipe in ('control','relational'):
            for stage in (1,2):
                cmd=command_for(fold,stage,recipe,teacher,'a'*64)
                original=student_command(fold,stage,recipe,teacher,'a'*64)
                expected=[s.replace('runs/round210_fold3_'+recipe,'runs/round211_fold3_'+recipe) for s in original]
                expected[expected.index('--batch-size')+1]='16'
                if recipe=='relational':
                    expected[expected.index('--feature-guidance-batch-samples')+1]='16'
                    self.assertEqual(cmd[cmd.index('--feature-guidance-checkpoint')+1],teacher)
                self.assertEqual(cmd,expected)
                self.assertEqual(cmd[cmd.index('--gradient-accumulation-steps')+1],'1')
                self.assertNotIn('--resume',cmd)
                if stage==1:
                    self.assertNotIn('--init-checkpoint',cmd)
                    self.assertEqual(cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1],
                                     'runs/round207_fold3_fcmae800/final_encoder.pt')
                else:
                    self.assertEqual(cmd[cmd.index('--init-checkpoint')+1],
                                     f'runs/round211_fold3_{recipe}_stage1/epochs/epoch_014.pt')


if __name__=='__main__': unittest.main()
