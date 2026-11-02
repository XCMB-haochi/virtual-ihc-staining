import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round235_local_full import command,assess
from run_round208_full_domain import supervised_command
from run_round216_distill import student_command

class ConfirmationTests(unittest.TestCase):
    def test_fixed_change_relative_to_best_recipe(self):
        for stage in (1,2):
            expected=supervised_command(1) if stage==1 else student_command()
            expected[expected.index('--model-arch')+1]='unext2-fcmae-tiny-local-condition'
            expected[expected.index('--output-dir')+1]=f'runs/round235_full_stage{stage}'
            if stage==2:expected[expected.index('--init-checkpoint')+1]='runs/round235_full_stage1/epochs/epoch_014.pt'
            self.assertEqual(command(stage),expected)
    def test_confirmation_requires_mean_and_fold_support(self):
        gains={'overall':.06,**{f'fold{i}':(.1 if i<3 else -.01) for i in range(5)}}
        self.assertTrue(assess(gains)['train_candidate'])
        gains['overall']=.01;self.assertFalse(assess(gains)['train_candidate'])
        gains['overall']=.1;gains['fold2']=-.01
        self.assertFalse(assess(gains)['train_candidate'])
        del gains['fold4']
        with self.assertRaises(ValueError):assess(gains)

if __name__=='__main__':unittest.main()
