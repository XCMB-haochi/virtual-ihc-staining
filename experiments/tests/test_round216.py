import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round216_distill import student_command,teacher_commands
from run_round208_full_domain import supervised_command
from run_round207_domain import pretrain_command
from run_round204_cd45_cv import training_command

class FixedDistillationTests(unittest.TestCase):
    def test_student_only_adds_distillation(self):
        expected=supervised_command(2)
        expected[expected.index('--output-dir')+1]='runs/round216_full_stage2'
        expected+=['--distillation-target-dir','runs/round216_oof_targets/targets','--distillation-marker','CD45RO','--distillation-weight','0.2']
        self.assertEqual(student_command(),expected)

    def test_new_teachers_keep_existing_recipe(self):
        fold={'name':'fold1','val_rois':['ROI010','ROI058','ROI093']}
        commands=teacher_commands(fold)
        pre,_=pretrain_command(fold)
        self.assertEqual(commands[0],[v.replace('round207_','round216_') for v in pre])
        for stage in (1,2):
            cmd=commands[stage]
            self.assertEqual(cmd[cmd.index('--batch-size')+1],'4')
            self.assertEqual(cmd[cmd.index('--gradient-accumulation-steps')+1],'4')
            self.assertEqual(cmd[cmd.index('--val-rois')+1],','.join(fold['val_rois']))

if __name__=='__main__':unittest.main()
