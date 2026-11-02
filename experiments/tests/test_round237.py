import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round237_context128_screen import command, selected_folds, validate_command, compare, audit_checkpoint


class ContextScreenTests(unittest.TestCase):
    def test_fold_exclusion_in_both_stages(self):
        for fold in selected_folds():
            for stage in (1,2):
                cmd=command(fold,stage)
                self.assertNotIn('--train-all',cmd)
                self.assertNotIn('--distillation-target-dir',cmd)
                self.assertEqual(cmd[cmd.index('--val-rois')+1],','.join(fold['val_rois']))
                if stage==2:
                    self.assertEqual(cmd[cmd.index('--init-checkpoint')+1],f"runs/round237_{fold['name']}_stage1/epochs/epoch_014.pt")
                for bad in ('--train-all','--distillation-target-dir'):
                    with self.assertRaises(ValueError): validate_command(cmd+[bad],fold)

    def test_counts_come_from_config_not_checkpoint_args(self):
        fold=selected_folds()[0]
        a={'train_all':False,'distillation_weight':0,'val_rois':','.join(fold['val_rois']),
           'train_rois_resolved':['ROI010'],'context_halo':128,'train_target_context_halo':64,
           'seed':2026,'batch_size':4,'gradient_accumulation_steps':4}
        ck={'args':a,'optimizer':{'state':{0:{'step':1720}}}}
        self.assertEqual(audit_checkpoint(ck,fold,config={'val_images':420,'train_images':1960}),1720)
        with self.assertRaises(AssertionError):
            audit_checkpoint(ck,fold,config={'val_images':420,'train_images':2380})

    def test_overall_uses_only_matched_rois(self):
        folds=selected_folds()
        baseline={'overall':{'gaussian11_score':999}}
        current={'overall':{'gaussian11_score':77+560/980,'images':980}}
        for fold,score in zip(folds,(76,77)):
            for key in [fold['name'],*fold['val_rois']]:
                baseline[key]={'gaussian11_score':score}
                current[key]={'gaussian11_score':score+1,'images':140*len(fold['val_rois'])}
        result=compare({'CD45RO':current},{'CD45RO':baseline},folds)
        self.assertAlmostEqual(result['CD45RO']['overall'],1)


if __name__=='__main__':unittest.main()
