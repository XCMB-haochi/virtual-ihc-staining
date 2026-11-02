import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from ihc_unext2 import UNeXt2VirtualStainer
from train_unet import build_optimizer
from run_round239_stage_decay import command

class StageDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=UNeXt2VirtualStainer(out_channels=4)

    def test_partition_and_cosine_ratios(self):
        opt,report=build_optimizer(self.model,'unext2-fcmae-tiny',3e-4,None,1e-4,.8)
        expected=[1,.4096,.512,.64,.8,1]
        ids=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in self.model.parameters()})
        for g,scale in zip(opt.param_groups,expected):self.assertAlmostEqual(g['lr']/3e-4,scale)
        for i,mod in enumerate([self.model.encoder.stem,*self.model.encoder.stages]):
            self.assertEqual({id(p) for p in opt.param_groups[i+1]['params']},{id(p) for p in mod.parameters()})
        scheduler=CosineAnnealingLR(opt,T_max=14,eta_min=0)
        for _ in range(7):opt.step();scheduler.step()
        for g,scale in zip(opt.param_groups,expected):self.assertAlmostEqual(g['lr']/(1.5e-4),scale)

    def test_default_keeps_original_two_groups(self):
        opt,report=build_optimizer(self.model,'unext2-fcmae-tiny',3e-4,None,1e-4)
        self.assertEqual([x['name'] for x in report],['decoder_head','encoder'])
        self.assertEqual([x['lr'] for x in opt.param_groups],[3e-4,3e-4])
        for invalid in (0,1.1,float('nan')):
            with self.assertRaises(ValueError):build_optimizer(self.model,'unext2-fcmae-tiny',3e-4,None,1e-4,invalid)

    def test_fold_and_full_commands_do_not_cross(self):
        fold={'name':'fold0','val_rois':['ROI005','ROI052','ROI087']}
        for stage in (1,2):
            cv=command(fold,stage);full=command(None,stage)
            self.assertNotIn('--train-all',cv);self.assertNotIn('--distillation-target-dir',cv)
            self.assertEqual(cv[cv.index('--val-rois')+1],','.join(fold['val_rois']))
            self.assertIn('--train-all',full)
            self.assertEqual('--distillation-target-dir' in full,stage==2)
            if stage==2:
                self.assertEqual(cv[cv.index('--init-checkpoint')+1],'runs/round239_fold0_stage1/epochs/epoch_014.pt')
                self.assertEqual(full[full.index('--init-checkpoint')+1],'runs/round239_full_stage1/epochs/epoch_014.pt')

if __name__=='__main__':unittest.main()
