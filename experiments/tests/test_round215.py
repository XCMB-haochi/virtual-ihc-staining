import sys
import unittest
import tempfile
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from run_round215_nested import partitions,pretraining_command
from train_unet import training_batch_losses, validate_distillation_target_coverage
from ihc_torch import CompetitionLoss

class Model(torch.nn.Module):
    def forward(self,x,condition=None):return x

class ZeroLoss(torch.nn.Module):
    def forward(self,prediction,target):return prediction.sum()*0

class NestedTests(unittest.TestCase):
    def test_incomplete_training_targets_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'ROI005_00_00.jpg').write_bytes(b'smoke')
            with self.assertRaisesRegex(ValueError,'missing for 1'):
                validate_distillation_target_coverage(root,['ROI005_00_00.jpg','ROI076_05_08.jpg'])
            validate_distillation_target_coverage(root,['ROI005_00_00.jpg'])

    def test_full_coverage_and_double_roi_exclusion(self):
        rois=[f'ROI{i:03}' for i in range(17)]
        for excluded in (rois[:3],rois[4:8]):
            groups=partitions(rois,excluded)
            held=[r for g in groups for r in g['holdout']]
            self.assertEqual(set(held),set(rois)-set(excluded))
            self.assertEqual(len(held),len(set(held)))
            for g in groups:
                self.assertFalse(set(g['train']) & (set(excluded)|set(g['holdout'])))
                self.assertEqual(set(g['train'])|set(g['holdout']),set(rois)-set(excluded))
                cmd,_=pretraining_command('outer_only_data','fold0',g)
                self.assertEqual(cmd[2],'outer_only_data')
                self.assertEqual(cmd[cmd.index('--batch-size')+1],'32')

    def test_cd45ro_teacher_gradients_reach_only_channel_one(self):
        prediction=torch.full((2,4,16,16),.2,requires_grad=True)
        teacher=torch.full((2,1,16,16),.9)
        criterion=CompetitionLoss(ssim_weight=0,l1_weight=1,mse_weight=0,ssim_mode='uniform7')
        total,*_=training_batch_losses(Model(),prediction,torch.zeros_like(prediction),None,ZeroLoss(),0,
            distillation_target=teacher,distillation_criterion=criterion,distillation_weight=.2,distillation_marker_index=1)
        total.backward()
        self.assertGreater(float(prediction.grad[:,1].abs().sum()),0)
        for channel in (0,2,3):self.assertEqual(float(prediction.grad[:,channel].abs().sum()),0)

if __name__=='__main__':unittest.main()
