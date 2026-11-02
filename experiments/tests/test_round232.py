import sys
import unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'src')]
from run_round232_primary import command
from run_round204_cd45_cv import training_command

class PrimaryObjectiveTests(unittest.TestCase):
    def test_only_target_weights_and_paths_change(self):
        for name in ('fold0','fold2'):
            fold={'name':name,'val_rois':['ROI005']}
            for stage in (1,2):
                expected,_=training_command(fold,stage,14)
                expected[expected.index('--marker-loss-weights')+1]='0,1,0,0'
                expected[expected.index('--output-dir')+1]=f'runs/round232_{name}_stage{stage}'
                if stage==1:expected[expected.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/{'round207' if name=='fold0' else 'round216'}_{name}_fcmae800/final_encoder.pt"
                else:expected[expected.index('--init-checkpoint')+1]=f'runs/round232_{name}_stage1/epochs/epoch_014.pt'
                self.assertEqual(command(fold,stage),expected)
    def test_other_marker_labels_have_no_loss_or_output_gradient(self):
        import torch
        from ihc_torch import CompetitionLoss
        torch.set_num_threads(2);torch.manual_seed(2026)
        prediction=torch.rand(2,4,16,16,requires_grad=True);target=torch.rand_like(prediction)
        loss=CompetitionLoss(ssim_mode='uniform7',channel_weights=[0,1,0,0])
        value=loss(prediction,target);value.backward()
        self.assertEqual(float(prediction.grad[:,[0,2,3]].abs().sum()),0)
        self.assertGreater(float(prediction.grad[:,1].abs().sum()),0)
        altered=target.clone();altered[:,[0,2,3]]=0
        torch.testing.assert_close(value,loss(prediction,altered),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
