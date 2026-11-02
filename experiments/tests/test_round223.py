import sys
import unittest
from pathlib import Path
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from ihc_torch import build_model,model_specification,build_model_from_checkpoint
from train_unet import signal_occupancy_loss,training_batch_losses
from ihc_torch import CompetitionLoss
from run_round223_signal_aux import command
from run_round204_cd45_cv import training_command

class SignalAuxTests(unittest.TestCase):
    def test_only_new_architecture_and_auxiliary_task_change(self):
        fold={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
        for stage in (1,2):
            expected,_=training_command(fold,stage,14)
            expected[expected.index('--model-arch')+1]='unext2-fcmae-tiny-signal-aux'
            expected[expected.index('--output-dir')+1]=f'runs/round223_fold3_stage{stage}'
            if stage==1:expected[expected.index('--fcmae-encoder-init-checkpoint')+1]='runs/round207_fold3_fcmae800/final_encoder.pt'
            else:expected[expected.index('--init-checkpoint')+1]='runs/round223_fold3_stage1/epochs/epoch_014.pt'
            expected+=['--signal-auxiliary-weight','0.05']
            self.assertEqual(command(fold,stage),expected)

    def test_target_channel_and_fraction_supervision(self):
        logits=torch.zeros(2,4,2,2,requires_grad=True)
        target=torch.zeros(2,4,32,32)
        target[:,1,:16,:16]=1
        loss=signal_occupancy_loss(logits,target,1);loss.backward()
        self.assertLess(float(logits.grad[:,1,0,0].sum()),0)
        self.assertGreater(float(logits.grad[:,1,1,1].sum()),0)
        self.assertEqual(float(logits.grad[:,[0,2,3]].abs().sum()),0)
        self.assertAlmostEqual(float(loss.detach()),float(torch.log(torch.tensor(2.))),places=6)

    def test_inference_equivalence_restore_and_encoder_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        baseline=build_model('unext2-fcmae-tiny',out_channels=4,condition_mode='none')
        torch.manual_seed(2026)
        candidate=build_model('unext2-fcmae-tiny-signal-aux',out_channels=4,condition_mode='none')
        for key,value in baseline.state_dict().items():self.assertTrue(torch.equal(value,candidate.state_dict()[key]))
        x=torch.rand(2,1,64,64); y=torch.rand(2,4,64,64)
        with torch.no_grad():
            torch.testing.assert_close(candidate(x),baseline(x),rtol=0,atol=0)
            candidate.signal_head[-1].weight.fill_(.01)
        losses=[]
        total,supervised,*_=training_batch_losses(candidate,x,y,None,CompetitionLoss(),0,
            signal_auxiliary_weight=.05,signal_auxiliary_marker_index=1,signal_auxiliary_output=losses)
        self.assertEqual(len(losses),1)
        total.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in candidate.encoder.parameters()))
        self.assertGreater(float(candidate.signal_head[-1].weight.grad.abs().sum()),0)
        candidate.zero_grad(set_to_none=True)
        _,aux=candidate.forward_with_signal_logits(x)
        signal_occupancy_loss(aux,y,1).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in candidate.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in candidate.decoder.parameters()))
        spec=model_specification('unext2-fcmae-tiny-signal-aux')
        payload={'model':candidate.state_dict(),'markers':['CD68','CD45RO','HLA-DR','Vimentin'],
                 'base_channels':32,'block_type':'residual','model_spec':spec,
                 'args':{'model_arch':'unext2-fcmae-tiny-signal-aux','condition_mode':'none'}}
        restored=build_model_from_checkpoint(payload,torch.device('cpu'))
        with torch.no_grad():torch.testing.assert_close(candidate(x),restored(x),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
