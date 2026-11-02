import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from ihc_unext2 import UNeXt2VirtualStainer
from train_unet import set_encoder_freeze_for_epoch,build_optimizer
from run_round244_decoder_first import command
from run_round239_stage_decay import command as reference

class DecoderFirstTests(unittest.TestCase):
    def test_frozen_encoder_and_restored_updates(self):
        torch.set_num_threads(2);torch.manual_seed(2026)
        model=UNeXt2VirtualStainer(out_channels=4)
        opt,_=build_optimizer(model,'unext2-fcmae-tiny',3e-4,None,1e-4,.8)
        enc=next(model.encoder.parameters());dec=next(model.decoder.parameters())
        original_enc=enc.detach().clone();original_dec=dec.detach().clone()
        self.assertTrue(set_encoder_freeze_for_epoch(model,1,1))
        x=torch.rand(1,1,64,64);model(x).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(enc,original_enc));self.assertFalse(torch.equal(dec,original_dec));self.assertIsNone(enc.grad)
        self.assertFalse(set_encoder_freeze_for_epoch(model,2,1))
        model(x).square().mean().backward();opt.step()
        self.assertFalse(torch.equal(enc,original_enc))
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))

    def test_matched_recipe_and_full_only_teacher(self):
        fold={'name':'fold0','val_rois':['ROI005','ROI052','ROI087']}
        for f in (None,fold):
            for stage in (1,2):
                base=reference(f,stage)
                base=[v.replace('round239_','round244_') for v in base]
                if stage==1:base+=['--encoder-freeze-epochs','1']
                if f is None and stage==2:base[base.index('--distillation-target-dir')+1]='runs/round240_oof_targets/targets'
                self.assertEqual(command(f,stage),base)
                self.assertEqual('--distillation-target-dir' in base,f is None and stage==2)

if __name__=='__main__':unittest.main()
