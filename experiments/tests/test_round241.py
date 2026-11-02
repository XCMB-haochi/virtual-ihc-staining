import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from skimage.metrics import structural_similarity
from ihc_torch import CompetitionLoss,ssim_gaussian_valid_per_channel
from run_round241_late_gaussian import command
from run_round239_stage_decay import command as fold_reference
from run_round240_teacher_refresh import command as full_reference

class LateGaussianTests(unittest.TestCase):
    def test_numerical_reference_and_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(2026)
        target=torch.rand(2,2,40,48)
        prediction=(target*.8+torch.rand_like(target)*.2).requires_grad_()
        actual=ssim_gaussian_valid_per_channel(prediction,target)
        expected=np.array([[structural_similarity(prediction[b,c].detach().numpy().astype('float64'),target[b,c].numpy().astype('float64'),data_range=1,win_size=11,gaussian_weights=True,sigma=1.5,use_sample_covariance=False) for c in range(2)] for b in range(2)])
        np.testing.assert_allclose(actual.detach().numpy(),expected,atol=3e-6,rtol=0)
        for weights in (None,[.15,.85]):
            loss=CompetitionLoss(ssim_mode='gaussian11_valid',channel_weights=weights)
            self.assertTrue(torch.allclose(loss(prediction,target),loss.per_sample(prediction,target).mean(),atol=1e-7))
            loss(prediction,target).backward(retain_graph=True)
        self.assertTrue(torch.isfinite(prediction.grad).all());self.assertGreater(prediction.grad.abs().sum().item(),0)
        blank=torch.zeros(1,1,16,16)
        self.assertAlmostEqual(ssim_gaussian_valid_per_channel(blank,blank).item(),1)

    def test_only_loss_and_output_change(self):
        f={'name':'fold0','val_rois':['ROI005','ROI052','ROI087']}
        for fold in (None,f):
            expected=full_reference() if fold is None else fold_reference(fold,2)
            name='full' if fold is None else fold['name']
            expected[expected.index('--output-dir')+1]=f'runs/round241_{name}_stage2'
            expected[expected.index('--ssim-mode')+1]='gaussian11_valid'
            self.assertEqual(command(fold),expected)
            self.assertEqual('--distillation-target-dir' in command(fold),fold is None)

if __name__=='__main__':unittest.main()
