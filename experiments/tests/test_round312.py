from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round300 import gaussian_ssim
from round308 import distilled_loss
from ihc_torch import CompetitionLoss


class GaussianLossTests(unittest.TestCase):
    def test_matches_validation_objective_and_gradients(self):
        torch.manual_seed(2026)
        x=torch.rand(1,1,32,32,requires_grad=True)
        y=torch.rand_like(x); teacher=torch.rand_like(x)
        criterion=CompetitionLoss(ssim_mode='gaussian11_valid')
        def reference(t):
            return .7*(1-gaussian_ssim(x,t))+.2*(x-t).abs().mean()+.1*(x-t).square().mean()
        actual=distilled_loss(x,y,teacher,criterion)
        expected=.8*reference(y)+.2*reference(teacher)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        a=torch.autograd.grad(actual,x,retain_graph=True)[0]
        b=torch.autograd.grad(expected,x)[0]
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.allclose(a,b,atol=1e-6))


if __name__=='__main__': unittest.main()
