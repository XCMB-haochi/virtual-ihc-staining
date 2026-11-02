import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round230_fit_probe import select_names,weighted_mse

class SelectionTests(unittest.TestCase):
    def test_mse_matches_existing_channel_weighted_objective(self):
        import torch
        from ihc_torch import CompetitionLoss
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        p=torch.rand(2,4,16,16,requires_grad=True);y=torch.rand_like(p);w=torch.tensor([.05,.85,.05,.05])
        fast=weighted_mse(p,y,w)
        reference=CompetitionLoss(ssim_weight=0,l1_weight=0,mse_weight=1,ssim_mode='uniform7',channel_weights=w)(p,y)
        torch.testing.assert_close(fast,reference)
        torch.testing.assert_close(torch.autograd.grad(fast,p)[0],torch.autograd.grad(reference,p)[0])

    def test_repeatable_unique_four_per_quartile(self):
        names=[str(i) for i in range(100)]
        selected=select_names(names,np.arange(100))
        self.assertEqual(selected,select_names(names,np.arange(100)))
        self.assertEqual(len(set(selected)),16)
        for q in range(4):self.assertEqual(sum(q*25<=int(n)<(q+1)*25 for n in selected),4)

if __name__=='__main__':unittest.main()
