from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round308 import geometry_consistency_loss
from ihc_torch import CompetitionLoss,apply_d4_transform,invert_d4_transform


class GeometryTests(unittest.TestCase):
    def test_all_inverse_transforms_restore_coordinates(self):
        image=torch.arange(256).reshape(1,1,16,16)
        for i in range(8): self.assertTrue(torch.equal(image,invert_d4_transform(apply_d4_transform(image,i),i)))

    def test_both_predictions_receive_gradient_without_rng_or_label_changes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.conv=torch.nn.Conv2d(1,1,3,padding=1); self.outputs=[]
            def forward(self,image,marker):
                out=self.conv(image).sigmoid(); out.retain_grad(); self.outputs.append(out); return out
        torch.manual_seed(2026); model=Model(); image=torch.rand(1,1,24,24)
        source=image.clone(); marker=torch.tensor([0]); prediction=model(image,marker)
        rng=torch.get_rng_state(); py_rng=random.getstate()
        loss=geometry_consistency_loss(model,image,marker,prediction,1,CompetitionLoss(ssim_mode='uniform7'))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(len(model.outputs),2)
        self.assertTrue(all(x.grad.abs().sum()>0 for x in model.outputs))
        self.assertTrue(torch.equal(image,source)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py_rng,random.getstate())


if __name__=='__main__': unittest.main()
