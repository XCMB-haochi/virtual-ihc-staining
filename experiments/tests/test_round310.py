import random
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round308 import training_input_gain


class GainTests(unittest.TestCase):
    def test_train_input_gain_leaves_input_and_global_rng_unchanged(self):
        image=torch.tensor([0.,.5,1.]); original=image.clone()
        state=random.getstate()
        output=training_input_gain(image,random.Random(2026),(.9,1.1))
        self.assertEqual(state,random.getstate())
        self.assertTrue(torch.equal(image,original))
        self.assertGreaterEqual(output.min().item(),0.)
        self.assertLessEqual(output.max().item(),1.)
        self.assertGreaterEqual(output[1].item(),.45)
        self.assertLessEqual(output[1].item(),.55)
        self.assertTrue(torch.equal(output,training_input_gain(image,random.Random(2026),(.9,1.1))))

    def test_unit_gain_and_clipping(self):
        image=torch.tensor([.2,.8,1.])
        self.assertTrue(torch.equal(image,training_input_gain(image,random.Random(2026),(1,1))))
        self.assertTrue(torch.allclose(training_input_gain(image,random.Random(2026),(1.1,1.1)),torch.tensor([.22,.88,1.])))


if __name__=='__main__': unittest.main()
