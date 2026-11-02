import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round303 import organ_loss_weights, epoch_order


class LossWeightsTests(unittest.TestCase):
    def test_exact_gradient_matches_weighted_organ_means(self):
        counts={'colon':5,'liver':2,'stomach':3}
        manifest={o:{'train':list(range(n))} for o,n in counts.items()}
        weights=organ_loss_weights(manifest)
        self.assertAlmostEqual(sum(counts[o]*weights[o] for o in counts),10.)
        values=torch.tensor([.2,.4,.6,.1,.3,.7,.8,.3,.4,.9],requires_grad=True)
        weighted=(values[:5].sum()*weights['colon']+values[5:7].sum()*weights['liver']+values[7:].sum()*weights['stomach'])/10
        direct=.1*values[:5].mean()+.2*values[5:7].mean()+.7*values[7:].mean()
        self.assertTrue(torch.allclose(weighted,direct))
        a=torch.autograd.grad(weighted,values,retain_graph=True)[0]
        b=torch.autograd.grad(direct,values)[0]
        self.assertTrue(torch.allclose(a,b))
        self.assertEqual(len(set(epoch_order(manifest,1))),10)

    def test_real_split_weights_use_train_only(self):
        manifest={o:{'train':[None]*n, 'val':[None]*9999} for o,n in zip(('colon','liver','stomach'),(5004,1820,1260))}
        weights=organ_loss_weights(manifest)
        for o,p in zip(manifest,(.1,.2,.7)):
            self.assertAlmostEqual(len(manifest[o]['train'])*weights[o]/8084,p)


if __name__=='__main__': unittest.main()
