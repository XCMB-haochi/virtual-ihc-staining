import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from paired_crop import draw_crop,apply_crop
from ihc_torch import apply_d4_transform
from ihc_single import SingleMarkerStainer


class CropTests(unittest.TestCase):
    def test_exact_pairing_and_original_preservation(self):
        source=torch.arange(256*256).reshape(1,1,256,256).float()
        original=source.clone()
        for view in range(8):
            transformed=apply_d4_transform(source,view)
            for top,left in ((0,0),(16,16),(3,11)):
                box=top,left,240
                actual=apply_crop(transformed,box)
                self.assertTrue(torch.equal(actual,transformed[...,top:top+240,left:left+240]))
                self.assertTrue(torch.equal(apply_crop(transformed+100000,box)-100000,actual))
                self.assertTrue(torch.equal(apply_crop(transformed*2,box),actual*2))
                self.assertEqual(tuple(actual.shape),(1,1,240,240))
        self.assertTrue(torch.equal(source,original))
        self.assertIs(apply_crop(source,None),source)
        self.assertIsNone(apply_crop(None,(0,0,240)))

    def test_rng_replay_and_bounds(self):
        global_state=random.getstate(); torch_state=torch.get_rng_state().clone()
        a=random.Random(2026+200000*15); b=random.Random(2026+200000*15)
        boxes=[draw_crop(a,256,256) for _ in range(1000)]
        self.assertEqual(boxes,[draw_crop(b,256,256) for _ in range(1000)])
        self.assertEqual(global_state,random.getstate())
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertTrue(any(box is None for box in boxes))
        self.assertTrue(any(box is not None for box in boxes))
        self.assertTrue(all(box is None or (0<=box[0]<=16 and 0<=box[1]<=16 and box[2]==240) for box in boxes))
        with self.assertRaises(ValueError): draw_crop(a,224,224)
        with self.assertRaises(ValueError): apply_crop(torch.zeros(1,1,256,256),(17,0,240))

    def test_model_supports_training_crop_and_full_validation(self):
        torch.set_num_threads(2); torch.manual_seed(2026)
        model=SingleMarkerStainer().eval()
        with torch.no_grad():
            for size in (240,256):
                x=torch.rand(1,1,size,size)
                result=model(x,torch.tensor([0]))
                self.assertEqual(tuple(result.shape),tuple(x.shape))
                self.assertTrue(torch.isfinite(result).all())


if __name__=='__main__': unittest.main()
