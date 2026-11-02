"""Contract checks for the new single-image inference and grouped split."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from round300 import inventory, gaussian_ssim, SingleMarkerStainer, MARKERS


class Round300Tests(unittest.TestCase):
    def test_group_split_is_disjoint_deterministic_and_organ_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for organ in ('colon','liver','stomach'):
                for marker in ('DAPI',)+MARKERS:
                    folder=root/organ/'train'/marker; folder.mkdir(parents=True)
                    for roi in range(10):
                        for column in range(2):
                            (folder/f'ROI{roi:03d}_00_{column:02d}.jpg').touch()
            a=inventory(root); self.assertEqual(a,inventory(root))
            for spec in a.values():
                self.assertFalse(set(spec['train_rois']) & set(spec['val_rois']))
                self.assertEqual(len(spec['train'])+len(spec['val']),20)
                self.assertTrue(all(n.split('_')[0] in spec['val_rois'] for n in spec['val']))

    def test_single_target_and_no_batch_or_context_input(self):
        torch.set_num_threads(4); torch.manual_seed(2026)
        model=SingleMarkerStainer().eval(); x=torch.rand(1,1,256,256)
        with torch.inference_mode():
            a=model(x,torch.tensor([0])); b=model(x,torch.tensor([1]))
            self.assertEqual(a.shape,x.shape)
            self.assertTrue(torch.isfinite(a).all())
            self.assertFalse(torch.equal(a,b))
            with self.assertRaises(ValueError): model(x.repeat(2,1,1,1),torch.tensor([0,1]))
            with self.assertRaises(TypeError): model(x,torch.tensor([0]),organ='colon')
        self.assertFalse(any('thumbnail' in n or 'condition_film' in n for n,_ in model.named_modules()))

    def test_gaussian_matches_independent_reference(self):
        from skimage.metrics import structural_similarity
        rng=np.random.default_rng(2026)
        x=rng.random((64,64),dtype=np.float32); y=np.clip(x+.03*rng.normal(size=x.shape),0,1).astype('float32')
        actual=gaussian_ssim(torch.from_numpy(x)[None,None],torch.from_numpy(y)[None,None]).item()
        expected=structural_similarity(x,y,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)
        self.assertAlmostEqual(actual,expected,places=5)


if __name__=='__main__': unittest.main()
