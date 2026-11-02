import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from PIL import Image
from run_round243_translation_tta import assemble,tiles,materialize
from ihc_torch import shift_array_edge,save_grayscale_rgb

class RoiTranslationTests(unittest.TestCase):
    def test_submission_encoding_matches_validation_units(self):
        import io
        values=np.linspace(0,255,256*256,dtype=np.float32).reshape(256,256)
        expected=io.BytesIO()
        Image.fromarray(np.rint(values).astype(np.uint8)).convert('RGB').save(expected,format='JPEG',quality=95)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'out.jpg'
            save_grayscale_rgb(values/255.0,p,quality=95)
            self.assertEqual(p.read_bytes(),expected.getvalue())

    def test_translation_crosses_tile_seams_and_inverse_coordinates(self):
        names=[f'ROI001_{r}_{c}.jpg' for r in range(2) for c in range(2)]
        rng=np.random.default_rng(2026);original=rng.integers(0,256,(512,512),dtype=np.uint8)
        self.assertTrue(np.array_equal(assemble(names,tiles(names,original)),original))
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/'original';dst=Path(folder)/'shifted';src.mkdir()
            for n,a in tiles(names,original).items():Image.fromarray(a).save(src/n,format='PNG')
            materialize(src,names,dst)
            moved=assemble(names,{n:np.array(Image.open(dst/n)) for n in names})
            np.testing.assert_array_equal(moved[8:,8:],original[:-8,:-8])
            restored=shift_array_edge(moved,(8,8))
            np.testing.assert_array_equal(restored[:-8,:-8],original[:-8,:-8])
            # Lost finite-ROI edges are explicitly replicated, not secretly restored.
            np.testing.assert_array_equal(restored[-8:,:-8],np.repeat(original[-9:-8,:-8],8,axis=0))

if __name__=='__main__':unittest.main()
