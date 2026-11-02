import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round220_texture_cv5 import command
from run_round219_texture import command as original

class ExpansionTests(unittest.TestCase):
    def test_expansion_preserves_recipe(self):
        for name in ('fold1','fold2','fold4'):
            fold={'name':name,'val_rois':['ROI010']}
            for stage in (1,2):
                expected=original(fold,stage)
                expected=[x.replace('round219_','round220_').replace('round207_','round216_') for x in expected]
                self.assertEqual(command(fold,stage),expected)

if __name__=='__main__':unittest.main()
