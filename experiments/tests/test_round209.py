import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round209_cd68 import command_for

class MatchedCD68Tests(unittest.TestCase):
    def test_initialization_is_only_recipe_difference(self):
        fold={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
        for stage in (1,2):
            public=command_for(fold,stage,'public')
            domain=command_for(fold,stage,'domain')
            expected=[x.replace('round209_fold3_public','round209_fold3_domain') for x in public]
            if stage==1:
                expected[expected.index('--fcmae-encoder-init-checkpoint')+1]='runs/round207_fold3_fcmae800/final_encoder.pt'
            self.assertEqual(domain,expected)
            self.assertEqual(domain[domain.index('--selection-marker')+1],'CD68')
            self.assertEqual(domain[domain.index('--epochs')+1],'14')
            self.assertEqual(domain[domain.index('--seed')+1],'2026')

if __name__=='__main__': unittest.main()
