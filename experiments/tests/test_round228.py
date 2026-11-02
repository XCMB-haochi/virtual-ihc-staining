import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round228_budget_cv5 import command
from run_round227_update_control import command as screen

class ExpansionTests(unittest.TestCase):
    def test_only_paths_change_from_registered_budget(self):
        for name,rois in [('fold1',['ROI010','ROI058','ROI093']),('fold4',['ROI045','ROI072','ROI076','ROI081'])]:
            fold={'name':name,'val_rois':rois}
            for stage in (1,2):
                expected=[s.replace('round227_','round228_').replace('round207_','round216_') for s in screen(fold,stage,'base')]
                self.assertEqual(command(fold,stage),expected)
                self.assertNotIn('--signal-auxiliary-weight',expected)

if __name__=='__main__':unittest.main()
