import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round224_cv5 import command,device_environment
from run_round223_signal_aux import command as original

class ExpansionTests(unittest.TestCase):
    def test_only_output_and_encoder_paths_change(self):
        for name in ('fold1','fold2','fold4'):
            f={'name':name,'val_rois':['ROI010']}
            for stage in (1,2):
                expected=[v.replace('round223_', 'round224_').replace('round207_', 'round216_') for v in original(f,stage)]
                self.assertEqual(command(f,stage),expected)

    def test_physical_device_selection_overrides_wrapper(self):
        self.assertEqual(device_environment(0)['CUDA_VISIBLE_DEVICES'],'0')
        self.assertEqual(device_environment(1)['CUDA_VISIBLE_DEVICES'],'1')

if __name__=='__main__':unittest.main()
