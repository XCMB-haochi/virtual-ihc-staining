import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round214_signal import command_for
from run_round211_batch16 import command_for as baseline

class SignalTests(unittest.TestCase):
    def test_only_stage2_weighting_and_output_path_change(self):
        for name in ['fold0','fold3']:
            fold={'name':name,'val_rois':['ROI005']}
            expected=baseline(fold,2,'control',None,None)
            cmd,_=command_for(fold)
            expected[expected.index('--output-dir')+1]=f'runs/round214_{name}_stage2'
            expected+=['--target-l1-weight-strength','2.0']
            self.assertEqual(cmd,expected)
            self.assertEqual(cmd[cmd.index('--init-checkpoint')+1],f'runs/round211_{name}_control_stage1/epochs/epoch_014.pt')

if __name__=='__main__':unittest.main()
