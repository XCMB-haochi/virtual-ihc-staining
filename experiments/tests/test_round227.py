import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round227_update_control import clock,command

class BudgetTests(unittest.TestCase):
    def test_clocks_come_from_data_counts(self):
        for count,epochs,endpoint in [(1960,17,15),(1820,18,16)]:
            plan=clock(count)
            self.assertEqual(plan['epochs'],epochs)
            self.assertEqual(plan['endpoint'],endpoint)
            self.assertLess(abs(plan['cycle_update_ratio']-1),.02)
            self.assertLess(abs(plan['endpoint_update_ratio']-1),.033)
    def test_paired_recipe_and_own_stage1(self):
        fold={'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}
        for stage in (1,2):
            base=command(fold,stage,'base');signal=command(fold,stage,'signal')
            expected=[x.replace('round227_fold3_base','round227_fold3_signal') for x in base]
            expected[expected.index('--model-arch')+1]='unext2-fcmae-tiny-signal-aux'
            expected+=['--signal-auxiliary-weight','0.05']
            self.assertEqual(signal,expected)
            self.assertEqual(base[base.index('--epochs')+1],'18')
            if stage==2:self.assertTrue(base[base.index('--init-checkpoint')+1].endswith('epoch_018.pt'))

if __name__=='__main__':unittest.main()
