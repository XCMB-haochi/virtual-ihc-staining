import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_round226_clock_audit import select_clock

class ClockTests(unittest.TestCase):
    def test_weighted_clock_and_omission(self):
        def rows(values,n):
            return [{'validation':{'count':n,'markers':{'CD45RO':{'competition_score_mean_image_psnr':v}}}} for v in values]
        data={'a':rows([2,1]+[0]*12,3),'b':rows([0,3]+[0]*12,4)}
        self.assertEqual(select_clock(data)['epoch'],2)
        self.assertEqual(select_clock(data,'b')['epoch'],1)
        self.assertEqual(select_clock({'a':rows([1]*14,1)})['epoch'],1)

if __name__=='__main__':unittest.main()
