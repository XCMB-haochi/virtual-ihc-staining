import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from round305 import recipe_args, choose


class QueuedComparisonTests(unittest.TestCase):
    def test_decision_uses_only_finite_val_and_retains_tie(self):
        self.assertFalse(choose(76.09,76.09))
        self.assertFalse(choose(76.09,75.9))
        self.assertTrue(choose(76.09,76.10))
        with self.assertRaises(ValueError): choose(76.09,float('nan'))

    def test_refit_can_use_either_stage_without_changing_markers(self):
        for epoch in (8,14,15,28):
            args=recipe_args(full=True,endpoint=epoch)
            self.assertEqual(args.endpoint,epoch)
            self.assertTrue(args.full_train)
            self.assertEqual(args.organ_markers,['CD68']*3)
            self.assertEqual(args.run_prefix,'round306')
        self.assertFalse(recipe_args().full_train)
        self.assertEqual(recipe_args().run_prefix,'round305')


if __name__=='__main__': unittest.main()
