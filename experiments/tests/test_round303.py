import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round303 import optimizer_for, epoch_order, SingleMarkerStainer, all_train_manifest


class RecipeTests(unittest.TestCase):
    def test_stage_rates_and_parameter_coverage(self):
        model=SingleMarkerStainer()
        for stage,lr in ((1,3e-4),(2,1e-4)):
            optimizer=optimizer_for(model,stage)
            expected=[lr]+[lr*.8**e for e in (4,3,2,1,0)]
            self.assertEqual([g['lr'] for g in optimizer.param_groups],expected)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,14,eta_min=0.)
            for _ in range(14): optimizer.step(); scheduler.step()
            self.assertTrue(all(g['lr']==0 for g in optimizer.param_groups))

    def test_each_image_once_no_validation(self):
        manifest={o:{'train':['ROI0_1.jpg','ROI0_2.jpg'],'val':['ROI1_1.jpg']}
                  for o in ('colon','liver','stomach')}
        order=epoch_order(manifest,1)
        self.assertEqual(len(order),6)
        self.assertEqual(set(order),{(o,n) for o in manifest for n in manifest[o]['train']})
        self.assertEqual(order,epoch_order(manifest,1))

    def test_full_refit_preserves_original_split_and_excludes_test(self):
        grouped={o:{'train':['ROI0_1.jpg'], 'val':['ROI1_1.jpg'],
                    'train_rois':['ROI0'],'val_rois':['ROI1'],'folder':'train'}
                 for o in ('colon','liver','stomach')}
        full=all_train_manifest(grouped)
        for o in grouped:
            self.assertEqual(grouped[o]['val'],['ROI1_1.jpg'])
            self.assertEqual(full[o]['train'],['ROI0_1.jpg','ROI1_1.jpg'])
            self.assertEqual(full[o]['val'],[])
            self.assertEqual(full[o]['train_rois'],['ROI0','ROI1'])
        self.assertEqual(len(epoch_order(full,1)),6)

    def test_epoch27_uses_14_epoch_second_stage_horizon(self):
        model=SingleMarkerStainer(); optimizer=optimizer_for(model,2)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,14,eta_min=0.)
        for _ in range(12): optimizer.step(); scheduler.step()
        import math
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],1e-4*(1+math.cos(math.pi*12/14))/2)


if __name__=='__main__': unittest.main()
