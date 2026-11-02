import copy
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from round308 import fixed_folds, teacher_split, distilled_loss, assign_heldout_rois
from round314 import make_full_targets


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.grouped={o:{'train':[f'ROI{r:03d}_{n}.jpg' for r in range(10) for n in range(r+1)],
                         'train_rois':[f'ROI{r:03d}' for r in range(10)],
                         'val':['ROI099_1.jpg'],'val_rois':['ROI099']}
                      for o in ('colon','liver','stomach')}

    def test_grouped_fold_targets_are_unique_and_teacher_never_sees_val(self):
        original=copy.deepcopy(self.grouped)
        state=random.getstate(); folds=fixed_folds(self.grouped)
        self.assertEqual(state,random.getstate())
        self.assertEqual(folds,fixed_folds(self.grouped))
        for o in self.grouped:
            targets=[n for f in folds for n in f[o]]
            self.assertEqual(len(targets),len(set(targets)))
            self.assertEqual(set(targets),set(self.grouped[o]['train']))
            memberships={}
            for i,fold in enumerate(folds):
                split=teacher_split(self.grouped,fold)
                self.assertFalse(set(split[o]['train'])&set(fold[o]))
                self.assertFalse(set(split[o]['train'])&set(self.grouped[o]['val']))
                self.assertEqual(split[o]['val'],[])
                for name in fold[o]: memberships.setdefault(name.split('_')[0],set()).add(i)
            self.assertTrue(all(len(s)==1 for s in memberships.values()))
        self.assertEqual(self.grouped,original)

    def test_teacher_mix_keeps_ground_truth_and_gradient_weights(self):
        pred=torch.tensor([.2,.7],requires_grad=True); gt=torch.tensor([.1,.9]); teacher=torch.tensor([.3,.6])
        original=gt.clone(); loss=distilled_loss(pred,gt,teacher,torch.nn.MSELoss())
        loss.backward()
        expected=.8*(pred.detach()-gt)+.2*(pred.detach()-teacher)
        self.assertTrue(torch.allclose(pred.grad,expected))
        self.assertTrue(torch.equal(gt,original))
        self.assertTrue(torch.equal(distilled_loss(pred,gt,None,torch.nn.MSELoss()),torch.nn.functional.mse_loss(pred,gt)))

    def test_heldout_validation_rois_are_assigned_whole_exactly_once(self):
        grouped=copy.deepcopy(self.grouped)
        for organ in grouped:
            grouped[organ]['val']=[f'ROI{r+10:03d}_{n}.jpg' for r in range(4) for n in range(r+1)]
            grouped[organ]['val_rois']=[f'ROI{r+10:03d}' for r in range(4)]
        assignments=assign_heldout_rois(grouped,5)
        for organ in grouped:
            names=[n for fold in assignments for n in fold[organ]]
            self.assertEqual(len(names),len(set(names)))
            self.assertEqual(set(names),set(grouped[organ]['val']))
            for roi in grouped[organ]['val_rois']:
                slots=[i for i,fold in enumerate(assignments) if any(n.split('_')[0]==roi for n in fold[organ])]
                self.assertEqual(len(slots),1)

    def test_full_refit_targets_cover_train_and_heldout_val(self):
        from unittest.mock import patch
        with __import__('tempfile').TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir(); output=root/'full'
            records=[]
            for organ in ('colon','liver','stomach'):
                (source/organ).mkdir()
                for name in self.grouped[organ]['train']:
                    (source/organ/(name+'.png')).write_bytes(b'train')
                    records.append({'organ':organ,'name':name,'fold':0,'teacher_sha256':'trainhash'})
            (source/'manifest.json').write_text(__import__('json').dumps({'items':records}))
            def fake_predict(checkpoint,grouped,names,folder,fold,items):
                for organ in names:
                    for name in names[organ]:
                        target=folder/organ/(name+'.png'); target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(b'val')
                        items[(organ,name)]={'organ':organ,'name':name,'fold':fold,'teacher_sha256':'valhash'}
            with patch('round308.predict_targets',side_effect=fake_predict), patch('round308.audit_targets') as audit:
                manifest=make_full_targets(self.grouped,[{} for _ in range(5)],source,output)
            expected={(o,n) for o in self.grouped for n in self.grouped[o]['train']+self.grouped[o]['val']}
            self.assertEqual({(x['organ'],x['name']) for x in manifest['items']},expected)
            self.assertTrue(all((output/o/(n+'.png')).exists() for o,n in expected))
            audit.assert_called_once()


if __name__=='__main__': unittest.main()
