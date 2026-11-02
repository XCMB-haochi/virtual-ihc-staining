import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_round210_relational import student_command, teacher_command, MODALITIES
from ihc_feature_guidance import load_frozen_multimarker_fcmae_teacher
from ihc_unext2 import unext2_encoder_specification


class RelationalScreenTests(unittest.TestCase):
    def test_teacher_uses_only_fold_train_and_fixed_160_epoch_budget(self):
        fold = {'name': 'fold0', 'val_rois': ['ROI005', 'ROI052', 'ROI087']}
        cmd, _ = teacher_command(fold)
        for key, value in {'--epochs':'160', '--warmup-epochs':'2', '--data-scope':'fold-train',
                           '--seed':'2026', '--pretraining-markers':MODALITIES}.items():
            self.assertEqual(cmd[cmd.index(key)+1], value)

    def test_student_pairs_match_except_guidance_and_paths(self):
        fold = {'name': 'fold3', 'val_rois': ['ROI044']}
        for stage in (1,2):
            control = student_command(fold, stage, 'control')
            candidate = student_command(fold, stage, 'relational', 'teacher.pt', 'a'*64)
            prefix = candidate[:candidate.index('--feature-guidance-checkpoint')]
            self.assertEqual(prefix, [x.replace('fold3_control', 'fold3_relational') for x in control])
            self.assertEqual(candidate[candidate.index('--batch-size')+1], '8')
            self.assertEqual(candidate[candidate.index('--gradient-accumulation-steps')+1], '1')
            self.assertEqual(candidate[candidate.index('--feature-guidance-marker')+1], 'CD45RO')
            self.assertEqual(candidate[candidate.index('--epochs')+1], '14')

    def test_loading_frozen_teacher_preserves_student_rng_and_checks_roi(self):
        saved = torch.nn.Linear(2,3).state_dict()
        payload = {'version':1, 'task':'dapi-fcmae-pretraining', 'epoch':160,
                   'encoder_spec':unext2_encoder_specification('tiny'),
                   'config':{'pretraining_markers':MODALITIES.split(',')},
                   'val_rois':['ROI005'], 'encoder':saved}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'teacher.pt'
            torch.save(payload,path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            torch.manual_seed(2026)
            before = torch.get_rng_state().clone()
            with mock.patch('ihc_feature_guidance.DapiFCMAE', side_effect=lambda _: SimpleNamespace(encoder=torch.nn.Linear(2,3))):
                teacher, _ = load_frozen_multimarker_fcmae_teacher(path,digest,{'ROI005'},torch.device('cpu'))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            with self.assertRaises(ValueError):
                load_frozen_multimarker_fcmae_teacher(path,digest,{'ROI010'},torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
