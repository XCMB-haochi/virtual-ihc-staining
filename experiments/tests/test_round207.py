import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from train_dapi_fcmae import initialize_packaged_public_encoder, select_pretraining_names
from ihc_unext2 import unext2_encoder_specification
from run_round207_domain import pretrain_command


class DomainPretrainingTests(unittest.TestCase):
    def test_exact_encoder_including_stem_is_loaded_and_trained_source_rejected(self):
        encoder = torch.nn.Linear(2, 2)
        saved = {k: torch.full_like(v, 0.25) for k, v in encoder.state_dict().items()}
        payload = {'task': 'dapi-fcmae-pretraining', 'epoch': 0, 'val_rois': [],
                   'config': {'data_scope': 'public-only'},
                   'encoder_spec': unext2_encoder_specification('tiny'), 'encoder': saved}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'encoder.pt'
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            initialize_packaged_public_encoder(encoder, path, digest, 'tiny')
            for key, value in encoder.state_dict().items():
                self.assertTrue(torch.equal(value, saved[key]))
            with self.assertRaises(ValueError):
                initialize_packaged_public_encoder(encoder, path, '0' * 64, 'tiny')
            payload['epoch'] = 800
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                initialize_packaged_public_encoder(encoder, path, hashlib.sha256(path.read_bytes()).hexdigest(), 'tiny')

    def test_pretraining_excludes_validation_and_keeps_effective_batch32(self):
        fold = {'name': 'fold0', 'val_rois': ['ROI005']}
        command, _ = pretrain_command(fold)
        self.assertEqual(command[command.index('--epochs') + 1], '800')
        self.assertEqual(command[command.index('--data-scope') + 1], 'fold-train')
        self.assertEqual(command[command.index('--seed') + 1], '2026')
        self.assertEqual(int(command[command.index('--batch-size') + 1]) * int(command[command.index('--gradient-accumulation-steps') + 1]), 32)
        names = ['ROI005_00_00.jpg', 'ROI010_00_00.jpg']
        self.assertEqual(select_pretraining_names(names, {'ROI005'}, 'fold-train'), ['ROI010_00_00.jpg'])


if __name__ == '__main__':
    unittest.main()
