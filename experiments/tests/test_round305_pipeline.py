"""Exercise queue control flow with temporary fixtures, without real training."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import round305


class PipelineTests(unittest.TestCase):
    def exercise(self, score):
        events=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'transfers') as folder:
            old=Path.cwd()
            try:
                os.chdir(folder)
                Path('reports').mkdir()
                base=Path('runs/round303'); base.mkdir(parents=True)
                cfg={'seed':2026,'supervised_markers':{'colon':['CD68'],'liver':['CD45RO'],'stomach':['CD68']}}
                (base/'config.json').write_text(json.dumps(cfg))
                (base/'split.json').write_text('{"same":"split"}')
                (base/'complete.json').write_text(json.dumps({'best_validation_score':76.09}))
                def dependency_finished(_seconds):
                    events.append('dependency-ready')
                    round305.control_record(Path('reports/round304_submission.json'),{'complete':True})
                def training(args):
                    self.assertTrue(Path('reports/round304_submission.json').exists())
                    events.append(args.run_prefix)
                    target=Path('runs')/args.run_prefix; target.mkdir()
                    candidate_cfg=dict(cfg,supervised_markers={o:['CD68'] for o in ('colon','liver','stomach')})
                    (target/'config.json').write_text(json.dumps(candidate_cfg))
                    (target/'split.json').write_bytes((base/'split.json').read_bytes())
                    if args.full_train: self.assertEqual(args.endpoint,22)
                def auditing(name,**kwargs):
                    events.append('audit-'+name)
                    return {'epoch':22,'validation':{'weighted_score':score,'organs':{}}}
                with patch.object(round305.time,'sleep',side_effect=dependency_finished), \
                     patch.object(round305.subprocess,'check_output',return_value='30000\n'), \
                     patch.object(round305,'run',side_effect=training), \
                     patch.object(round305,'audit',side_effect=auditing), \
                     patch('package_round302.main',side_effect=lambda n: events.append('package-'+str(n))):
                    round305.pipeline()
                decision=json.loads(Path('reports/round305_decision.json').read_text())
                completed=json.loads(Path('reports/round305_queue_complete.json').read_text())
                self.assertEqual(decision['refit_cd68'],score>76.09)
                self.assertEqual(completed['full_cd68'],score>76.09)
                self.assertFalse(list(Path('reports').glob('*.writing')))
            finally: os.chdir(old)
        return events

    def test_wait_then_win_trains_and_packages_once(self):
        self.assertEqual(self.exercise(76.2),['dependency-ready','round305','audit-round305','round306','audit-round306','package-306'])

    def test_wait_then_loss_does_not_start_full_refit(self):
        self.assertEqual(self.exercise(75.8),['dependency-ready','round305','audit-round305'])

    def test_wait_then_tie_keeps_existing_package(self):
        self.assertEqual(self.exercise(76.09),['dependency-ready','round305','audit-round305'])


if __name__=='__main__': unittest.main()
