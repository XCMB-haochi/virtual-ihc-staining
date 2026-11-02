"""Fixed supervision/validation targets and the live-launcher continuation."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from round300 import (ORGANS, MARKERS, evaluate, inventory, marker_policy, preload,
                      queued_supervised_handoff, summarize_results)
from round302_supervised import audit_candidate, check_inherited, command_for, load_plan


class FixedMarkerTests(unittest.TestCase):
    def test_only_declared_labels_are_read_and_evaluated(self):
        torch.set_num_threads(4)
        policy=marker_policy(['CD68','CD45RO','CD68'])
        with tempfile.TemporaryDirectory() as directory:
            manifest={}
            for organ in ORGANS:
                folder=Path(directory)/organ/'train'
                # Other marker files deliberately do not exist.
                for marker in ('DAPI',)+policy[organ]:
                    (folder/marker).mkdir(parents=True)
                    Image.fromarray(np.zeros((256,256),dtype=np.uint8)).save(folder/marker/'ROI000_00_00.jpg')
                manifest[organ]={'folder':str(folder),'train':[], 'val':['ROI000_00_00.jpg']}
            cache=preload(manifest,policy)
            for organ in ORGANS:
                self.assertEqual(set(cache[organ,'ROI000_00_00.jpg']),{'DAPI',*policy[organ]})

            class Recorder(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.tasks=[]

                def forward(self,image,marker):
                    self.tasks.append(marker.item())
                    return torch.zeros_like(image)

            model=Recorder()
            result=evaluate(model,manifest,cache,policy=policy,device='cpu')
            self.assertEqual(model.tasks,[0,1,0])
            self.assertEqual(result['selected_markers'],dict(zip(ORGANS,['CD68','CD45RO','CD68'])))
            self.assertAlmostEqual(result['weighted_score'],100.)
            for organ in ORGANS:
                self.assertEqual(set(result['organs'][organ]),set(policy[organ]))

    def test_multimarker_score_includes_all_supervised_targets(self):
        results={organ:{marker:{'score':80. if marker=='CD68' else 20.} for marker in MARKERS} for organ in ORGANS}
        score=summarize_results(results,marker_policy())
        self.assertAlmostEqual(score['weighted_score'],35.)
        self.assertEqual(score['selected_markers'],{organ:list(MARKERS) for organ in ORGANS})

    def test_registered_commands_and_legacy_arguments(self):
        plan=load_plan(ROOT/'configs/round302_supervised.json')
        commands=[command_for(plan,job) for job in plan['jobs']]
        self.assertEqual(commands[0][-4:],['--organ-markers','CD68','CD45RO','CD68'])
        self.assertEqual(commands[1][-4:],['--organ-markers','CD68','CD68','CD68'])
        self.assertNotEqual(commands[0][commands[0].index('--output')+1],commands[1][commands[1].index('--output')+1])
        self.assertEqual(commands[0][commands[0].index('--domain-encoder')+1],commands[1][commands[1].index('--domain-encoder')+1])
        args=argparse.Namespace(output=ROOT/plan['output_root'],domain_encoder=ROOT/plan['domain_encoder'],
                                public=ROOT/plan['public'],data=ROOT/plan['data'],epochs=192,validation_every=16)
        check_inherited(plan,args)
        args.epochs=48
        with self.assertRaises(ValueError): check_inherited(plan,args)

    def test_child_data_argument_preserves_pretraining_manifest(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            data=Path(directory).relative_to(ROOT)
            for organ in ORGANS:
                for marker in ('DAPI',)+MARKERS:
                    folder=data/organ/'train'/marker; folder.mkdir(parents=True)
                    for roi in range(5): (folder/f'ROI{roi:03d}_00_00.jpg').touch()
            pretraining=inventory(data)
            plan=load_plan(ROOT/'configs/round302_supervised.json'); plan['data']=str(data)
            command=command_for(plan,plan['jobs'][0])
            self.assertEqual(pretraining,inventory(Path(command[command.index('--data')+1])))

    def test_queued_entry_routes_to_plan_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'supervised_handoff.json').write_text(json.dumps({
                'kind':'round302-fixed-marker-comparison','plan':'configs/round302_supervised.json'}))
            args=argparse.Namespace(output=output,smoke=False,resume=False,audit=False,
                                    organ_markers=None,check_handoff=True)
            with patch('round302_supervised.run_plan') as dispatch:
                self.assertTrue(queued_supervised_handoff(args))
            dispatch.assert_called_once_with(Path('configs/round302_supervised.json'),check_only=True,inherited=args)
            args.organ_markers=['CD68']*3
            with self.assertRaises(ValueError): queued_supervised_handoff(args)

    def test_candidate_audit_accepts_complete_run_and_rejects_wrong_label_counts(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            root=Path(directory); plan=load_plan(ROOT/'configs/round302_supervised.json')
            plan['output_root']=str(root.relative_to(ROOT)); job=plan['jobs'][0]
            output=root/'smoke_mixed'; output.mkdir()
            config={'seed':2026,'epochs':192,'validation_every':16,'batch':1,
                    'organ_probabilities':[.1,.2,.7],'domain_encoder_sha256':'fixture',
                    'smoke':True,'supervised_markers':{o:[job['organ_markers'][o]] for o in ORGANS}}
            validation={'weighted_score':10.,'selected_markers':job['organ_markers'],
                        'organs':{o:{job['organ_markers'][o]:{'score':10.}} for o in ORGANS}}
            row={'epoch':1,'train_loss':.1,'validation_score':10.,
                 'sampled_targets':{'colon':{'CD68':10},'liver':{'CD45RO':10},'stomach':{'CD68':12}}}
            for filename,payload in [('config.json',config),('validation_001.json',validation),
                                     ('complete.json',{'epochs':1,'best_validation_score':10.})]:
                (output/filename).write_text(json.dumps(payload))
            (output/'metrics.jsonl').write_text(json.dumps(row)+'\n')
            torch.save({'config':config,'markers':job['organ_markers'],'epoch':1,
                        'model':{'weight':torch.tensor([1.])}},output/'best.pt')
            torch.save({'epoch':1,'optimizer':{'state':{0:{'step':torch.tensor(4.)}}}},output/'last.pt')
            self.assertEqual(audit_candidate(plan,job,'fixture',smoke=True)['optimizer_steps'],4)
            row['sampled_targets']['liver']={'CD68':10}
            (output/'metrics.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaises(AssertionError): audit_candidate(plan,job,'fixture',smoke=True)


if __name__=='__main__': unittest.main()
