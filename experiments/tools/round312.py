"""Matched second-stage Gaussian-valid objective with the existing teacher cache."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
import torch
from round308 import ROOT, fit, audit_fit, digest
from round300 import inventory
from round305 import control_record


def main(smoke=False,round_id=312):
    torch.set_num_threads(4)
    split=inventory(Path('dataset'))
    base=Path('runs/round307')
    cfg=json.loads((base/'config.json').read_text(encoding='utf-8'))
    domain=torch.load('runs/round302_pretrain/final_encoder.pt',map_location='cpu',weights_only=False)
    assert domain['split']==split and domain['epoch']==800
    assert digest('runs/round302_pretrain/final_encoder.pt')==cfg['domain_encoder_sha256']
    baseline_root=Path('runs/round308/student')
    assert json.loads((baseline_root/'split.json').read_text(encoding='utf-8'))==split
    baseline_cfg=json.loads((baseline_root/'config.json').read_text(encoding='utf-8'))
    initial=base/'stage1_final.pt'; targets=Path('runs/round308/targets')
    assert digest(initial)==baseline_cfg['initial_checkpoint_sha256']
    manifest=json.loads((targets/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['original_split']==split and len(manifest['items'])==8084
    assert round_id in (312,313)
    root=Path(f'runs/round{round_id}'+('_smoke' if smoke else ''))
    endpoint=15 if smoke else 28
    fit(root,split,cfg,endpoint,domain['encoder'],initial=initial,targets=targets,
        validate=True,smoke=smoke,training_ssim='gaussian11_valid' if round_id==312 else 'uniform7',
        consistency_weight=.1 if round_id==313 else 0.)
    audit_fit(root,split,endpoint,initial=True,smoke=smoke)
    current=json.loads((root/'config.json').read_text(encoding='utf-8'))
    current['endpoint']=28; current['smoke']=False
    allowed=('training_ssim','loss') if round_id==312 else ('training_geometry_consistency_weight',)
    assert {k:v for k,v in current.items() if k not in allowed}=={k:v for k,v in baseline_cfg.items() if k not in allowed}
    checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    validation=json.loads((root/f"validation_{checkpoint['epoch']:03d}.json").read_text(encoding='utf-8'))
    if smoke:
        print('Matched configuration and real stage2 smoke passed',flush=True); return
    baseline=json.loads((baseline_root/'complete.json').read_text(encoding='utf-8'))['best_validation_score']
    rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert validation['weighted_score']==max(r['validation_score'] for r in rows)
    result={'baseline_round308':baseline,f'candidate_round{round_id}':validation['weighted_score'],
            'difference':validation['weighted_score']-baseline,'selected_epoch':checkpoint['epoch'],
            'candidate_for_review':validation['weighted_score']>baseline,'organs':validation['organs'],
            'selection':'same outer grouped Val, best of14 second-stage endpoints; no Test use',
            'limitations':'one split; only3 Stomach ROIs; no automatic full refit or submission'}
    control_record(Path(f'reports/round{round_id}_decision.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--log',action='store_true')
    parser.add_argument('--round',dest='round_id',type=int,choices=(312,313),default=312)
    args=parser.parse_args(); os.chdir(ROOT)
    if args.log:
        sys.stdout=open(f'reports/round{args.round_id}.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open(f'reports/round{args.round_id}.err','a',encoding='utf-8',buffering=1)
    try: main(args.smoke,args.round_id)
    except Exception:
        traceback.print_exc(); sys.exit(1)
