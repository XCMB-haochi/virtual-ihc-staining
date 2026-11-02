"""Grouped-Val test with marker-specific heads active from stage1."""
import argparse
import os
import sys
import traceback
import json
from pathlib import Path
import torch
import round308 as recipe
from round300 import inventory
from round314 import read


def main(smoke=False):
    os.chdir(recipe.ROOT); torch.set_num_threads(4)
    split=inventory(Path('dataset')); cfg=read('runs/round307/config.json')
    reference=read('runs/round318/config.json')
    initial=None
    encoder_path=Path('runs/round302_pretrain/final_encoder.pt')
    assert recipe.digest(encoder_path)==reference['domain_encoder_sha256']
    domain=torch.load(encoder_path,map_location='cpu',weights_only=False)
    assert domain['split']==split and domain['epoch']==800
    root=Path('runs/round325_smoke' if smoke else 'runs/round325')
    recipe.fit(root,split,cfg,15 if smoke else 28,domain['encoder'],initial=initial,
               targets=Path('runs/round308/targets'),validate=True,smoke=smoke,
               consistency_weight=.2,distillation_weight=.2,marker_heads_from_start=True)
    recipe.audit_fit(root,split,15 if smoke else 28,initial=False,smoke=smoke)
    actual=read(root/'config.json'); actual.update(endpoint=28,smoke=False)
    rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    assert all(('geometry_consistency_loss' in row)==(row['epoch']>=15) for row in rows)
    assert actual['consistency_start_epoch']==actual['selection_start_epoch']==15
    allowed=('model','initial_checkpoint_sha256','consistency_start_epoch','selection_start_epoch')
    assert {k:v for k,v in actual.items() if k not in allowed}=={
        k:v for k,v in reference.items() if k not in allowed}
    best=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    assert best['epoch']>=15
    val=read(root/f"validation_{best['epoch']:03d}.json")
    baseline=read('runs/round318/complete.json')['best_validation_score']
    report=Path(f'reports/round325'+('_smoke' if smoke else '')+'_decision.json')
    recipe.write_json(report,{'baseline_round318':baseline,'candidate_round325':val['weighted_score'],
        'difference':val['weighted_score']-baseline,'selected_epoch':best['epoch'],
        'organs':val['organs'],'automatic_refit':False,
        'limitation':'one grouped split; from-stage1 head split; no Test use'})
    assert read(report)['candidate_round325']==val['weighted_score']
    if smoke: print('Round325 from-stage1 heads smoke and report checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--log',action='store_true')
    args=p.parse_args(); os.chdir(recipe.ROOT)
    if args.log:
        sys.stdout=open('reports/round325.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round325.err','a',encoding='utf-8',buffering=1)
    try: main(args.smoke)
    except Exception:
        traceback.print_exc(); sys.exit(1)
