"""One matched grouped-Val test: teacher loss fraction 0.4 versus Round313's 0.2."""
import argparse
import os
import sys
import traceback
from pathlib import Path
import torch
import round308 as recipe
from round300 import inventory
from round314 import read


def main(smoke=False,round_id=315):
    assert round_id in (315,316,317,318,320,322,323,326)
    os.chdir(recipe.ROOT); torch.set_num_threads(4)
    split=inventory(Path('dataset')); cfg=read('runs/round307/config.json')
    baseline_id=318 if round_id in (320,322,323,326) else 313
    baseline=Path(f'runs/round{baseline_id}'); reference=read(baseline/'config.json')
    assert read(baseline/'split.json')==split
    initial=Path('runs/round307/stage1_final.pt')
    assert recipe.digest(initial)==reference['initial_checkpoint_sha256']
    encoder_path=Path('runs/round302_pretrain/final_encoder.pt')
    assert recipe.digest(encoder_path)==reference['domain_encoder_sha256']
    domain=torch.load(encoder_path,map_location='cpu',weights_only=False)
    assert domain['split']==split and domain['epoch']==800
    root=Path(f'runs/round{round_id}'+('_smoke' if smoke else ''))
    recipe.fit(root,split,cfg,15 if smoke else 28,domain['encoder'],initial=initial,
               targets=Path('runs/round308/targets'),validate=True,smoke=smoke,
               consistency_weight=.4 if round_id==326 else (.2 if round_id in (318,320,322,323) else .1),distillation_weight=.4 if round_id==315 else .2,
               organ_objective=(.1,.1,.8) if round_id==316 else None,
               stage2_lr_scale=.5 if round_id==317 else 1.,stratified_updates=round_id==320,
               marker_heads=round_id==322,paired_crop=round_id==323)
    recipe.audit_fit(root,split,15 if smoke else 28,initial=True,smoke=smoke)
    actual=read(root/'config.json'); actual.update(endpoint=28,smoke=False)
    allowed={315:('distillation_weight',),316:('organ_loss_weights','organ_loss_objective'),
             317:('stage_lrs','stage2_lr_scale'),318:('training_geometry_consistency_weight',),320:('update_order',),322:('model',),323:('training_paired_crop',),326:('training_geometry_consistency_weight',)}[round_id]
    assert {k:v for k,v in actual.items() if k not in allowed}=={
        k:v for k,v in reference.items() if k not in allowed}
    best=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    val=read(root/f"validation_{best['epoch']:03d}.json")
    baseline_score=read(baseline/'complete.json')['best_validation_score']
    report=Path(f'reports/round{round_id}'+('_smoke' if smoke else '')+'_decision.json')
    recipe.write_json(report,{
        f'baseline_round{baseline_id}':baseline_score,f'candidate_round{round_id}':val['weighted_score'],
        'difference':val['weighted_score']-baseline_score,'selected_epoch':best['epoch'],
        'organs':val['organs'],'automatic_refit':False,
        'limitation':'one grouped split; review organ/ROI changes and trajectory before refit'})
    assert read(report)['candidate_round'+str(round_id)]==val['weighted_score']
    if smoke: print(f'Round{round_id} real-data smoke, matched config and report serialization passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--log',action='store_true')
    p.add_argument('--no-followup',action='store_true',help='Recover/audit a completed run without launching its successor')
    args=p.parse_args(); os.chdir(recipe.ROOT)
    if args.log:
        sys.stdout=open('reports/round315.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round315.err','a',encoding='utf-8',buffering=1)
    try:
        main(args.smoke)
        if not args.smoke and not args.no_followup:
            import subprocess
            print('Starting queued Round316',flush=True)
            subprocess.run([sys.executable,'-u','-X','utf8','tools/round316.py','--log'],check=True)
    except Exception:
        traceback.print_exc(); sys.exit(1)
