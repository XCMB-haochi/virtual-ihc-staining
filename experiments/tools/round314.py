"""Corrected full-data stage2 refit; reuse the verified Round309 full stage1."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
import torch
import round308 as recipe
from round300 import inventory


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_stage1(full, cfg):
    source=Path('runs/round309')
    assert read(source/'split.json')==full, 'Stage1 source must contain ALL official Train'
    source_cfg=read(source/'config.json')
    for key in ('seed','stages','epochs_per_stage','accumulation','batch','stage_lrs',
                'encoder_stage_decay','minimum_lr','weight_decay','clip','precision',
                'loss','sampling','supervised_markers','domain_encoder_sha256','organ_loss_objective'):
        assert source_cfg[key]==cfg[key], key
    assert source_cfg['full_train'] and not source_cfg['smoke']
    assert not source_cfg.get('training_geometry_consistency_weight',0)
    rows=[json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row['epoch']<=14]
    assert [r['epoch'] for r in rows]==list(range(1,15))
    counts={o:len(full[o]['train']) for o in recipe.ORGANS}
    assert sum(counts.values())==10356
    assert all(r['samples']==counts and r['updates']==648 and r['stage']==1 for r in rows)
    initial=source/'stage1_final.pt'
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    assert saved['epoch']==14
    assert all(k in saved for k in ('python_rng','numpy_rng','torch_rng','cuda_rng'))
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    return initial


def main(smoke=False,round_id=314):
    assert round_id in (314,319,321)
    selection_round={314:313,319:318,321:320}[round_id]
    os.chdir(recipe.ROOT); torch.set_num_threads(4)
    grouped=inventory(Path('dataset')); full=recipe.all_train_manifest(grouped)
    cfg=read('runs/round307/config.json')
    initial=verify_stage1(full,cfg)
    domain_path=Path('runs/round302_pretrain/final_encoder.pt')
    assert recipe.digest(domain_path)==cfg['domain_encoder_sha256']
    domain=torch.load(domain_path,map_location='cpu',weights_only=False)
    assert domain['split']==grouped and domain['epoch']==800
    decision=read(f'reports/round{selection_round}_decision.json')
    assert (decision['candidate_for_review'] if round_id==314 else decision['difference']>0)
    assert decision['selected_epoch']==26
    targets=Path('runs/round308/full_targets')
    items=read(targets/'manifest.json')['items']
    assert len(items)==10356 and {(r['organ'],r['name']) for r in items}=={
        (o,n) for o in recipe.ORGANS for n in full[o]['train']}
    root=Path(f'runs/round{round_id}'+('_repair_smoke' if smoke else ''))
    endpoint=15 if smoke else 26
    recipe.fit(root,full,cfg,endpoint,domain['encoder'],initial=initial,targets=targets,
               smoke=smoke,consistency_weight=.1 if round_id==314 else .2,
               stratified_updates=round_id==321)
    recipe.audit_fit(root,full,endpoint,initial=True,smoke=smoke)
    recipe.write_json(root/'selection_split.json',grouped)
    recipe.write_json(root/'stage1_provenance.json',{
        'source':str(initial),'sha256':recipe.digest(initial),'full_train_images':10356,
        'stage1_epochs':14,'stage1_updates':9072,'stage2_epochs':endpoint-14,
        'recipe':f'Round309 full stage1 + Round{selection_round} stage2 consistency',
        'target_manifest_sha256':recipe.digest(targets/'manifest.json')})
    if not smoke:
        from package_round302 import main as package
        package(round_id)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--log',action='store_true')
    args=p.parse_args(); os.chdir(recipe.ROOT)
    if args.log:
        sys.stdout=open('reports/round314.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round314.err','a',encoding='utf-8',buffering=1)
    try: main(args.smoke)
    except Exception:
        traceback.print_exc(); sys.exit(1)
