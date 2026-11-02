"""Fresh all-Train supervised refit of the Val-selected paired-crop recipe."""
import argparse
import os
import sys
import traceback
from pathlib import Path
import torch
import round308 as recipe
from round300 import inventory
from round314 import read


def verify_refit(run, full, endpoint, smoke=False):
    first=Path('runs/round324_stage1_smoke' if smoke else 'runs/round324_stage1')
    first_cfg=read(first/'config.json'); cfg=read(run/'config.json')
    assert first_cfg['initial_checkpoint_sha256'] is None
    assert first_cfg['full_train'] and first_cfg['smoke']==smoke
    assert first_cfg['distillation_weight']==0 and first_cfg['teacher_cache'] is None
    assert first_cfg.get('training_geometry_consistency_weight',0)==0
    assert 'training_paired_crop' not in first_cfg
    assert read(first/'split.json')==read(run/'split.json')==full
    assert read(first/'complete.json')['epochs']==14
    assert read(first/'audit.json')['updates']==14*(2 if smoke else 648)
    initial=first/'stage1_final.pt'
    assert cfg['initial_checkpoint_sha256']==recipe.digest(initial)
    assert cfg['full_train'] and cfg['smoke']==smoke and cfg['endpoint']==endpoint
    assert cfg['distillation_weight']==.2 and cfg['training_geometry_consistency_weight']==.2
    assert cfg['training_paired_crop']=={'size':240,'probability':.5,'stage':2,
        'seed_rule':'2026+200000*epoch','validation_crop':False}
    assert cfg['domain_encoder_sha256']==first_cfg['domain_encoder_sha256']
    assert read(run/'complete.json')['epochs']==endpoint
    provenance=read(run/'stage1_provenance.json')
    assert provenance['source']==str(initial) and provenance['sha256']==recipe.digest(initial)
    assert provenance['full_train_images']==10356 and provenance['stage1_epochs']==14
    assert provenance['fresh_supervised_refit'] is True


def main(smoke=False):
    os.chdir(recipe.ROOT); torch.set_num_threads(4)
    grouped=inventory(Path('dataset')); full=recipe.all_train_manifest(grouped)
    assert sum(len(s['train']) for s in full.values())==10356
    assert grouped==read('runs/round323/split.json')
    decision=read('reports/round323_decision.json')
    assert decision['difference']>0 and decision['selected_epoch']==23
    cfg=read('runs/round307/config.json')
    domain_path=Path('runs/round302_pretrain/final_encoder.pt')
    assert recipe.digest(domain_path)==cfg['domain_encoder_sha256']
    domain=torch.load(domain_path,map_location='cpu',weights_only=False)
    assert domain['split']==grouped and domain['epoch']==800
    targets=Path('runs/round308/full_targets')
    items=read(targets/'manifest.json')['items']
    assert len(items)==10356 and {(r['organ'],r['name']) for r in items}=={
        (o,n) for o in recipe.ORGANS for n in full[o]['train']}
    first=Path('runs/round324_stage1_smoke' if smoke else 'runs/round324_stage1')
    run=Path('runs/round324_smoke' if smoke else 'runs/round324')
    endpoint=15 if smoke else decision['selected_epoch']
    # Recompute all 14 full-data supervised epochs; do not load a prior
    # supervised checkpoint or choose an endpoint using full-data evaluation.
    recipe.fit(first,full,cfg,14,domain['encoder'],smoke=smoke)
    recipe.audit_fit(first,full,14,smoke=smoke)
    initial=first/'stage1_final.pt'
    recipe.fit(run,full,cfg,endpoint,domain['encoder'],initial=initial,targets=targets,
               smoke=smoke,consistency_weight=.2,paired_crop=True)
    recipe.audit_fit(run,full,endpoint,initial=True,smoke=smoke)
    recipe.write_json(run/'selection_split.json',grouped)
    recipe.write_json(run/'stage1_provenance.json',{
        'source':str(initial),'sha256':recipe.digest(initial),'full_train_images':10356,
        'stage1_epochs':14,'stage2_epochs':endpoint-14,'fresh_supervised_refit':True,
        'selection_round':323,'target_manifest_sha256':recipe.digest(targets/'manifest.json')})
    verify_refit(run,full,endpoint,smoke)
    if smoke:
        print('Fresh stage1 -> cropped stage2, optimizer and packaging provenance checks passed',flush=True)
        return
    from package_round302 import main as package
    package(324)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--log',action='store_true')
    args=p.parse_args(); os.chdir(recipe.ROOT)
    if args.log:
        sys.stdout=open('reports/round324.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round324.err','a',encoding='utf-8',buffering=1)
    try: main(args.smoke)
    except Exception:
        traceback.print_exc(); sys.exit(1)
