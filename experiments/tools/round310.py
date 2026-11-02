"""Matched A100 stage2 control and mild training-only DAPI gain intervention."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import round308 as recipe
from round300 import preload, read_image
from round305 import control_record


def portable_preload(split, policy):
    portable=copy.deepcopy(split)
    for spec in portable.values(): spec['folder']=spec['folder'].replace('\\','/')
    return preload(portable,policy)


def main(args):
    os.chdir(recipe.ROOT); torch.set_num_threads(4)
    bundle=Path('inputs/round310')
    cfg=json.loads((bundle/'base_config.json').read_text(encoding='utf-8'))
    split=json.loads((bundle/'split.json').read_text(encoding='utf-8'))
    expected=json.loads((bundle/'artifacts.json').read_text(encoding='utf-8'))
    for filename,sha in expected.items(): assert recipe.digest(bundle/filename)==sha,filename
    encoder_path=bundle/'final_encoder.pt'
    assert recipe.digest(encoder_path)==cfg['domain_encoder_sha256']
    domain=torch.load(encoder_path,map_location='cpu',weights_only=False)
    assert domain['split']==split and domain['epoch']==800
    seed_state=torch.load(bundle/'stage1_final.pt',map_location='cpu',weights_only=False)
    assert seed_state['epoch']==14
    del seed_state
    files=json.loads((bundle/'files.json').read_text(encoding='utf-8'))
    for item in files:
        path=Path(item['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert recipe.digest(path)==item['sha256'],str(path)
        if 'decoded_sha256' in item:
            assert hashlib.sha256(read_image(path).tobytes()).hexdigest()==item['decoded_sha256']
    assert len(files)==2*sum(len(s['train'])+len(s['val']) for s in split.values())==20712
    if args.check:
        print(json.dumps({'files_verified':len(files),'split_verified':True,'encoder_verified':True}),flush=True); return
    assert torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
    index=310 if args.arm=='control' else 311
    root=Path(f'runs/round{index}'+('_smoke' if args.smoke else ''))
    recipe.preload=portable_preload
    begin=time.monotonic()
    endpoint=15 if args.smoke else 28
    recipe.fit(root,split,cfg,endpoint,domain['encoder'],initial=bundle/'stage1_final.pt',
               validate=True,smoke=args.smoke,dapi_gain_range=None if args.arm=='control' else (.9,1.1))
    recipe.audit_fit(root,split,endpoint,initial=True,smoke=args.smoke)
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    result=json.loads((root/f"validation_{ck['epoch']:03d}.json").read_text(encoding='utf-8'))
    report={'arm':args.arm,'round':index,'epoch':ck['epoch'],'weighted_score':result['weighted_score'],
            'organs':result['organs'],'seconds':time.monotonic()-begin,'smoke':args.smoke,
            'source_stage1_sha256':expected['stage1_final.pt'],
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'training_only_gain':None if args.arm=='control' else [.9,1.1],
            'checkpoint_sha256':recipe.digest(root/'best.pt')}
    control_record(Path(f'reports/round{index}'+('_smoke' if args.smoke else '')+'_result.json'),report)
    print(json.dumps(report),flush=True)
    if not args.smoke: compare_if_ready()


def compare_if_ready():
    a=Path('reports/round310_result.json'); b=Path('reports/round311_result.json')
    if not (a.exists() and b.exists()): return
    control=json.loads(a.read_text(encoding='utf-8')); candidate=json.loads(b.read_text(encoding='utf-8'))
    assert control['source_stage1_sha256']==candidate['source_stage1_sha256']
    control_cfg=json.loads(Path('runs/round310/config.json').read_text(encoding='utf-8'))
    candidate_cfg=json.loads(Path('runs/round311/config.json').read_text(encoding='utf-8'))
    assert {k:v for k,v in candidate_cfg.items() if k!='training_dapi_gain_range'}==control_cfg
    baseline=json.loads(Path('inputs/round310/base_complete.json').read_text(encoding='utf-8'))['best_validation_score']
    decision={'control':control['weighted_score'],'gain_augmented':candidate['weighted_score'],
              'gain_vs_matched_control':candidate['weighted_score']-control['weighted_score'],
              'original_round307':baseline,'gain_vs_original':candidate['weighted_score']-baseline,
              'candidate_for_review':candidate['weighted_score']>max(control['weighted_score'],baseline),
              'scope':'Train/Val only; input gain used in training only; no automated submission/refit'}
    control_record(Path('reports/round311_comparison.json'),decision)
    print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('control','gain'),required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--check',action='store_true')
    main(p.parse_args())
