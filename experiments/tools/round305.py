"""One queued CD68 comparison, then a full refit only if grouped Val improves."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from round303 import ROOT, run
from round300 import ORGANS, write_json


def control_record(path, value):
    temporary=path.with_suffix(path.suffix+'.writing')
    write_json(temporary,value)
    temporary.replace(path)


def recipe_args(full=False, endpoint=27, smoke=False):
    return SimpleNamespace(smoke=smoke, resume=True, full_train=full,
                           organ_markers=['CD68']*3, run_prefix='round306' if full else 'round305',
                           endpoint=endpoint, selected_by=f'Round305 grouped Val epoch{endpoint}; both cosine horizons14')


def choose(baseline, candidate):
    """Fixed Val criterion; no leaderboard, fold or ROI threshold."""
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError('Nonfinite validation score')
    return candidate > baseline


def audit(run_name, smoke=False, full=False, endpoint=28):
    root=Path('runs')/(run_name+('_smoke' if smoke else ''))
    count=2 if smoke else endpoint
    rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,count+1))
    counts={'colon':6296,'liver':2380,'stomach':1680} if full else {'colon':5004,'liver':1820,'stomach':1260}
    if not smoke:
        assert all(r['samples']==counts and r['updates']==math.ceil(sum(counts.values())/16) for r in rows)
    assert all(math.isfinite(r['loss']) for r in rows)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    assert cfg['supervised_markers']=={o:['CD68'] for o in ORGANS}
    ck=torch.load(root/('final.pt' if full else 'best.pt'),map_location='cpu',weights_only=False)
    assert ck['markers']=={o:'CD68' for o in ORGANS}
    assert all(torch.isfinite(v).all() for v in ck['model'].values())
    last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    steps={int(s['step']) for s in last['optimizer']['state'].values() if 'step' in s}
    stage_length=1 if smoke else 14
    stage_epoch=1+(count-1)%stage_length
    expected=(2 if smoke else math.ceil(sum(counts.values())/16))*stage_epoch
    assert last['epoch']==count and steps=={expected}
    if full:
        assert ck['epoch']==count
        assert all(torch.equal(v,last['model'][k]) for k,v in ck['model'].items())
        return {'epoch':count,'updates':sum(r['updates'] for r in rows)}
    validation=json.loads((root/f"validation_{ck['epoch']:03d}.json").read_text(encoding='utf-8'))
    assert validation['weighted_score']==max(r['validation_score'] for r in rows)
    return {'epoch':ck['epoch'],'validation':validation}


def wait_dependency():
    deadline=time.monotonic()+12*3600
    while not Path('reports/round304_submission.json').exists():
        error=Path('reports/round304.err')
        if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):
            raise RuntimeError('Round304 failed; preserve that training before starting the queued comparison')
        if time.monotonic()>deadline: raise TimeoutError('Round304 dependency exceeded12h')
        time.sleep(30)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(hardware.strip())<4096: raise RuntimeError('Insufficient free GPU memory at phase boundary')


def pipeline(smoke=False):
    torch.set_num_threads(4)
    if not smoke: wait_dependency()
    run(recipe_args(smoke=smoke))
    candidate=audit('round305',smoke=smoke)
    if smoke:
        run(recipe_args(full=True,smoke=True))
        audit('round306',smoke=True,full=True,endpoint=2)
        print('Both queued training paths passed smoke',flush=True)
        return
    cfg=json.loads(Path('runs/round305/config.json').read_text(encoding='utf-8'))
    base_cfg=json.loads(Path('runs/round303/config.json').read_text(encoding='utf-8'))
    assert {k:v for k,v in cfg.items() if k!='supervised_markers'}=={k:v for k,v in base_cfg.items() if k!='supervised_markers'}
    assert Path('runs/round305/split.json').read_bytes()==Path('runs/round303/split.json').read_bytes()
    baseline=json.loads(Path('runs/round303/complete.json').read_text(encoding='utf-8'))['best_validation_score']
    score=candidate['validation']['weighted_score']
    decision={'baseline_round303':baseline,'candidate_round305':score,'difference':score-baseline,
              'selected_epoch':candidate['epoch'],'refit_cd68':choose(baseline,score),
              'selection':'best weighted grouped Val; exact tie retains Round304; no Test feedback',
              'candidate_organs':candidate['validation']['organs'],
              'limitation':'one split,3 Stomach ROIs; no claim of uniform ROI improvement'}
    control_record(Path('reports/round305_decision.json'),decision)
    print(json.dumps(decision),flush=True)
    if decision['refit_cd68']:
        run(recipe_args(full=True,endpoint=candidate['epoch']))
        result=audit('round306',full=True,endpoint=candidate['epoch'])
        write_json(Path('reports/round306_audit.json'),result)
        from package_round302 import main
        manifest=Path('reports/round306_submission.json')
        if not manifest.exists(): main(306)
        else:
            import hashlib
            packaged=json.loads(manifest.read_text(encoding='utf-8'))
            assert hashlib.sha256(Path(packaged['package']).read_bytes()).hexdigest()==packaged['sha256']
    control_record(Path('reports/round305_queue_complete.json'),{'full_cd68':decision['refit_cd68']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--log',action='store_true')
    args=parser.parse_args(); os.chdir(ROOT)
    if args.log:
        sys.stdout=open('reports/round305_queue.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round305_queue.err','a',encoding='utf-8',buffering=1)
    try: pipeline(args.smoke)
    except Exception:
        traceback.print_exc(); sys.exit(1)
