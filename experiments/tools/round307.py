"""Matched test of organ-weighted loss without oversampling or changing labels."""
import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from round303 import ROOT, run, organ_loss_weights
from round300 import ORGANS
from round305 import control_record


def audit(smoke=False):
    root=Path('runs/round307_smoke' if smoke else 'runs/round307')
    rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    count=2 if smoke else 28
    assert [r['epoch'] for r in rows]==list(range(1,count+1))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['validation_score']) for r in rows)
    if not smoke:
        assert all(r['samples']=={'colon':5004,'liver':1820,'stomach':1260} and r['updates']==506 for r in rows)
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    reference=json.loads(Path('runs/round303/config.json').read_text(encoding='utf-8'))
    compare={k:v for k,v in cfg.items() if k not in ('organ_loss_weights','organ_loss_objective')}
    compare['smoke']=False
    assert compare==reference
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    assert split==json.loads(Path('runs/round303/split.json').read_text(encoding='utf-8'))
    assert cfg['organ_loss_weights']==organ_loss_weights(split) and cfg['organ_loss_objective']==[.1,.2,.7]
    last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    steps={int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v}
    assert last['epoch']==count and steps==({2} if smoke else {14*506})
    checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    result=json.loads((root/f"validation_{checkpoint['epoch']:03d}.json").read_text(encoding='utf-8'))
    assert result['weighted_score']==max(r['validation_score'] for r in rows)
    return checkpoint['epoch'],result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--log',action='store_true')
    args=parser.parse_args(); os.chdir(ROOT)
    if args.log:
        sys.stdout=open('reports/round307.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round307.err','a',encoding='utf-8',buffering=1)
    try:
        run(SimpleNamespace(smoke=args.smoke,resume=True,full_train=False,
                            run_prefix='round307',score_weighted_loss=True))
        epoch,result=audit(args.smoke)
        if args.smoke:
            print('Weighted-loss two-stage smoke passed',flush=True)
        else:
            baseline=json.loads(Path('runs/round303/complete.json').read_text(encoding='utf-8'))['best_validation_score']
            decision={'baseline_round303':baseline,'candidate_round307':result['weighted_score'],
                      'difference':result['weighted_score']-baseline,'selected_epoch':epoch,
                      'package':result['weighted_score']>baseline,'organs':result['organs'],
                      'selection':'best weighted grouped Val only; no additional ROI threshold',
                      'limitation':'one split with3 Stomach ROIs; validate gains before any full refit'}
            control_record(Path('reports/round307_decision.json'),decision)
            print(json.dumps(decision),flush=True)
            if decision['package']:
                manifest=Path('reports/round307_submission.json')
                if not manifest.exists():
                    from package_round302 import main
                    main(307)
                else:
                    existing=json.loads(manifest.read_text(encoding='utf-8'))
                    assert hashlib.sha256(Path(existing['package']).read_bytes()).hexdigest()==existing['sha256']
            control_record(Path('reports/round307_complete.json'),{'packaged':decision['package']})
    except Exception:
        traceback.print_exc(); sys.exit(1)
