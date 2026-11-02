"""Full-Train refit of the Round303 Val-selected recipe, followed by packaging."""
import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
import torch
from round303 import ROOT, run


def audit(smoke=False):
    root=Path('runs/round304_smoke' if smoke else 'runs/round304')
    count=2 if smoke else 27
    rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,count+1))
    assert all(math.isfinite(r['loss']) and r['validation_score'] is None for r in rows)
    if not smoke:
        assert all(r['samples']=={'colon':6296,'liver':2380,'stomach':1680} and r['updates']==648 for r in rows)
    last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
    steps={int(s['step']) for s in last['optimizer']['state'].values() if 'step' in s}
    assert last['epoch']==count and steps==({2} if smoke else {648*13})
    checkpoint=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==count and checkpoint['config']['full_train']
    assert all(torch.isfinite(p).all() for p in checkpoint['model'].values())
    assert all(torch.equal(checkpoint['model'][key],last['model'][key]) for key in checkpoint['model'])
    return {'epochs':count,'samples':sum(sum(r['samples'].values()) for r in rows),
            'optimizer_updates':sum(r['updates'] for r in rows),'finite_weights':True,
            'endpoint':'fixed by Round303 Val, no full-data checkpoint selection'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--log',action='store_true')
    args=parser.parse_args(); args.full_train=True; os.chdir(ROOT)
    if args.log:
        sys.stdout=open('reports/round304.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round304.err','a',encoding='utf-8',buffering=1)
    try:
        run(args)
        result=audit(args.smoke)
        Path(f'reports/round304{"_smoke" if args.smoke else ""}_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
        if not args.smoke:
            from package_round302 import main
            main(304)
    except Exception:
        traceback.print_exc(); sys.exit(1)
