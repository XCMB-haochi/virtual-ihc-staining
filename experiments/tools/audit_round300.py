"""Audit the completed baseline before a matched budget experiment."""
import json
from pathlib import Path
import torch

root=Path('runs/round300')
rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in rows]==list(range(1,49))
assert all(all(not isinstance(v,float) or __import__('math').isfinite(v) for v in r.values()) for r in rows)
last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
best=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
assert last['epoch']==best['epoch']==48
assert all(torch.isfinite(v).all() for v in best['model'].values())
steps={int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v}
assert steps=={12288},steps
result={'epochs':48,'best_epoch':best['epoch'],'optimizer_steps':sorted(steps),
        'training_seconds':sum(r['train_seconds'] for r in rows),
        'validation_seconds':sum(r.get('validation_seconds',0) for r in rows),
        'seed':best['config']['seed'],'markers':best['markers']}
assert result['seed']==2026
Path('reports/round300_completion_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
