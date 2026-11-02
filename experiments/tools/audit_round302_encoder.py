"""Check the800-epoch dependency before starting supervised transfer."""
import json
import math
import hashlib
from pathlib import Path
import torch
from round300 import inventory

root=Path('runs/round302_pretrain')
rows=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in rows]==list(range(1,801))
assert all(r['images']==8084 and r['updates']==253 and math.isfinite(r['loss']) for r in rows)
encoder=torch.load(root/'final_encoder.pt',map_location='cpu',weights_only=False)
assert encoder['epoch']==800 and not encoder['config']['benchmark']
assert encoder['split']==inventory(Path('dataset')) and encoder['config']['seed']==2026
assert all(torch.isfinite(v).all() for v in encoder['encoder'].values())
last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
steps={int(v['step']) for v in last['optimizer']['state'].values() if 'step' in v}
assert last['epoch']==800 and steps=={202400},steps
result={'epochs':800,'images_seen':8084*800,'optimizer_steps':202400,'val_excluded':True,
        'seconds':sum(r['seconds'] for r in rows),
        'encoder_sha256':hashlib.sha256((root/'final_encoder.pt').read_bytes()).hexdigest()}
Path('reports/round302_pretrain_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
