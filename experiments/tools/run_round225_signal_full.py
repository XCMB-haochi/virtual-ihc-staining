"""Fixed full-data auxiliary-task confirmation without pixel distillation."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round208_full_domain import supervised_command
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest,run_logged


def command(stage,smoke=False):
    cmd=supervised_command(stage)
    prefix='runs/round225_smoke' if smoke else 'runs/round225_full'
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd[cmd.index('--model-arch')+1]='unext2-fcmae-tiny-signal-aux'
    cmd+=['--signal-auxiliary-weight','0.05']
    if stage==2:
        cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    if smoke:
        cmd[cmd.index('--epochs')+1]='1'
        cmd+=['--max-train-images','128']
    return cmd


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    encoder=Path('runs/round208_full_fcmae800/final_encoder.pt')
    reference=torch.load('runs/round208_full_stage1/epochs/epoch_014.pt',map_location='cpu',weights_only=False)
    assert digest(encoder)==reference['args']['init_checkpoint_report']['checkpoint_sha256']
    for stage in (1,2):
        cmd=command(stage,args.smoke);root=Path(cmd[cmd.index('--output-dir')+1])
        if root.exists():raise FileExistsError(root)
        print('Training full-data signal auxiliary',stage,flush=True)
        run_logged(cmd,f"reports/round225_{'smoke' if args.smoke else 'full'}_stage{stage}.log")
        rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,(1 if args.smoke else 14)+1))
        assert all(math.isfinite(r['train_loss']) and math.isfinite(r['train_signal_auxiliary_loss']) for r in rows)
        assert any(r['train_signal_auxiliary_loss']>0 for r in rows)
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        steps=[float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
        assert ck['args']['train_all'] and ck['args']['distillation_weight']==0 and ck['args']['signal_auxiliary_weight']==.05
        assert ck['args']['selection_marker']=='CD45RO'
        print('Verified',stage,'updates',min(steps),'peak GiB',rows[-1]['peak_gpu_gib'],flush=True)
    if args.smoke:return
    checkpoint='runs/round225_full_stage2/epochs/epoch_012.pt'
    p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert p['epoch']==12 and all(torch.isfinite(v).all() for v in p['model'].values())
    pred='predictions/round225_cd45ro_q95';package='submissions/semifinal_round225_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Output exists')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,checkpoint,pred,'--marker','CD45RO',
        '--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,f'{DATASET}/test/DAPI',package,'--marker','CD45RO'],check=True)
    subprocess.run([sys.executable,'-m','zipfile','-t',package],check=True)
    Path('reports/round225_hashes.json').write_text(json.dumps({checkpoint:digest(checkpoint),package:digest(package)},indent=2)+'\n')


if __name__=='__main__':main()
