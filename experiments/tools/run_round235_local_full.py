"""One fixed full-data local-condition candidate, preserving Round216 distillation."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round208_full_domain import supervised_command
from run_round216_distill import student_command
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest,run_logged


def command(stage,smoke=False):
    cmd=supervised_command(1) if stage==1 else student_command()
    prefix='runs/round235_smoke' if smoke else 'runs/round235_full'
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd[cmd.index('--model-arch')+1]='unext2-fcmae-tiny-local-condition'
    if stage==2:
        cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    if smoke:
        cmd[cmd.index('--epochs')+1]='1'
        cmd+=['--max-train-images','128']
    return cmd


def assess(gains):
    fold_keys=[f'fold{i}' for i in range(5)]
    if not all(k in gains for k in ['overall',*fold_keys]):
        raise ValueError('Incomplete fold results')
    positive=sum(gains[k]>0 for k in fold_keys)
    return {'mean_gain':gains['overall'],'positive_folds':positive,
            'train_candidate':gains['overall']>=.05 and positive>=3,
            'rule':'One fixed confirmation if mean >=0.05 and at least3 folds positive; not automatic promotion.'}


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if not args.smoke:
        gains=json.loads(Path('reports/round234_cv5_gains.json').read_text())['CD45RO']
        report=json.loads(Path('reports/round234_cv5_oof.json').read_text())
        assert report['markers']['CD45RO']['overall']['images']==2380
        decision=assess(gains)
        Path('reports/round235_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        print('Confirmation decision',json.dumps(decision),flush=True)
        if not decision['train_candidate']:
            print('No automatic full-data run: inspect complete fold/ROI evidence.',flush=True)
            return
    cache=Path('runs/round216_oof_targets')
    m=json.loads((cache/'manifest.json').read_text());assert m['marker']=='CD45RO' and m['images']==2380
    names={p.name for p in (Path(DATASET)/'train'/'DAPI').glob('*.jpg')}
    assert len(names)==2380 and names=={p.name for p in (cache/'targets').glob('*.jpg')}
    reference=torch.load('runs/round208_full_stage1/epochs/epoch_014.pt',map_location='cpu',weights_only=False)
    encoder=Path('runs/round208_full_fcmae800/final_encoder.pt')
    assert digest(encoder)==reference['args']['init_checkpoint_report']['checkpoint_sha256']
    for stage in (1,2):
        cmd=command(stage,args.smoke);output=Path(cmd[cmd.index('--output-dir')+1])
        if output.exists():raise FileExistsError(output)
        print('Training local-condition full-data',stage,flush=True)
        run_logged(cmd,f"reports/round235_{'smoke' if args.smoke else 'full'}_stage{stage}.log")
        rows=[json.loads(l) for l in (output/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==(1 if args.smoke else 14) and all(math.isfinite(r['train_loss']) for r in rows)
        last=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        steps=[float(s['step']) for s in last['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
        assert last['args']['model_arch']=='unext2-fcmae-tiny-local-condition'
        if stage==2:assert last['args']['distillation_marker_index']==1 and last['args']['distillation_weight']==.2
        print('Verified stage',stage,'updates',min(steps),'peak GiB',rows[-1]['peak_gpu_gib'],flush=True)
    if args.smoke:return
    checkpoint='runs/round235_full_stage2/epochs/epoch_012.pt'
    pred='predictions/round235_cd45ro_q95';package='submissions/semifinal_round235_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Output exists')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,checkpoint,pred,'--marker','CD45RO',
        '--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,f'{DATASET}/test/DAPI',package,'--marker','CD45RO'],check=True)
    subprocess.run([sys.executable,'-m','zipfile','-t',package],check=True)
    Path('reports/round235_hashes.json').write_text(json.dumps({checkpoint:digest(checkpoint),package:digest(package)},indent=2)+'\n')


if __name__=='__main__':main()
