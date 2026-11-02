"""Fixed five-fold stochastic-depth transfer screen on the domain baseline."""
import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET,training_command
from run_round224_cv5 import device_environment
from run_round210_relational import digest,run_logged


def command(fold,stage,smoke=False):
    cmd,_=training_command(fold,stage,14,smoke)
    prefix='runs/round231_smoke' if smoke else f"runs/round231_{fold['name']}"
    source='round207' if fold['name'] in ('fold0','fold3') else 'round216'
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    if stage==1:cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/{source}_{fold['name']}_fcmae800/final_encoder.pt"
    else:cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    cmd+=['--encoder-drop-path-rate','0.2']
    return cmd


def worker(folds,gpu):
    for fold in folds:
        for stage in (1,2):
            print('Training DropPath',fold['name'],stage,'GPU',gpu,flush=True)
            with Path(f"reports/round231_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(command(fold,stage),env=device_environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    if args.smoke:
        for stage in (1,2):
            cmd=command(folds[0],stage,True)
            print('Smoke',stage,flush=True)
            run_logged(cmd,f'reports/round231_smoke_stage{stage}.log')
            root=Path(cmd[cmd.index('--output-dir')+1]);ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v];assert min(steps)>0
            assert ck['args']['encoder_drop_path_rate']==.2
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            print('Verified updates',min(steps),'peak',rows[-1]['peak_gpu_gib'],flush=True)
        return
    for fold in folds:
        for stage in (1,2):
            cmd=command(fold,stage)
            if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing trajectory')
            if stage==1:
                path=Path(cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1])
                prefix='round207' if fold['name'] in ('fold0','fold3') else 'round216'
                baseline=torch.load(f"runs/{prefix}_{fold['name']}_stage1/epochs/epoch_014.pt",map_location='cpu',weights_only=False)
                assert digest(path)==baseline['args']['init_checkpoint_report']['checkpoint_sha256']
    # GPU1 has competing compute at this launch; give it one shorter fold.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,folds[:4],0),pool.submit(worker,[folds[4]],1)]
        for job in concurrent.futures.as_completed(jobs):job.result()
    sources=[]
    for fold in folds:
        for stage in (1,2):
            root=Path(f"runs/round231_{fold['name']}_stage{stage}")
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,15)) and all(math.isfinite(r['train_loss']) for r in rows)
            ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            assert ck['args']['encoder_drop_path_rate']==.2 and ck['args']['seed']==2026
            steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step'in v];assert min(steps)>0
        path=root/'epochs'/'epoch_012.pt'
        sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path)})
    root=Path('runs/round231_oof_sources');root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','images':2380,
        'output_source_offset_applied':False,'folds':sources},indent=2)+'\n')
    output=Path('reports/round231_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),'--mode','models','--device','cuda:0'],env=device_environment(0),check=True)
    current=json.loads(output.read_text())['markers'];base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains={m:{s:v['gaussian11_score']-base[m][s]['gaussian11_score'] for s,v in scopes.items()} for m,scopes in current.items()}
    Path('reports/round231_cv5_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('DropPath full-fold CD45RO gains',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()
