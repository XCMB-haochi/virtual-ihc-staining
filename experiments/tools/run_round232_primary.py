"""Keep four-output capacity fixed and ablate auxiliary-marker supervision."""
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
    prefix='runs/round232_smoke' if smoke else f"runs/round232_{fold['name']}"
    source='round207' if fold['name'] in ('fold0','fold3') else 'round216'
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd[cmd.index('--marker-loss-weights')+1]='0,1,0,0'
    if stage==1:cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/{source}_{fold['name']}_fcmae800/final_encoder.pt"
    else:cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    return cmd


def worker(folds,gpu):
    for fold in folds:
        for stage in (1,2):
            print('Training CD45RO-only objective',fold['name'],stage,'GPU',gpu,flush=True)
            with Path(f"reports/round232_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(command(fold,stage),env=device_environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    if args.smoke:
        for stage in (1,2):
            cmd=command(folds[0],stage,True)
            run_logged(cmd,f'reports/round232_smoke_stage{stage}.log')
            root=Path(cmd[cmd.index('--output-dir')+1]);p=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            assert p['args']['marker_loss_weights_resolved']==[0.,1.,0.,0.] and p['markers'][1]=='CD45RO'
            steps=[float(v['step']) for v in p['optimizer']['state'].values() if 'step'in v];assert min(steps)>0
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            print('Smoke',stage,'actual updates',min(steps),'peak',rows[-1]['peak_gpu_gib'],flush=True)
        return
    for fold in folds:
        for stage in (1,2):
            cmd=command(fold,stage)
            if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing trajectory')
            if stage==1:
                encoder=Path(cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1])
                prefix='round207' if fold['name'] in ('fold0','fold3') else 'round216'
                reference=torch.load(f"runs/{prefix}_{fold['name']}_stage1/epochs/epoch_014.pt",map_location='cpu',weights_only=False)
                assert digest(encoder)==reference['args']['init_checkpoint_report']['checkpoint_sha256']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,[folds[0],folds[3]],0),pool.submit(worker,[folds[1],folds[2],folds[4]],1)]
        for job in concurrent.futures.as_completed(jobs):job.result()
    sources=[]
    for fold in folds:
        for stage in (1,2):
            root=Path(f"runs/round232_{fold['name']}_stage{stage}")
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,15)) and all(math.isfinite(r['train_loss']) for r in rows)
            ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            assert ck['args']['marker_loss_weights_resolved']==[0.,1.,0.,0.] and ck['args']['seed']==2026
            steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step'in v];assert min(steps)>0
        ck=root/'epochs'/'epoch_012.pt'
        sources.append({**fold,'images':len(fold['val_rois'])*140,'checkpoint':str(ck.resolve()),'checkpoint_sha256':digest(ck)})
    directory=Path('runs/round232_oof_sources');directory.mkdir(exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','images':2380,
        'output_source_offset_applied':False,'folds':sources},indent=2)+'\n')
    output=Path('reports/round232_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),str(output),'--mode','models','--device','cuda:0'],env=device_environment(1),check=True)
    current=json.loads(output.read_text())['markers']['CD45RO'];base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']['CD45RO']
    gains={scope:r['gaussian11_score']-base[scope]['gaussian11_score'] for scope,r in current.items()}
    Path('reports/round232_cd45ro_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('CD45RO-only objective gains',json.dumps(gains),flush=True)


if __name__=='__main__':main()
